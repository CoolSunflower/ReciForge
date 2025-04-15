#!/usr/bin/env python
"""
inference.py

This script performs ensemble inference for fine-tuning a QoR (Area) predictor on a new design.
Usage:
    python inference.py {design_name} "command1" "command2" ... [--initial_area INITIAL_AREA]

It:
  1. Loads the design's BENCH file from the designs/ folder.
  2. Loads the corresponding dataset CSV from datasets/ (if needed for further fine-tuning).
  3. Determines the “initial area” using a predefined BASE_AREAS dictionary (if available) or the user-provided value.
  4. Loads all 8 finetuned models from weights/ (files named "{model_name}_finetuned.pt").
  5. Runs inference on the finetuned ensemble using the provided recipe:
       - Case 1 (design has a finetuned model): Uses a weighted average (79% for the main model, 3% each for the others).
       - Case 2 (design has no dedicated finetuned model): Uses an arithmetic average.
  6. Prints the predicted area after each synthesis step.

Ensure that the folders "designs/", "datasets/", and "weights/" exist and contain the appropriate files.
"""

import os
import sys
import argparse
import re
import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data

# BASE_AREAS dictionary: if a design is in this dictionary, use its defined initial area.
BASE_AREAS = {
    'apex1': 1577,
    'bc0': 1592,
    'c6288': 2337,
    'c7552': 2198,
    'i2c': 1169,
    'max': 2865,
    'sasc': 613,
    'simple_spi': 930,
}

# List of valid synthesis commands (as used in training).
VALID_COMMANDS = [
    "rewrite -z", "rewrite -l", "rewrite", "balance",
    "resub", "refactor", "resub -z", "refactor -z"
]

def determine_initial_area(design_name, user_initial_area=None):
    """
    Determine the initial area for a design.
    If design_name is in BASE_AREAS, return that value; otherwise, use the provided user_initial_area.
    """
    if design_name in BASE_AREAS:
        return BASE_AREAS[design_name]
    elif user_initial_area is not None:
        try:
            return float(user_initial_area)
        except ValueError:
            print("Error: Provided initial_area must be a number.")
            sys.exit(1)
    else:
        print("Error: Design not in BASE_AREAS. You must supply an initial_area via --initial_area.")
        sys.exit(1)

def validate_file(path, description):
    """Exit if the given file path does not exist."""
    if not os.path.exists(path):
        print(f"Error: {description} not found at '{path}'.")
        sys.exit(1)
    return path

def validate_recipe(recipe):
    """Ensure all commands in the recipe are valid."""
    invalid = [cmd for cmd in recipe if cmd not in VALID_COMMANDS]
    if invalid:
        print(f"Error: Invalid commands found in recipe: {invalid}.")
        print(f"Valid commands are: {VALID_COMMANDS}")
        sys.exit(1)
    return recipe

def parse_bench_file(file_path):
    """
    Parse a BENCH file into a networkx DiGraph.
    Each node (INPUT, OUTPUT, AND, NOT) is added; levels, fanin and fanout are computed.
    """
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("INPUT("):
                node = re.search(r"INPUT\((.*?)\)", line).group(1)
                G.add_node(node, type="INPUT", level=0)
            elif line.startswith("OUTPUT("):
                node = re.search(r"OUTPUT\((.*?)\)", line).group(1)
                G.add_node(node, type="OUTPUT")
            elif "=" in line:
                parts = line.split("=")
                target = parts[0].strip()
                expression = parts[1].strip()
                if "AND(" in expression:
                    gate_type = "AND"
                    inputs_str = re.search(r"AND\((.*?)\)", expression).group(1)
                    inputs = [inp.strip() for inp in inputs_str.split(",")]
                elif "NOT(" in expression:
                    gate_type = "NOT"
                    inputs = [re.search(r"NOT\((.*?)\)", expression).group(1).strip()]
                else:
                    continue
                G.add_node(target, type=gate_type)
                for inp in inputs:
                    G.add_edge(inp, target)
    # Compute levels and fanin/fanout.
    for node in nx.topological_sort(G):
        if G.nodes[node].get('type') == "INPUT":
            G.nodes[node]['level'] = 0
        else:
            preds = list(G.predecessors(node))
            level = max([G.nodes[p]['level'] for p in preds]) + 1 if preds else 0
            G.nodes[node]['level'] = level
    for node in G.nodes():
        G.nodes[node]['fanin'] = G.in_degree(node)
        G.nodes[node]['fanout'] = G.out_degree(node)
    return G

def extract_circuit_features(G):
    """
    Given a networkx graph G for a circuit, extract node features and edge indices.
    Returns a dictionary with keys:
       - "x": tensor of node features.
       - "edge_index": tensor of edge indices.
    Each feature vector is a 7-dimensional vector:
       4 for one-hot encoding of node type (INPUT, OUTPUT, AND, NOT) plus level, fanin, fanout.
    """
    type_mapping = {"INPUT": 0, "OUTPUT": 1, "AND": 2, "NOT": 3}
    features = []
    nodes = list(G.nodes())
    for node in nodes:
        attr = G.nodes[node]
        one_hot = [0, 0, 0, 0]
        one_hot[type_mapping[attr['type']]] = 1
        level = attr.get('level', 0)
        fanin = attr.get('fanin', 0)
        fanout = attr.get('fanout', 0)
        features.append(one_hot + [level, fanin, fanout])
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    edge_index = []
    for src, tgt in G.edges():
        edge_index.append([node_to_idx[src], node_to_idx[tgt]])
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return {"x": x, "edge_index": edge_index}

def encode_recipe(recipe):
    """
    Encode the provided synthesis recipe (list of commands) as a tensor of one-hot vectors.
    Each command is encoded into an 8-dim vector according to VALID_COMMANDS.
    The returned tensor shape is [seq_len, 8].
    """
    cmd_to_idx = {cmd: i for i, cmd in enumerate(VALID_COMMANDS)}
    encoded = []
    for cmd in recipe:
        one_hot = [0] * len(VALID_COMMANDS)
        one_hot[cmd_to_idx[cmd]] = 1
        encoded.append(one_hot)
    return torch.tensor(encoded, dtype=torch.float)

def load_all_finetuned_models(device):
    """
    Load all 8 finetuned models from the weights folder.
    Models are expected to be stored as weights/{model_name}_finetuned.pt.
    Returns a dictionary mapping model names to instances of CircuitQoRModel.
    """
    from main import CircuitQoRModel  # Import the model class
    model_names = ['apex1', 'bc0', 'c6288', 'c7552', 'i2c', 'max', 'sasc', 'simple_spi']
    models = {}
    for name in model_names:
        weights_path = f'weights/{name}.pt'
        if os.path.exists(weights_path):
            model = CircuitQoRModel(node_feature_dim=7, recipe_dim=8, hidden_dim=128)
            state_dict = torch.load(weights_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models[name] = model
    return models

def ensemble_inference(design_name, recipe_list, device, initial_area):
    """
    Run inference using all 8 finetuned models on the provided design and recipe.
    There are two cases:
      1) If a model exists for design_name (i.e. design_name in the ensemble dictionary),
         use weighted ensemble: the "main" model (for the design) gets weight 0.79, the other 7 get weight 0.03 each.
      2) Otherwise, take the simple arithmetic average of the 8 models.
    Print the predicted area after each synthesis step.
    """
    # Validate that the design (BENCH file) exists.
    design_path = validate_file(f"designs/{design_name}.bench", f"Design BENCH file for '{design_name}'")
    # Parse the BENCH file into a graph.
    circuit_graph = parse_bench_file(design_path)
    # Extract features.
    features = extract_circuit_features(circuit_graph)
    # Convert to a torch_geometric Data object.
    circuit_data = Data(x=features["x"], edge_index=features["edge_index"])
    circuit_data = circuit_data.to(device)
    # Encode the recipe.
    recipe_encoded = encode_recipe(recipe_list)
    recipe_encoded = recipe_encoded.unsqueeze(0).to(device)  # shape: [1, seq_len, 8]
    
    # Load all 8 finetuned models.
    models = load_all_finetuned_models(device)
    if len(models) < 1:
        print("Error: No finetuned models found in weights/ folder.")
        sys.exit(1)
    
    # Run inference with each model.
    predictions = {}  # key: model name, value: tuple (final_area, step_areas, uncertainty)
    with torch.no_grad():
        for name, model in models.items():
            final_area, step_areas, uncertainty = model(circuit_data, recipe_encoded)
            # Squeeze the batch dimension.
            step_areas = step_areas.squeeze(0).cpu().numpy()
            final_area = final_area.item()
            uncertainty = uncertainty.item()
            predictions[name] = (final_area, step_areas, uncertainty)
    
    # Ensemble the predictions.
    seq_len = predictions[list(predictions.keys())[0]][1].shape[0]
    combined_steps = np.zeros(seq_len)
    combined_final = 0.0
    combined_uncertainty = 0.0

    if design_name in models:
        # Case 1: A finetuned model for this design exists.
        # Use weight 0.79 for the main design model and 0.03 for each of the other 7.
        main_weight = 0.79
        other_weight = 0.03
        for name, (f_area, s_areas, unc) in predictions.items():
            weight = main_weight if name == design_name else other_weight
            combined_steps += weight * s_areas
            combined_final += weight * f_area
            combined_uncertainty += weight * unc
    else:
        # Case 2: No dedicated finetuned model for this design, take average.
        num_models = len(predictions)
        for (f_area, s_areas, unc) in predictions.values():
            combined_steps += s_areas / num_models
            combined_final += f_area / num_models
            combined_uncertainty += unc / num_models

    # Print out the results.
    print(f"Initial Area: {round(initial_area)}")
    for i, cmd in enumerate(recipe_list):
        print(f"Predicted area after step {i+1} ('{cmd}'): {round(combined_steps[i] * (initial_area//2))}")
    # print(f"\nFinal predicted area: {combined_final}")
    print(f"Ensemble uncertainty (1-sigma): {combined_uncertainty}")

def main():
    parser = argparse.ArgumentParser(description="Ensemble QoR (Area) Predictor Inference")
    parser.add_argument("design_name", type=str,
                        help="Name of the design (look for designs/{design_name}.bench)")
    parser.add_argument("recipe", type=str, nargs="+",
                        help="Synthesis recipe as a sequence of commands (space separated)")
    parser.add_argument("--initial_area", type=float, default=None,
                        help="Optional initial area (only used if design not in BASE_AREAS)")
    args = parser.parse_args()

    # Validate recipe.
    validate_recipe(args.recipe)
    # Determine the initial area.
    init_area = determine_initial_area(args.design_name, args.initial_area)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run ensemble inference.
    ensemble_inference(args.design_name, args.recipe, device, init_area)

if __name__ == "__main__":
    main()
