# #!/usr/bin/env python
# """
# finetune.py

# Fine-tunes a pre-trained QoR (Area) predictor on a new design.
# Usage: python finetune.py {design_name} {initial_area}
# This script looks for the design file in "designs/{design_name}.bench" and the dataset CSV in "datasets/{design_name}.csv".
# It then uses all already-trained models (in weights/ folder) to make inferences on the new training dataset,
# selects the model with the lowest inference error, fine-tunes it on the new dataset, and saves the fine-tuned weights.
# """

# import os
# import sys
# import argparse
# import torch
# import numpy as np
# from torch.utils.data import DataLoader, random_split

# from main import CircuitDataset, collate_fn, train_model, CircuitQoRModel
# from loss import DomainIndependentLosses, EnhancedCircuitLoss
# from visualiser import TrainingVisualizer

# # List of pretrained model design names
# PRETRAINED_MODELS = ["apex1", "bc0", "c6288", "c7552", "i2c", "max", "sasc", "simple_spi"]

# def validate_file(path, file_description):
#     """Check if the given file exists; otherwise exit with an error."""
#     if not os.path.exists(path):
#         print(f"Error: {file_description} not found at '{path}'.")
#         sys.exit(1)
#     return path

# def load_pretrained_model(model_design, device):
#     """
#     Given a pretrained model name (e.g., "bc0"), load the model weights from weights/{model_design}.pt.
#     Returns an instance of CircuitQoRModel loaded on the specified device.
#     """
#     weights_path = os.path.join("weights", f"{model_design}.pt")
#     if not os.path.exists(weights_path):
#         print(f"Warning: Pretrained model '{model_design}' not found; skipping.")
#         return None
#     node_feature_dim = 7    # 4-d one-hot for node type + level, fanin, fanout
#     recipe_dim = 8          # 8 valid synthesis commands
#     hidden_dim = 128
#     model = CircuitQoRModel(node_feature_dim=node_feature_dim,
#                             recipe_dim=recipe_dim,
#                             hidden_dim=hidden_dim)
#     try:
#         state = torch.load(weights_path, map_location=device, weights_only=False)
#         model.load_state_dict(state)
#     except Exception as e:
#         print(f"Error loading model '{model_design}': {e}")
#         return None
#     model.to(device)
#     model.eval()
#     return model

# def evaluate_pretrained_model(model, dataloader, device):
#     """
#     Evaluate the given model on the new design's training dataset.
#     Computes the mean squared error on the final predicted area.
#     """
#     model.eval()
#     mse_losses = []
#     mse_loss_fn = torch.nn.MSELoss(reduction="mean")
#     with torch.no_grad():
#         for batch in dataloader:
#             # Move circuit and recipe data to device.
#             circuit_data = batch['circuit'].to(device)
#             recipe_data = batch['recipe'].to(device)
#             area_targets = batch['areas'].to(device)
#             # Forward pass: the forward method returns (final_pred, step_preds, uncertainty)
#             final_pred, step_preds, uncertainty = model(circuit_data, recipe_data)
#             # Calculate error on the final step prediction.
#             error = mse_loss_fn(final_pred, area_targets[:, -1].unsqueeze(1))
#             mse_losses.append(error.item())
#     return np.mean(mse_losses)

# def main():
#     parser = argparse.ArgumentParser(description="Fine-tune QoR (Area) Predictor for a new design.")
#     parser.add_argument("design_name", type=str,
#                         help="Name of the new design. Expects 'designs/{design_name}.bench' and 'datasets/{design_name}.csv'.")
#     parser.add_argument("initial_area", type=int,
#                         help="Initial area of the design.")
#     args = parser.parse_args()
#     design_name = args.design_name
#     init_area = args.initial_area

#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[INFO] Using device: {device}")

#     # Validate that required files exist.
#     bench_path = os.path.join("designs", f"{design_name}.bench")
#     csv_path = os.path.join("datasets", f"{design_name}.csv")
#     validate_file(bench_path, f"Design BENCH file for '{design_name}'")
#     validate_file(csv_path, f"Dataset CSV file for '{design_name}'")

#     # Load the new design dataset using the preexisting CircuitDataset.
#     new_dataset = CircuitDataset(csv_path=csv_path, designs_dir="designs", init_area=init_area)
#     dataset_size = len(new_dataset)
#     if dataset_size == 0:
#         print("Error: No data found in the CSV file.")
#         sys.exit(1)

#     # For fine-tuning, we split the dataset into training and validation sets (e.g., 80/20 split).
#     train_size = int(0.8 * dataset_size)
#     val_size = dataset_size - train_size
#     train_dataset, val_dataset = random_split(new_dataset, [train_size, val_size],
#                                                 generator=torch.Generator().manual_seed(42))
#     print(f"[INFO] New design '{design_name}' dataset loaded: {dataset_size} samples (Training: {train_size}, Validation: {val_size}).")
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
#                               collate_fn=collate_fn, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
#                             collate_fn=collate_fn, num_workers=4, pin_memory=True)

#     # Evaluate each pretrained model on the new design's training set.
#     model_errors = {}
#     print("[INFO] Evaluating pretrained models on the new design's training set:")
#     for pretrained in PRETRAINED_MODELS:
#         model = load_pretrained_model(pretrained, device)
#         if model is None:
#             continue
#         error = evaluate_pretrained_model(model, train_loader, device)
#         model_errors[pretrained] = error
#         print(f"  Model '{pretrained}': MSE error = {error:.4f}")
#     if not model_errors:
#         print("Error: No pretrained models could be loaded. Exiting.")
#         sys.exit(1)
#     # Select the model with the lowest error.
#     best_model_name = min(model_errors, key=model_errors.get)
#     best_error = model_errors[best_model_name]
#     print(f"[INFO] Selected pretrained model '{best_model_name}' (MSE error = {best_error:.4f}) as the starting point for fine-tuning.")

#     # Load the best pretrained model again.
#     best_model = load_pretrained_model(best_model_name, device)
#     if best_model is None:
#         print("Error: Could not load the selected pretrained model. Exiting.")
#         sys.exit(1)

#     # Fine-tuning parameters
#     finetune_epochs = 50
#     finetune_lr = 0.0005
#     checkpoint_dir = "finetune_checkpoints"
#     log_dir = "finetune_logs"
#     vis_dir = "finetune_visualizations"

#     # Fine-tune using your existing training function (train_model from main.py).
#     print("[INFO] Starting fine-tuning ...")
#     finetuned_model = train_model(
#         model=best_model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         device=device,
#         num_epochs=finetune_epochs,
#         learning_rate=finetune_lr,
#         weight_decay=1e-5,
#         checkpoint_dir=checkpoint_dir,
#         log_dir=log_dir,
#         patience=10,
#         vis_dir=vis_dir,
#         design_name=design_name 
#     )

#     # Save the fine-tuned model weights.
#     finetuned_weights_path = os.path.join("weights", f"{design_name}_finetuned.pt")
#     torch.save(finetuned_model.state_dict(), finetuned_weights_path)
#     print(f"[INFO] Fine-tuning finished. Fine-tuned model saved to '{finetuned_weights_path}'.")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python
"""
finetune.py

Fine-tunes a pre-trained QoR (Area) predictor on a new design.
Usage: python finetune.py {design_name} {initial_area}
This script looks for the design file in "designs/{design_name}.bench" and the dataset CSV in "datasets/{design_name}.csv".
It then uses all already-trained models (in weights/ folder) to make inferences on the new training dataset,
selects the model with the lowest inference error (computed as the step-wise MSE error over all synthesis steps),
fine-tunes it on the new dataset, and saves the fine-tuned weights.
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from main import CircuitDataset, collate_fn, train_model, CircuitQoRModel, evaluate_model
from loss import DomainIndependentLosses, EnhancedCircuitLoss
from visualiser import TrainingVisualizer

# List of pretrained model design names
PRETRAINED_MODELS = ["apex1", "bc0", "c6288", "c7552", "i2c", "max", "sasc", "simple_spi"]

def validate_file(path, file_description):
    """Check if the given file exists; otherwise exit with an error."""
    if not os.path.exists(path):
        print(f"Error: {file_description} not found at '{path}'.")
        sys.exit(1)
    return path

def load_pretrained_model(model_design, device):
    """
    Given a pretrained model name (e.g., "bc0"), load the model weights from weights/{model_design}.pt.
    Returns an instance of CircuitQoRModel loaded on the specified device.
    """
    weights_path = os.path.join("weights", f"{model_design}.pt")
    if not os.path.exists(weights_path):
        print(f"Warning: Pretrained model '{model_design}' not found; skipping.")
        return None
    # Node feature dimensions: 4-d one-hot for node type + [level, fanin, fanout]
    node_feature_dim = 7
    # Recipe dimension: 8 valid synthesis commands one-hot encoded
    recipe_dim = 8
    hidden_dim = 128
    model = CircuitQoRModel(node_feature_dim=node_feature_dim,
                            recipe_dim=recipe_dim,
                            hidden_dim=hidden_dim)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
    except Exception as e:
        print(f"Error loading model '{model_design}': {e}")
        return None
    model.to(device)
    model.eval()
    return model

def evaluate_pretrained_model(model, dataloader, device):
    """
    Evaluate the given model on the new design's training dataset by computing the step-wise MSE error.
    Instead of using only the final prediction, we calculate the error over each synthesis step.
    """
    model.eval()
    mse_losses = []
    mse_loss_fn = torch.nn.MSELoss(reduction="mean")
    with torch.no_grad():
        for batch in dataloader:
            # Move circuit and recipe data to device.
            circuit_data = batch['circuit'].to(device)
            recipe_data = batch['recipe'].to(device)
            # Target areas shape: (batch_size, num_steps)
            area_targets = batch['areas'].to(device)
            
            # Forward pass; assume model returns a tuple:
            # (final_pred, step_preds, uncertainty)
            final_pred, step_preds, uncertainty = model(circuit_data, recipe_data)
            # Assume step_preds is a tensor of shape (batch_size, num_steps, 1).
            # Squeeze the last dimension to match the shape of area_targets.
            step_preds = step_preds.squeeze(-1)  # Now shape: (batch_size, num_steps)
            
            # Compute the MSE across all synthesis steps.
            error = mse_loss_fn(step_preds, area_targets)
            mse_losses.append(error.item())
    return np.mean(mse_losses)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune QoR (Area) Predictor for a new design.")
    parser.add_argument("design_name", type=str,
                        help="Name of the new design. Expects 'designs/{design_name}.bench' and 'datasets/{design_name}.csv'.")
    args = parser.parse_args()
    design_name = args.design_name

    # Set device: use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Validate that required files exist.
    bench_path = os.path.join("designs", f"{design_name}.bench")
    csv_path = "./finetuning/clustered.csv"
    validate_file(bench_path, f"Design BENCH file for '{design_name}'")
    validate_file(csv_path, f"Dataset CSV file for '{design_name}'")

    # Load the new design dataset using the CircuitDataset.
    new_dataset = CircuitDataset(csv_path=csv_path, designs_dir="designs", init_area=1, divide_int=False)
    dataset_size = len(new_dataset)
    if dataset_size == 0:
        print("Error: No data found in the CSV file.")
        sys.exit(1)

    # For fine-tuning, split the dataset into training and validation sets (e.g., 80/20 split).
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(new_dataset, [train_size, val_size],
                                                generator=torch.Generator().manual_seed(42))
    print(f"[INFO] New design '{design_name}' dataset loaded: {dataset_size} samples (Training: {train_size}, Validation: {val_size}).")
    
    # Create data loaders.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Evaluate each pretrained model on the training dataset using step-wise predictions.
    model_errors = {}
    print("[INFO] Evaluating pretrained models on the new design's training set:")
    for pretrained in PRETRAINED_MODELS:
        model = load_pretrained_model(pretrained, device)
        if model is None:
            continue
        error = evaluate_pretrained_model(model, train_loader, device)
        model_errors[pretrained] = error
        print(f"  Model '{pretrained}': Step-wise MSE error = {error:.4f}")
    if not model_errors:
        print("Error: No pretrained models could be loaded. Exiting.")
        sys.exit(1)
    # Select the model with the lowest step-wise error.
    best_model_name = min(model_errors, key=model_errors.get)
    best_error = model_errors[best_model_name]
    print(f"[INFO] Selected pretrained model '{best_model_name}' (Step-wise MSE error = {best_error:.4f}) as the starting point for fine-tuning.")

    # Load the best pretrained model.
    best_model = load_pretrained_model(best_model_name, device)
    if best_model is None:
        print("Error: Could not load the selected pretrained model. Exiting.")
        sys.exit(1)

    # Initial evaluation on best_model 
    evaluate_model(best_model, val_loader, device, 'PreTrainingEvaluation')

    # Fine-tuning parameters.
    finetune_epochs = 100
    finetune_lr = 0.0005
    checkpoint_dir = "finetune_checkpoints"
    log_dir = "finetune_logs"
    vis_dir = "finetune_visualizations"

    # Fine-tune using the training function from main.py.
    print("[INFO] Starting fine-tuning ...")
    finetuned_model = train_model(
        model=best_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=finetune_epochs,
        learning_rate=finetune_lr,
        weight_decay=1e-5,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        patience=20,
        vis_dir=vis_dir,
        design_name=design_name 
    )

    # Save the fine-tuned model weights.
    finetuned_weights_path = os.path.join("weights", f"{design_name}_finetuned.pt")
    torch.save(finetuned_model.state_dict(), finetuned_weights_path)
    print(f"[INFO] Fine-tuning finished. Fine-tuned model saved to '{finetuned_weights_path}'.")

    # Model evaluation
    evaluate_model(finetuned_model, val_loader, device, 'PostTrainingEvaluation')

if __name__ == "__main__":
    main()