class MCTSNode:
    """Monte Carlo Tree Search node for synthesis recipe optimization."""
    
    def __init__(self, recipe=None, parent=None, command=None, depth=0):
        self.recipe = recipe if recipe is not None else []
        self.parent = parent
        self.command = command  # Command used to reach this node from parent
        self.depth = depth
        self.children = {}  # command -> node
        self.visits = 0
        self.value = 0.0
        self.uncertainty = 0.0
        self.best_value = float('inf')  # Best (lowest) area found in subtree
        self.fully_expanded = False
        self.is_terminal = False
    
    def is_fully_expanded(self, valid_commands):
        """Check if all possible children have been expanded."""
        if self.fully_expanded:
            return True
        return len(self.children) == len(valid_commands)
    
    def best_child(self, exploration_weight=1.0, progressive_widening=False):
        """Select the best child according to UCB1 formula with progressive widening."""
        import math
        
        if not self.children:
            return None
            
        # Determine the number of children to consider based on visits
        if progressive_widening:
            # With many visits, consider all children; with few visits, consider fewer
            k = max(1, int(math.sqrt(self.visits)))
            child_items = sorted(
                self.children.items(), 
                key=lambda x: x[1].value / max(1, x[1].visits)
            )[:k]
            child_dict = dict(child_items)
        else:
            child_dict = self.children
        
        # UCB1 formula: vi + C * sqrt(ln(N) / ni) with uncertainty bonus
        def ucb1(child):
            if child.visits == 0:
                return float('inf')
            
            exploit = -child.value / child.visits  # Negative because lower area is better
            explore = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            uncertainty_bonus = 0.1 * child.uncertainty / (child.visits**0.5)
            
            return exploit + explore + uncertainty_bonus
            
        return max(child_dict.values(), key=ucb1)
    
    def add_child(self, command, recipe, depth):
        """Add a child node for the given command."""
        child = MCTSNode(
            recipe=recipe,
            parent=self,
            command=command,
            depth=depth
        )
        self.children[command] = child
        return child
        
    def backpropagate(self, value, uncertainty):
        """Update node statistics from leaf to root."""
        node = self
        while node:
            node.visits += 1
            # Lower area is better, so negative reward
            node.value += value
            node.uncertainty = (node.uncertainty * (node.visits - 1) + uncertainty) / node.visits
            node.best_value = min(node.best_value, value)
            node = node.parent


def monte_carlo_tree_search(
    design_name, 
    recipe_length, 
    iterations=500, 
    initial_area=None,
    exploration_weight=2.0,
    use_progressive_widening=True
):
    """
    Build an optimal synthesis recipe using Monte-Carlo Tree Search.
    
    Args:
        design_name: Target circuit design name
        recipe_length: Desired length of the recipe
        iterations: Number of MCTS iterations to perform
        initial_area: Initial circuit area (optional)
        exploration_weight: Controls exploration vs exploitation in UCB
        use_progressive_widening: Whether to use progressive widening technique
        
    Returns:
        List of synthesis commands (optimal recipe)
    """
    import math
    import random
    import time
    import torch
    import numpy as np
    from torch_geometric.data import Data
    from inference import determine_initial_area, load_all_finetuned_models
    from inference import validate_file, parse_bench_file, extract_circuit_features, encode_recipe
    from inference import VALID_COMMANDS
    
    # Setup
    start_time = time.time()
    eval_cache = {}  # Cache for evaluated recipes to avoid redundant computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize root node (empty recipe)
    root = MCTSNode()
    
    # Load models and prepare circuit data
    print(f"Loading models and preparing circuit data...")
    init_area = determine_initial_area(design_name, initial_area)
    models = load_all_finetuned_models(device)
    design_path = validate_file(f"designs/{design_name}.bench", f"Design BENCH file")
    circuit_graph = parse_bench_file(design_path)
    features = extract_circuit_features(circuit_graph)
    circuit_data = Data(x=features["x"], edge_index=features["edge_index"])
    circuit_data = circuit_data.to(device)
    
    print(f"Starting MCTS for {design_name} (initial area: {init_area})")
    print(f"Building recipe of length {recipe_length} using {iterations} iterations...")
    
    # Fast rollout policy based on command effectiveness statistics
    command_stats = {cmd: {'visits': 0, 'value': 0} for cmd in VALID_COMMANDS}
    
    def evaluate_recipe(recipe):
        """Evaluate a recipe using ensemble inference and caching."""
        recipe_tuple = tuple(recipe)
        if recipe_tuple in eval_cache:
            return eval_cache[recipe_tuple]
            
        if not recipe:
            return init_area, 0.0
        
        recipe_encoded = encode_recipe(recipe)
        recipe_encoded = recipe_encoded.unsqueeze(0).to(device)
        
        predictions = {}
        with torch.no_grad():
            for name, model in models.items():
                model.eval()  # Ensure model is in evaluation mode
                final_area, step_areas, uncertainty = model(circuit_data, recipe_encoded)
                step_areas = step_areas.squeeze(0).cpu().numpy()
                predictions[name] = (final_area.item(), step_areas, uncertainty.item())
        # Ensemble the predictions
        combined_final = 0.0
        combined_uncertainty = 0.0
        
        if design_name in models:
            # Weighted ensemble
            main_weight = 0.79
            other_weight = 0.03
            for name, (f_area, _, unc) in predictions.items():
                weight = main_weight if name == design_name else other_weight
                combined_final += weight * f_area
                combined_uncertainty += weight * unc
        else:
            # Simple average
            combined_final = sum(f_area for f_area, _, _ in predictions.values()) / len(predictions)
            combined_uncertainty = sum(unc for _, _, unc in predictions.values()) / len(predictions)
        
        result = (combined_final * (init_area//2), combined_uncertainty)
        eval_cache[recipe_tuple] = result
        return result
    
    def rollout_policy(recipe, remaining_steps):
        """Policy for simulation phase - choose commands with adaptive exploration."""
        if remaining_steps == 0:
            return recipe
            
        complete_recipe = list(recipe)
        
        # Combine exploitation (best commands) with exploration (random commands)
        for _ in range(remaining_steps):
            # With 70% probability, choose commands based on statistics
            if random.random() < 0.7 and all(command_stats[cmd]['visits'] > 0 for cmd in VALID_COMMANDS):
                # Compute UCB1 scores for each command
                ucb_scores = {}
                total_visits = sum(command_stats[cmd]['visits'] for cmd in VALID_COMMANDS)
                
                for cmd in VALID_COMMANDS:
                    stats = command_stats[cmd]
                    # Use average value with exploration term
                    value = stats['value'] / stats['visits'] if stats['visits'] > 0 else 0
                    explore = math.sqrt(2 * math.log(total_visits) / stats['visits']) if stats['visits'] > 0 else float('inf')
                    ucb_scores[cmd] = -value + explore  # Negative because lower area is better
                
                # Select command with highest UCB score
                next_cmd = max(ucb_scores.items(), key=lambda x: x[1])[0]
            else:
                # Random exploration
                next_cmd = random.choice(VALID_COMMANDS)
            
            complete_recipe.append(next_cmd)
        
        return complete_recipe
    
    def update_command_stats(recipe, area_reduction):
        """Update command statistics based on evaluation results."""
        for cmd in recipe:
            stats = command_stats[cmd]
            stats['visits'] += 1
            # Higher area reduction (negative area) is better
            stats['value'] += area_reduction
    
    # MCTS algorithm
    best_recipe = None
    best_value = float('inf')
    progress_interval = max(1, iterations // 20)
    
    for iter_num in range(iterations):
        if iter_num % progress_interval == 0 or iter_num == iterations - 1:
            elapsed = time.time() - start_time
            if best_recipe:
                best_area = best_value
                reduction = (init_area - best_area) / init_area * 100
                print(f"Iteration {iter_num}/{iterations} ({elapsed:.1f}s) - Best area: {best_area:.1f} ({reduction:.2f}% reduction)")
        
        # 1. Selection phase: navigate from root to leaf using UCB
        node = root
        while node.is_fully_expanded(VALID_COMMANDS) and node.children and not node.is_terminal:
            node = node.best_child(
                exploration_weight=exploration_weight,
                progressive_widening=use_progressive_widening
            )
            
            # Handle case where best_child returns None
            if node is None:
                break
                
            # Terminal node at max depth
            if len(node.recipe) >= recipe_length:
                node.is_terminal = True
        
        # 2. Expansion phase: add a new child if not terminal
        if not node.is_terminal and len(node.recipe) < recipe_length:
            unexpanded = [cmd for cmd in VALID_COMMANDS if cmd not in node.children]
            
            if unexpanded:
                # Choose command to expand - prioritize promising commands
                if random.random() < 0.8:
                    # Choose based on global statistics when available
                    cmd_scores = {}
                    for cmd in unexpanded:
                        stats = command_stats[cmd]
                        if stats['visits'] > 0:
                            cmd_scores[cmd] = -stats['value'] / stats['visits']
                        else:
                            cmd_scores[cmd] = 0
                    
                    if cmd_scores:
                        # Add some randomization to exploration
                        weights = np.array(list(cmd_scores.values()))
                        weights = np.exp(weights - np.max(weights))  # Softmax-like weighting
                        weights = weights / np.sum(weights)
                        command = np.random.choice(unexpanded, p=weights)
                    else:
                        command = random.choice(unexpanded)
                else:
                    command = random.choice(unexpanded)
                
                # Create new child node
                child_recipe = node.recipe + [command]
                child = node.add_child(command, child_recipe, node.depth + 1)
                
                # Fully expanded check
                if len(node.children) == len(VALID_COMMANDS):
                    node.fully_expanded = True
                
                # Terminal check for leaf nodes
                if len(child_recipe) >= recipe_length:
                    child.is_terminal = True
                
                node = child
        
        # 3. Simulation phase: complete the recipe and evaluate
        if node.is_terminal or len(node.recipe) >= recipe_length:
            # Node is already a complete recipe
            simulation_recipe = node.recipe
        else:
            # Complete recipe with rollout policy
            remaining_steps = recipe_length - len(node.recipe)
            simulation_recipe = rollout_policy(node.recipe, remaining_steps)
        
        # Evaluate the complete recipe
        area, uncertainty = evaluate_recipe(simulation_recipe)
        
        # Update best recipe if better
        if area < best_value:
            best_value = area
            best_recipe = simulation_recipe
            print(f"New best recipe found (area: {area:.1f}):")
            print(f"  {simulation_recipe}")
        
        # 4. Backpropagation: update statistics up the tree
        node.backpropagate(area, uncertainty)
        
        # Update command statistics for rollout policy
        area_reduction = init_area - area
        update_command_stats(simulation_recipe, area_reduction)
    
    # Extract the best recipe from the tree or use the stored best
    if best_recipe is None:
        # Construct from most-visited path
        best_recipe = []
        node = root
        for _ in range(recipe_length):
            if not node.children:
                # If node has no children, add random commands to complete
                remaining = recipe_length - len(best_recipe)
                best_recipe.extend(random.choices(VALID_COMMANDS, k=remaining))
                break
            
            # Choose child with highest visit count
            next_node = max(node.children.values(), key=lambda child: child.visits)
            best_recipe.append(next_node.command)
            node = next_node
    
    # Final evaluation of best recipe
    best_area, _ = evaluate_recipe(best_recipe)
    
    # Print results
    total_elapsed = time.time() - start_time
    area_reduction = (init_area - best_area) / init_area * 100
    print(f"\nOptimization complete! ({total_elapsed:.1f}s)")
    print(f"Initial area: {init_area:.1f}")
    print(f"Final predicted area: {best_area:.1f}")
    print(f"Area reduction: {area_reduction:.2f}%")
    print(f"Optimized recipe: {best_recipe}")
    
    # Command statistics analysis
    print("\nCommand statistics:")
    for cmd in VALID_COMMANDS:
        stats = command_stats[cmd]
        avg_value = stats['value'] / stats['visits'] if stats['visits'] > 0 else 0
        print(f"  {cmd}: Used {stats['visits']} times, Avg value: {avg_value:.4f}")
    
    return best_recipe
