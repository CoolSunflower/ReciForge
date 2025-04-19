def iterative_greedy_sampling(design_name, recipe_length, initial_area=None, exploration_factor=0.1, parallel_evals=4):
    """
    Build an optimal synthesis recipe using greedy command selection with enhanced exploration.
    
    Args:
        design_name: Target circuit design name
        recipe_length: Desired length of the recipe
        initial_area: Initial circuit area (optional)
        exploration_factor: Controls probability of non-greedy exploration (0-1)
        parallel_evals: Number of parallel command evaluations
    
    Returns:
        List of synthesis commands (optimal recipe)
    """
    # Import necessary modules
    from inference import determine_initial_area, load_all_finetuned_models
    from inference import validate_file, parse_bench_file, extract_circuit_features, encode_recipe
    from inference import VALID_COMMANDS
    import torch
    import numpy as np
    from torch_geometric.data import Data
    import concurrent.futures
    from collections import defaultdict
    import time
    
    # Setup
    start_time = time.time()
    recipe = []
    command_history = defaultdict(list)  # Store command effectiveness history
    cache = {}  # Cache predictions to avoid recomputation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models and prepare circuit data
    print(f"Loading models and preparing circuit data...")
    init_area = determine_initial_area(design_name, initial_area)
    models = load_all_finetuned_models(device)
    design_path = validate_file(f"designs/{design_name}.bench", f"Design BENCH file")
    circuit_graph = parse_bench_file(design_path)
    features = extract_circuit_features(circuit_graph)
    circuit_data = Data(x=features["x"], edge_index=features["edge_index"])
    circuit_data = circuit_data.to(device)
    
    print(f"Starting greedy optimization for {design_name} (initial area: {init_area})")
    print(f"Building recipe of length {recipe_length}...")
    
    # Track best area at each step for plateauing detection
    best_areas = [init_area]
    plateau_count = 0
    
    def evaluate_command(cmd, current_recipe):
        """Evaluate a single command using ensemble prediction"""
        if tuple(current_recipe + [cmd]) in cache:
            return cache[tuple(current_recipe + [cmd])]
            
        candidate_recipe = current_recipe + [cmd]
        recipe_encoded = encode_recipe(candidate_recipe)
        recipe_encoded = recipe_encoded.unsqueeze(0).to(device)
        
        # Predict using ensemble of models
        predictions = {}
        with torch.no_grad():
            for name, model in models.items():
                model.eval()  # Ensure model is in evaluation mode
                final_area, step_areas, uncertainty = model(circuit_data, recipe_encoded)
                step_areas = step_areas.squeeze(0).cpu().numpy()
                predictions[name] = (final_area.item(), step_areas, uncertainty.item())
        
        # Ensemble the predictions with weighted averaging
        combined_steps = np.zeros(len(candidate_recipe))
        combined_uncertainty = 0.0
        
        if design_name in models:
            # Weighted ensemble (79% main model, 3% others)
            main_weight = 0.79
            other_weight = 0.03
            for name, (_, s_areas, unc) in predictions.items():
                weight = main_weight if name == design_name else other_weight
                combined_steps += weight * s_areas
                combined_uncertainty += weight * unc
        else:
            # Simple average
            num_models = len(predictions)
            for (_, s_areas, unc) in predictions.values():
                combined_steps += s_areas / num_models
                combined_uncertainty += unc / num_models
        
        predicted_area = combined_steps[-1] * (init_area//2)
        result = (predicted_area, combined_uncertainty)
        cache[tuple(candidate_recipe)] = result
        return result
    
    # Iteratively build the recipe
    for i in range(recipe_length):
        # To prevent getting stuck in local minima, occasionally explore less promising commands
        exploration_mode = np.random.random() < exploration_factor
        
        command_scores = []
        
        # Parallel evaluation of commands
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_evals) as executor:
            future_to_cmd = {executor.submit(evaluate_command, cmd, recipe): cmd for cmd in VALID_COMMANDS}
            for future in concurrent.futures.as_completed(future_to_cmd):
                cmd = future_to_cmd[future]
                try:
                    predicted_area, uncertainty = future.result()
                    # Score includes uncertainty for exploration-exploitation balance
                    score = predicted_area + (uncertainty if exploration_mode else -uncertainty)
                    command_scores.append((cmd, predicted_area, uncertainty, score))
                except Exception as exc:
                    print(f"Command {cmd} generated an exception: {exc}")
        
        # Sort by score (lower is better)
        command_scores.sort(key=lambda x: x[3])
        
        # Select the best command or explore based on uncertainty
        if exploration_mode and len(command_scores) > 1:
            # In exploration mode, sometimes select the second-best command
            best_cmd, best_area, uncertainty, _ = command_scores[1]
            exploration_str = " (exploration mode)"
        else:
            best_cmd, best_area, uncertainty, _ = command_scores[0]
            exploration_str = ""
            
        # Update command history
        relative_improvement = (best_areas[-1] - best_area) / best_areas[-1] if best_areas[-1] > 0 else 0
        command_history[best_cmd].append(relative_improvement)
        
        # Detect plateauing (no significant improvement)
        if best_area > best_areas[-1] * 0.99:  # Less than 1% improvement
            plateau_count += 1
        else:
            plateau_count = 0
            
        # If plateauing for too long, try to break it by forcing exploration
        if plateau_count >= 3:
            # Find commands we haven't used much
            rare_commands = [cmd for cmd in VALID_COMMANDS if len(command_history[cmd]) < 2]
            if rare_commands:
                best_cmd = np.random.choice(rare_commands)
                best_area, uncertainty = evaluate_command(best_cmd, recipe)
                exploration_str = " (plateau breaking)"
                plateau_count = 0
                
        # Add the best command to the recipe
        recipe.append(best_cmd)
        best_areas.append(best_area)
        
        # Print progress with detailed information
        elapsed = time.time() - start_time
        print(f"Step {i+1}/{recipe_length} [{elapsed:.1f}s]:{exploration_str}")
        print(f"  Selected: '{best_cmd}', Predicted area: {round(best_area)}, Uncertainty: {uncertainty:.2f}")
        print(f"  Improvement: {relative_improvement*100:.2f}%")
        
        # Update statistics for adaptive exploration rate
        if i > 5:
            # If we're making good progress, reduce exploration
            recent_improvements = [best_areas[j] - best_areas[j+1] for j in range(max(0, i-5), i)]
            if sum(recent_improvements) > 0.05 * best_areas[max(0, i-5)]:
                exploration_factor *= 0.9
            else:
                # If we're stagnating, increase exploration
                exploration_factor = min(exploration_factor * 1.5, 0.5)
    
    # Print final results
    total_elapsed = time.time() - start_time
    area_reduction = (init_area - best_areas[-1]) / init_area * 100
    print(f"\nOptimization complete! ({total_elapsed:.1f}s)")
    print(f"Initial area: {round(init_area)}")
    print(f"Final predicted area: {round(best_areas[-1])}")
    print(f"Area reduction: {area_reduction:.2f}%")
    print(f"Optimized recipe: {recipe}")
    
    # Command effectiveness analysis
    print("\nCommand effectiveness:")
    for cmd in VALID_COMMANDS:
        avg_improvement = np.mean(command_history[cmd]) * 100 if command_history[cmd] else 0
        count = len(command_history[cmd])
        print(f"  {cmd}: Used {count} times, Avg improvement: {avg_improvement:.2f}%")
    
    return recipe
