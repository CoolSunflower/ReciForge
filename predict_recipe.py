#!/usr/bin/env python
"""
Finds optimized synthesis recipes using QoR prediction models.
Usage:
    python predict_recipe.py {design_name} --method {greedy|mcts|ppo} --length LENGTH 
        [--iterations ITERATIONS] [--episodes EPISODES] [--initial_area INITIAL_AREA] 
        [--output OUTPUT_FILE] [--compare]
"""

import argparse
import sys
import time
import os

# Import the optimization functions
from recipe_optimization.iterative_greedy_sampling import iterative_greedy_sampling
from recipe_optimization.monte_carlo_tree_search import monte_carlo_tree_search
from recipe_optimization.ppo_recipe_optimization import ppo_recipe_optimization

def validate_design(design_name):
    """Ensure the design BENCH file exists."""
    design_path = f"designs/{design_name}.bench"
    if not os.path.exists(design_path):
        print(f"Error: Design file not found at '{design_path}'")
        sys.exit(1)
    return design_name

def main():
    parser = argparse.ArgumentParser(description="Synthesis Recipe Optimization")
    parser.add_argument("design_name", type=str,
                        help="Name of the design (designs/{design_name}.bench)")
    parser.add_argument("--method", type=str, choices=["greedy", "mcts", "ppo"], default="greedy",
                        help="Optimization method (default: greedy)")
    parser.add_argument("--length", type=int, required=True,
                        help="Desired length of the recipe")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of iterations for MCTS (default: 500)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes for PPO (default: 1000)")
    parser.add_argument("--initial_area", type=float, default=None,
                        help="Initial area (if design not in BASE_AREAS)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file to save the optimized recipe")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all optimization methods on the same design")
    args = parser.parse_args()
    
    # Validate inputs
    design_name = validate_design(args.design_name)
    
    if args.length <= 0:
        print("Error: Recipe length must be positive.")
        sys.exit(1)
    
    if args.method == "mcts" and args.iterations <= 0:
        print("Error: Number of iterations must be positive.")
        sys.exit(1)
        
    if args.method == "ppo" and args.episodes <= 0:
        print("Error: Number of episodes must be positive.")
        sys.exit(1)
    
    if args.compare:
        # Run all three methods and compare results
        print(f"Comparing all optimization methods on {design_name}...")
        
        print("\n===== GREEDY SAMPLING =====")
        start_time = time.time()
        greedy_recipe = iterative_greedy_sampling(
            design_name, 
            args.length, 
            args.initial_area
        )
        greedy_time = time.time() - start_time
        
        print("\n===== MONTE CARLO TREE SEARCH =====")
        start_time = time.time()
        mcts_recipe = monte_carlo_tree_search(
            design_name, 
            args.length, 
            args.iterations,
            args.initial_area
        )
        mcts_time = time.time() - start_time
        
        print("\n===== PROXIMAL POLICY OPTIMIZATION =====")
        start_time = time.time()
        ppo_recipe = ppo_recipe_optimization(
            design_name, 
            args.length,
            args.episodes,
            args.initial_area
        )
        ppo_time = time.time() - start_time
        
        print("\n===== COMPARISON SUMMARY =====")
        print(f"Greedy Sampling: Completed in {greedy_time:.1f}s")
        print(f"MCTS: Completed in {mcts_time:.1f}s")
        print(f"PPO: Completed in {ppo_time:.1f}s")
        print("\nOptimized Recipes:")
        print(f"Greedy: {greedy_recipe}")
        print(f"MCTS:   {mcts_recipe}")
        print(f"PPO:    {ppo_recipe}")
        
        # Save all recipes if output file specified
        if args.output:
            with open(f"{args.output}_greedy.txt", 'w') as f:
                for cmd in greedy_recipe:
                    f.write(f"{cmd}\n")
            with open(f"{args.output}_mcts.txt", 'w') as f:
                for cmd in mcts_recipe:
                    f.write(f"{cmd}\n")
            with open(f"{args.output}_ppo.txt", 'w') as f:
                for cmd in ppo_recipe:
                    f.write(f"{cmd}\n")
            print(f"\nRecipes saved to: {args.output}_greedy.txt, {args.output}_mcts.txt, and {args.output}_ppo.txt")
    else:
        # Run selected optimization method
        if args.method == "greedy":
            recipe = iterative_greedy_sampling(
                design_name, 
                args.length, 
                args.initial_area
            )
        elif args.method == "mcts":
            recipe = monte_carlo_tree_search(
                design_name, 
                args.length, 
                args.iterations, 
                args.initial_area
            )
        else:  # ppo
            recipe = ppo_recipe_optimization(
                design_name, 
                args.length,
                args.episodes,
                args.initial_area
            )
        
        # Save recipe to file if output path provided
        if args.output:
            with open(args.output, 'w') as f:
                for cmd in recipe:
                    f.write(f"{cmd}\n")
            print(f"Recipe saved to: {args.output}")

if __name__ == "__main__":
    main()
