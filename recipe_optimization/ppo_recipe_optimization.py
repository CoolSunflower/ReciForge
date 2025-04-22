def ppo_recipe_optimization(design_name, recipe_length, training_episodes=1000, initial_area=None):
    """
    Build an optimal synthesis recipe using Proximal Policy Optimization.
    
    Args:
        design_name: Target circuit design name
        recipe_length: Desired length of the recipe
        training_episodes: Number of training episodes for PPO
        initial_area: Initial circuit area (optional)
        
    Returns:
        List of synthesis commands (optimal recipe)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    from torch.distributions import Categorical
    import time
    from inference import determine_initial_area, load_all_finetuned_models
    from inference import validate_file, parse_bench_file, extract_circuit_features, encode_recipe
    from inference import VALID_COMMANDS
    from torch_geometric.data import Data
    
    # PPO hyperparameters
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    CRITIC_DISCOUNT = 0.5
    ENTROPY_BETA = 0.01
    PPO_EPOCHS = 4
    BATCH_SIZE = 64
    LR = 3e-4
    
    class ActorCritic(nn.Module):
        """Actor-Critic neural network for PPO."""
        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(ActorCritic, self).__init__()
            
            # Shared feature extractor
            self.feature_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            # Actor head (policy network)
            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            
            # Critic head (value network)
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, x):
            """Forward pass through the network."""
            features = self.feature_net(x)
            action_logits = self.actor(features)
            value = self.critic(features)
            return action_logits, value
        
        def act(self, state, deterministic=False):
            """Select an action based on the current policy."""
            model_device = next(self.parameters()).device
            state = torch.FloatTensor(state).unsqueeze(0).to(model_device)
            logits, value = self.forward(state)

            dist = Categorical(logits=logits)
            
            if deterministic:
                action_tensor = torch.argmax(logits, dim=1)
                action = action_tensor.item()
                return action, value.item(), None
            else:
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor)
                action = action_tensor.item()
                return action, value.item(), log_prob.item()
        
        def evaluate(self, states, actions):
            """Evaluate actions given states."""
            logits, values = self.forward(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_probs, values.squeeze(), entropy
    
    class SynthesisEnvironment:
        """Environment for synthesis recipe optimization."""
        
        def __init__(self, design_name, recipe_length, initial_area, device):
            self.design_name = design_name
            self.recipe_length = recipe_length
            self.initial_area = initial_area
            self.device = device
            self.valid_commands = VALID_COMMANDS
            
            # Load QoR prediction models
            self.models = load_all_finetuned_models(device)
            
            # Prepare circuit data
            design_path = validate_file(f"designs/{design_name}.bench", f"Design BENCH file")
            circuit_graph = parse_bench_file(design_path)
            features = extract_circuit_features(circuit_graph)
            self.circuit_data = Data(x=features["x"], edge_index=features["edge_index"])
            self.circuit_data = self.circuit_data.to(device)
            
            # Reset the environment
            self.reset()
        
        def reset(self):
            """Reset the environment to initial state."""
            self.current_recipe = []
            self.current_step = 0
            self.current_area = self.initial_area
            
            # State representation: one-hot encoded recipe and position indicator
            # Shape: (recipe_length * command_count) + recipe_length
            state_shape = (self.recipe_length * len(self.valid_commands)) + self.recipe_length
            self.state = np.zeros(state_shape, dtype=np.float32)
            
            # Set position indicator for first step
            self.state[len(self.valid_commands) * self.recipe_length] = 1
            
            return self.state
        
        def step(self, action):
            """Take a step in the environment by adding a command to the recipe."""
            command = self.valid_commands[action]
            self.current_recipe.append(command)
            
            # Update state representation
            # 1. Set the one-hot encoding for the chosen command at current step
            cmd_idx = self.current_step * len(self.valid_commands) + action
            self.state[cmd_idx] = 1
            
            # 2. Update position indicator
            pos_base_idx = len(self.valid_commands) * self.recipe_length
            self.state[pos_base_idx + self.current_step] = 0  # Clear current position
            if self.current_step + 1 < self.recipe_length:
                self.state[pos_base_idx + self.current_step + 1] = 1  # Set next position
            
            # Predict area using QoR models
            recipe_encoded = encode_recipe(self.current_recipe)
            recipe_encoded = recipe_encoded.unsqueeze(0).to(self.device)
            
            predictions = {}
            with torch.no_grad():
                for name, model in self.models.items():
                    model.eval()
                    final_area, step_areas, uncertainty = model(self.circuit_data, recipe_encoded)
                    step_areas = step_areas.squeeze(0).cpu().numpy()
                    predictions[name] = (step_areas[-1], uncertainty.item())
            
            # Ensemble the predictions with weighted averaging
            if self.design_name in self.models:
                # Weighted ensemble (79% main model, 3% others)
                main_weight = 0.79
                other_weight = 0.03
                predicted_area = 0.0
                predicted_uncertainty = 0.0
                for name, (area, uncertainty) in predictions.items():
                    weight = main_weight if name == self.design_name else other_weight
                    predicted_area += weight * area
                    predicted_uncertainty += weight * uncertainty
            else:
                # Simple average
                predicted_area = sum(p[0] for p in predictions.values()) / len(predictions)
                predicted_uncertainty = sum(p[1] for p in predictions.values()) / len(predictions)
            
            # Scale predicted area and calculate reward
            predicted_area = predicted_area * (self.initial_area//2)
            area_reduction = self.current_area - predicted_area
            
            # Calculate reward based on area reduction with uncertainty penalty
            if area_reduction > 0:
                reward = area_reduction / self.initial_area  # Normalized reduction
            else:
                reward = -0.1  # Penalty for area increase
            
            # Uncertainty penalty (higher uncertainty → lower reward)
            confidence_factor = 1.0 / (1.0 + predicted_uncertainty)
            reward *= confidence_factor
            
            # Update current area and step
            self.current_area = predicted_area
            self.current_step += 1
            
            # Check if episode is done
            done = self.current_step >= self.recipe_length
            
            # Additional reward for final state based on total area reduction
            if done:
                total_reduction = (self.initial_area - predicted_area) / self.initial_area
                reward += total_reduction * 2.0  # Bonus for overall performance
            
            return self.state, reward, done, {"area": predicted_area}
    
    # Initialize device and environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_area = determine_initial_area(design_name, initial_area)
    env = SynthesisEnvironment(design_name, recipe_length, init_area, device)
    
    # Initialize model
    state_dim = env.state.shape[0]
    action_dim = len(VALID_COMMANDS)
    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Starting PPO training for {design_name} with {training_episodes} episodes...")
    print(f"Initial area: {init_area}, Recipe length: {recipe_length}")
    
    best_recipe = None
    best_area = float('inf')
    start_time = time.time()
    
    # Training loop
    for episode in range(training_episodes):
        # Storage for episode data
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        
        # Collect experience
        state = env.reset()
        episode_reward = 0
        
        for _ in range(recipe_length):
            action, value, log_prob = model.act(state)
            
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Compute returns and advantages using GAE
        returns = []
        advantages = []
        gae = 0
        
        # Add dummy next value for last state
        next_val = 0 if dones[-1] else values[-1]
        values.append(next_val)
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
        
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(PPO_EPOCHS):
            # Create random indices for mini-batches
            indices = np.random.permutation(len(states))
            
            # Mini-batch training
            for start_idx in range(0, len(states), BATCH_SIZE):
                idx = indices[start_idx:start_idx+BATCH_SIZE]
                
                # Extract mini-batch
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                
                # Get new policy distributions and values
                new_log_probs, values, entropy = model.evaluate(mb_states, mb_actions)
                
                # Policy ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Surrogate losses
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * mb_advantages
                
                # PPO losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = actor_loss + CRITIC_DISCOUNT * critic_loss + ENTROPY_BETA * entropy_loss
                
                # Update network
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        
        # Track best recipe
        if env.current_area < best_area:
            best_area = env.current_area
            best_recipe = env.current_recipe.copy()
            area_reduction = (init_area - best_area) / init_area * 100
            print(f"Episode {episode+1}: New best recipe found!")
            print(f"  Area: {best_area:.1f}, Reduction: {area_reduction:.2f}%")
            print(f"  Recipe: {best_recipe}")
        
        # Progress reporting
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode+1}/{training_episodes} ({elapsed:.1f}s)")
            print(f"  Episode reward: {episode_reward:.4f}")
    
    # Final evaluation with the best policy
    print("\nRunning final evaluation with trained policy...")
    state = env.reset()
    done = False
    final_recipe = []
    
    while not done:
        action, _, _ = model.act(state, deterministic=True)
        next_state, _, done, info = env.step(action)
        final_recipe.append(VALID_COMMANDS[action])
        state = next_state
    
    final_area = info["area"]
    area_reduction = (init_area - final_area) / init_area * 100
    
    print(f"\nOptimization complete! ({time.time() - start_time:.1f}s)")
    print(f"Initial area: {init_area}")
    print(f"Final predicted area: {final_area:.1f}")
    print(f"Area reduction: {area_reduction:.2f}%")
    print(f"Optimized recipe: {final_recipe}")
    
    # Return best recipe found during training
    return best_recipe if best_area < final_area else final_recipe
