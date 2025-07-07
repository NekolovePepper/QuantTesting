import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
# Assuming your environment is in src.core.env
# from src.core.env import StockTradingEnv 

class ActorCritic(nn.Module):
    """
    Actor-Critic Network for A2C.
    Outputs action distribution parameters (mean for continuous actions) and state value.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # Shared feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head - outputs mean of the action distribution
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Ensures portfolio weights sum to 1
        )

        # Critic head - outputs state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Action variance (covariance matrix diagonal) - can be fixed or learned
        # For simplicity, let's start with a fixed small variance or make it learnable
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim)) # Learnable log std

    def forward(self, state):
        features = self.feature_layer(state)
        action_mean = self.actor_mean(features)
        state_value = self.critic(features)
        
        action_std = torch.exp(self.action_log_std)
        # Ensure action_std is broadcastable to action_mean's shape if state is a batch
        if action_mean.ndim > 1 and action_std.ndim == 2 and action_std.shape[0] == 1:
             action_std = action_std.expand_as(action_mean)

        dist = MultivariateNormal(action_mean, torch.diag_embed(action_std.pow(2)))
        return dist, state_value

    def evaluate_actions(self, state, action):
        dist, state_value = self.forward(state)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_value, dist_entropy


class A2CAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, 
                 entropy_coef=0.01, device='auto'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"A2C using device: {self.device}")

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.policy.actor_mean.parameters(), lr=lr_actor)
        # If action_log_std is learnable, add it to actor_optimizer or a separate one
        # self.actor_optimizer = optim.Adam(list(self.policy.actor_mean.parameters()) + [self.policy.action_log_std], lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=lr_critic)
        
        # Buffer to store transitions for an episode or a fixed number of steps
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

        self.training = True

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0) # Add batch dimension
        with torch.no_grad():
            dist, state_value = self.policy(state)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            
            action_log_prob = dist.log_prob(action)

        # Normalize action to sum to 1 (important for portfolio)
        action = action / (action.sum(dim=-1, keepdim=True) + 1e-8)
        
        if self.training:
            # Store for training, ensure tensors are on CPU for append if mixing devices later
            self.states.append(state.squeeze(0).cpu()) 
            self.actions.append(action.squeeze(0).cpu())
            self.log_probs.append(action_log_prob.cpu())
            self.values.append(state_value.cpu())

        return action.squeeze(0).cpu().numpy()

    def store_reward_done(self, reward, done):
        """ Call this after env.step() to store reward and done for the last stored transition """
        if self.training:
            self.rewards.append(reward)
            self.dones.append(done)
    
    def clear_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def train(self):
        if not self.states: # or len(self.states) < some_min_batch_size
            return

        # Convert lists to tensors
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device).unsqueeze(1)
        
        # Use stored values and log_probs
        old_log_probs = torch.stack(self.log_probs).to(self.device).detach()
        old_values = torch.stack(self.values).to(self.device).detach()
        
        # Calculate returns (Gt)
        returns = []
        discounted_reward = 0
        # If the last state was not terminal, bootstrap from its value estimate
        if not dones[-1]:
            with torch.no_grad():
                # Get value of the very last state encountered (s_T)
                # This requires self.states to include the state *after* the last action
                # Or, pass next_state of the last transition explicitly
                # For simplicity, if we store s,a,r,s',d, then the last s' is needed.
                # Current buffer stores s, a, log_p, V(s). We need V(s_T) if not done.
                # Let's assume for now the episode finished or we handle this outside.
                # A common way is to get V(s_last_next_state) if not done.
                # For now, if not done, let's use the value of the last state in buffer.
                # This is an approximation if the buffer doesn't end at a terminal state.
                # A better way: ensure the last reward/done corresponds to the last state in self.states
                # and if not done, use self.policy(last_next_state).value
                # For now, let's assume the buffer is processed at episode end or fixed length
                # where the last state's value is either 0 (if terminal) or bootstrapped.
                # The current `store_reward_done` and `select_action` implies `values` are V(s_t)
                # So, for GAE or returns, we need V(s_{t+1}).
                # Let's re-evaluate states for V(s_t) and V(s_{t+1}) during training or adjust buffer.

                # Simplified: Calculate returns assuming buffer is one full episode or n-steps
                # The last value in `old_values` is V(s_{T-1}) if buffer has T steps.
                # We need V(s_T) for the last step if not terminal.
                # For simplicity, if the episode is not done at the end of buffer,
                # we would need the value of the state that *would have come next*.
                # Let's assume for now, if dones[-1] is False, we use the last computed value as bootstrap.
                # This is a common simplification for batched A2C.
                # A more correct n-step return would be:
                # R_t = r_t + gamma * r_{t+1} + ... + gamma^(n-1) * r_{t+n-1} + gamma^n * V(s_{t+n})
                
                # Let's use a simpler GAE-like calculation for advantages
                # We need V(s_next) for each step.
                # The current `self.values` stores V(s_t).
                # We need to re-calculate values or store next_state_values.
                # For A2C, it's common to calculate returns and then advantages.
                
                # Recalculate values for all states in the buffer for consistency
                # This is not ideal as `old_values` were from the policy at decision time.
                # A2C is on-policy, so this is fine.
                
                # Let's compute returns: R_t = r_t + gamma * V(s_{t+1})
                # For the last step, if not done, V(s_T_next) is needed.
                # If done, V(s_T_next) = 0.
                
                # Let's compute discounted returns from rewards
                # G_t = r_t + gamma * G_{t+1}
                # This is standard for policy gradient.
                
                next_value = 0 # V(s_T) if terminal, or bootstrap
                if not dones[-1]: # If the episode (or batch) didn't end with a terminal state
                    # Bootstrap from the value of the state that followed the last action
                    # This requires having the next_state for the last action.
                    # For now, let's assume the buffer is an episode, or we handle this outside.
                    # If we are doing n-step returns, the last `next_value` would be V(s_{t+n})
                    # For simplicity, if the last state in buffer is s_N, and it's not terminal,
                    # we'd ideally use V(s_{N+1}). If we don't have s_{N+1}, we can use V(s_N) from critic.
                    # This is a slight simplification.
                    # Let's assume `rewards` and `dones` align with `states` and `actions`.
                    # `values` are V(states[t]).
                    
                    # If the last transition in the buffer is (s_k, a_k, r_k, done_k, V(s_k), logp_k)
                    # and done_k is False, we need V(s_{k+1}) to calculate return for s_k.
                    # If we don't have s_{k+1} (e.g. end of data collection before terminal),
                    # we can use V(s_k) from `values` as a proxy for V(s_{k+1}) if we assume
                    # the value function is somewhat stable, or set it to 0 if we cut off.
                    # A common approach for n-step A2C is to get the value of the n-th next state.
                    
                    # Let's use the standard Monte Carlo returns if episode ends.
                    # If collecting fixed n-steps, the last return is bootstrapped with V(s_{t+n}).
                    # For now, let's assume the buffer is one episode.
                    pass # `next_value` remains 0 if we assume episode ends or cut-off.
                     # Or, if we have the state that *would* be next:
                     # last_state_in_buffer = torch.FloatTensor(self.states[-1]).to(self.device).unsqueeze(0)
                     # _, last_next_val = self.policy(last_state_in_buffer) # This is V(s_last)
                     # We need V(s_really_last_next_state)
                     # For simplicity, if not dones[-1], we can use the critic's estimate of the last state in the trajectory
                     # as the bootstrap value. This is what `old_values[-1]` is.
                    if len(old_values) > 0: # Ensure buffer is not empty
                         next_value = old_values[-1].detach() * (1-dones[-1]) # Bootstrap if not done


        for i in reversed(range(len(rewards))):
            discounted_reward = rewards[i] + self.gamma * next_value * (1 - dones[i]) # dones[i] is for s_{i+1}
            returns.insert(0, discounted_reward)
            next_value = discounted_reward # This is not quite right for standard returns.
            # Corrected: next_value should be V(s_{i+1}) for calculating R_i
            # R_i = r_i + gamma * V(s_{i+1})
            # Let's use GAE-like calculation for advantages, using V(s_t) and V(s_{t+1})
            # For returns: G_t = r_t + gamma * r_{t+1} + ... + gamma^k * V(s_{t+k})
            # Let's use simple discounted rewards for now.
        
        # Recalculate returns properly:
        returns = []
        R = 0
        # If the last state in the buffer was not terminal, bootstrap its value.
        # `old_values` are V(s_t). We need V(s_{t+1}) for the return calculation.
        # Let's assume `next_value_for_last_step` is V(s_N) if s_{N-1} is the last state in buffer.
        # Or, if `states` contains s_0 to s_{N-1}, then `rewards` r_0 to r_{N-1}.
        # R_{N-1} = r_{N-1} + gamma * V(s_N) * (1-done_{N-1})
        # V(s_N) would be from self.policy(state_N) if state_N is available.
        # If we only have up to s_{N-1}, and it's not terminal, we use V(s_{N-1}) as bootstrap.
        
        # Let's assume `old_values` are V(s_0), V(s_1), ..., V(s_{T-1})
        # And `rewards` are r_0, r_1, ..., r_{T-1}
        # And `dones` are d_0, d_1, ..., d_{T-1} (is s_{t+1} terminal?)
        
        # Standard calculation of returns (targets for value function)
        # R_t = r_t + gamma * R_{t+1} (if not done) else r_t
        # Or, R_t = r_t + gamma * V(s_{t+1}) (if not done)
        
        # Let's use n-step returns or Monte Carlo returns if episode is complete
        # For simplicity, let's use the rewards to go, and bootstrap if not done.
        
        next_return_val = 0.0
        if not dones[-1]: # If the last collected transition was not terminal
            # Bootstrap with the value of the state that *would have been next*
            # This requires having that next state. If we don't, we can use the value of the last state.
            # `old_values[-1]` is V(s_last).
            next_return_val = old_values[-1].item() # Bootstrap with V(s_last) if not terminal

        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * next_return_val * (1 - dones[i]) # dones[i] is for s_{i+1}
            returns.insert(0, R)
            # For the next iteration, R becomes the "value of the next state"
            # This is correct for calculating rewards-to-go.
            next_return_val = R 
            # If we are using V(s) from critic for bootstrapping:
            # next_return_val = old_values[i] # This would be if R_t = r_t + gamma*V(s_{t+1})
            # The above is correct if R is G_t (return to go)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device).detach()
        if returns.ndim == 1:
            returns = returns.unsqueeze(1)

        # Advantages A(s_t, a_t) = R_t - V(s_t)
        advantages = returns - old_values
        
        # Re-evaluate actions with current policy to get new log_probs and entropy
        # This is needed because A2C is on-policy and updates from current experiences
        # The `old_log_probs` were from the policy at the time of action.
        # For actor loss, we need log_prob(a_t | s_t, current_theta)
        
        # Convert buffered states and actions to tensors for batch processing
        batch_states = torch.stack(self.states).to(self.device).detach()
        batch_actions = torch.stack(self.actions).to(self.device).detach()

        # Get log_probs, values, and entropy from the current policy
        log_probs_new, values_new, entropy = self.policy.evaluate_actions(batch_states, batch_actions)

        # Actor Loss (Policy Gradient)
        actor_loss = -(log_probs_new * advantages.detach()).mean() - self.entropy_coef * entropy.mean()
        
        # Critic Loss (MSE)
        # Target for critic is `returns`
        critic_loss = nn.MSELoss()(values_new, returns.detach())

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm_(self.policy.actor_mean.parameters(), 0.5) # Optional
        self.actor_optimizer.step()

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 0.5) # Optional
        self.critic_optimizer.step()
        
        self.clear_buffer()


    def save(self, directory, name="a2c_agent"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        model_path = os.path.join(directory, f"{name}.pth")
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, model_path)
        print(f"A2C model saved to {model_path}")

    def load(self, directory, name="a2c_agent"):
        model_path = os.path.join(directory, f"{name}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            if 'actor_optimizer_state_dict' in checkpoint:
                 self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            if 'critic_optimizer_state_dict' in checkpoint:
                 self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            print(f"A2C model loaded from {model_path}")
            return True
        else:
            print(f"Could not load A2C model from {model_path}, file not found.")
            return False

# Example usage (conceptual)
if __name__ == '__main__':
    # from src.core.env import StockTradingEnv # Make sure this import works
    # dummy_env = StockTradingEnv(...) # Initialize your environment
    # state_dim = dummy_env.observation_space.shape[0]
    # action_dim = dummy_env.action_space.shape[0]
    
    state_dim = 10 # Example
    action_dim = 3  # Example

    agent = A2CAgent(state_dim, action_dim)
    
    # Dummy training loop
    for episode in range(10):
        # state = dummy_env.reset()
        state = np.random.rand(state_dim) # Dummy state
        done = False
        episode_reward = 0
        
        # It's better to collect a trajectory of N steps or a full episode
        # For simplicity, let's assume we collect one episode then train
        
        while not done:
            action = agent.select_action(state)
            # next_state, reward, done, _ = dummy_env.step(action)
            next_state = np.random.rand(state_dim) # Dummy next state
            reward = np.random.rand()             # Dummy reward
            done = np.random.rand() > 0.95        # Dummy done
            
            agent.store_reward_done(reward, done) # Store r_t, d_t for (s_t, a_t)
            
            state = next_state
            episode_reward += reward
            if done:
                break
        
        agent.train() # Train at the end of the episode
        print(f"Episode {episode+1}, Reward: {episode_reward}")

    # agent.save("./models", "a2c_test")
    # agent.load("./models", "a2c_test")
