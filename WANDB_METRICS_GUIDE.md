# WandB Metrics Guide for Agent Training

## Metric Groups and Chart Types

### 🔄 **TRAINING METRICS** (Line Plots - Training Progress)
These metrics track the overall training progress and should show as line plots over timesteps:

- **`timesteps`**: Total environment steps taken (primary x-axis)
- **`updates`**: Number of PPO updates performed
- **`step_count`**: Alias for updates (for clarity)
- **`env_steps`**: Steps per environment (timesteps / parallel_envs)
- **`parallel_envs`**: Number of parallel environments running
- **`episodes_this_interval`**: Episodes completed since last logging interval
- **`has_episodes`**: Binary flag (0/1) indicating if episode data is available

### 🧠 **PPO TRAINING METRICS** (Line Plots - Algorithm Learning)
These track the PPO algorithm's internal learning metrics:

- **`ppo_value_loss`**: Value function loss (should decrease over time)
- **`ppo_policy_loss`**: Policy gradient loss (should decrease over time)
- **`ppo_entropy_loss`**: Entropy loss (encourages exploration)
- **`ppo_approx_kl`**: Approximate KL divergence (measures policy change)
- **`ppo_clip_fraction`**: Fraction of clipped policy updates
- **`ppo_learning_rate`**: Current learning rate (may decay)
- **`ppo_n_updates`**: Running count of updates

### 🎯 **EPISODE PERFORMANCE METRICS** (Line Plots - Agent Performance)
These track the agent's actual performance in the environment:

- **`episode_mean_reward`**: Average reward over last 20 episodes
- **`episode_best_reward`**: Best reward in last 20 episodes
- **`episode_worst_reward`**: Worst reward in last 20 episodes
- **`episode_reward_std`**: Standard deviation of rewards (variability)
- **`all_time_best_reward`**: Best reward achieved so far
- **`episode_mean_length`**: Average episode length over last 20 episodes
- **`total_episode_count`**: Total episodes completed across all environments
- **`episode_success_rate`**: Fraction of episodes with positive reward
- **`learning_improvement`**: Improvement from early to recent episodes

### 📊 **EPISODE-BY-EPISODE METRICS** (Scatter/Line Plots - Individual Episodes)
Logged immediately when each episode completes:

- **`episode_reward`**: Reward for this specific episode
- **`episode_length`**: Length of this specific episode
- **`episode_env_id`**: Which environment this episode came from
- **`episode_total_count`**: Running total of episodes completed
- **`episode_recent_mean`**: Mean of last 10 episodes (when available)
- **`episode_recent_std`**: Std of last 10 episodes (when available)

### 🎯 **FINAL METRICS** (Summary Values)
Logged at training end and shown in WandB summary:

- **`final_avg_reward`**: Average reward over final episodes
- **`final_best_reward`**: Best reward achieved
- **`final_worst_reward`**: Worst reward achieved
- **`final_total_episodes`**: Total episodes completed
- **`final_reward_std`**: Standard deviation of final rewards
- **`final_eval_mean_reward`**: Mean reward from final evaluation
- **`final_eval_std_reward`**: Std of reward from final evaluation
- **`training_completed`**: Binary flag (1) when training finishes

### 📈 **MONITOR VERIFICATION METRICS** (Validation)
Logged from Monitor CSV files as verification:

- **`monitor_total_episodes`**: Episodes recorded by Monitor wrapper
- **`monitor_mean_reward`**: Mean reward from Monitor data
- **`monitor_reward_std`**: Reward std from Monitor data
- **`monitor_best_reward`**: Best reward from Monitor data
- **`monitor_worst_reward`**: Worst reward from Monitor data
- **`monitor_mean_length`**: Mean episode length from Monitor data
- **`monitor_length_std`**: Episode length std from Monitor data

## Expected WandB Dashboard Panels

### **Training Progress Panel**
- Line plot: `timesteps` vs `updates`
- Line plot: `episodes_this_interval` (shows episode completion rate)
- Line plot: `parallel_envs` (constant, shows parallelization)

### **PPO Learning Panel**
- Line plot: `ppo_value_loss` (should decrease)
- Line plot: `ppo_policy_loss` (should decrease)
- Line plot: `ppo_entropy_loss` (exploration indicator)
- Line plot: `ppo_clip_fraction` (clipping ratio)
- Line plot: `ppo_learning_rate` (learning rate schedule)

### **Reward Performance Panel**
- Line plot: `episode_mean_reward` (primary performance metric)
- Line plot: `episode_best_reward` (peak performance)
- Line plot: `all_time_best_reward` (overall best achieved)
- Line plot: `episode_reward_std` (performance consistency)
- Line plot: `learning_improvement` (learning progress)

### **Episode Statistics Panel**
- Line plot: `episode_mean_length` (episode duration)
- Line plot: `total_episode_count` (cumulative episodes)
- Line plot: `episode_success_rate` (success percentage)

### **Individual Episodes Panel**
- Scatter plot: `episode_reward` vs episode number
- Line plot: `episode_recent_mean` (smoothed performance)
- Bar chart: `episode_env_id` distribution (optional)

## Chart Configuration Tips

### For Line Plots (Time Series):
- X-axis: `timesteps` (most metrics) or episode count (episode metrics)
- Smoothing: Light smoothing (0.3-0.7) for noisy metrics
- Range: Auto-scale for most, fixed range for loss metrics (0-10)

### For Reward Metrics:
- Primary metric: `episode_mean_reward`
- Secondary: `episode_best_reward`, `all_time_best_reward`
- Goal: Increasing trend over time

### For PPO Metrics:
- Value Loss: Should decrease steadily
- Policy Loss: Should decrease steadily
- KL Divergence: Should stay within reasonable bounds (< 0.1)
- Clip Fraction: Should be reasonable (0.1-0.3)

## Troubleshooting Chart Issues

### If Seeing Bar Charts Instead of Lines:
- Check for string/categorical values in metrics
- Ensure all numeric values are floats/ints
- Remove any string identifiers from regular logging

### If Metrics Don't Appear:
- Check that metrics are logged with `step=` parameter
- Ensure WandB run is initialized before logging
- Verify metric names don't contain special characters

### If Charts Are Too Noisy:
- Use rolling averages for episode metrics
- Apply smoothing in WandB chart settings
- Log less frequently for very noisy metrics

## Key Performance Indicators

1. **Learning Success**: `episode_mean_reward` increasing over time
2. **Algorithm Health**: PPO losses decreasing, KL divergence stable
3. **Training Efficiency**: Episodes completing at reasonable rate
4. **Final Performance**: `final_eval_mean_reward` meets target
5. **Consistency**: Low `episode_reward_std` indicates stable performance
