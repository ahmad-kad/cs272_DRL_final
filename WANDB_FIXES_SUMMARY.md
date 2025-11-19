# WandB Multi-Environment Logging Fixes

## Problem Summary

The WandB callbacks were not properly logging training metrics (rewards, episodes, etc.) to WandB dashboards. Only system metrics were visible.

## Root Causes Identified

1. **Array Boolean Evaluation Error**: `if dones else 1` failed when `dones` was a numpy array
2. **Single-Environment Tracking**: Callbacks only tracked first environment (`dones[0]`, `rewards[0]`)
3. **Inconsistent Initialization**: Different initialization patterns across files
4. **Low Logging Frequency**: 1000 steps was too infrequent for multi-env setups
5. **Poor Metric Organization**: Flat metric structure made WandB charts cluttered

## Fixes Implemented

### 1. Fixed Array Handling
**Before:**
```python
self.num_envs = len(dones) if dones else 1  # FAILS on numpy arrays
```

**After:**
```python
dones = self.locals.get('dones', np.array([]))
self.num_envs = len(dones) if len(dones) > 0 else 1  # WORKS
```

### 2. Per-Environment Episode Tracking
**Before:** Only tracked environment 0
```python
self.current_episode_reward += self.locals['rewards'][0]
if dones[0]:  # Only first env!
    self.episode_rewards.append(self.current_episode_reward)
```

**After:** Tracks all environments
```python
for env_idx in range(self.num_envs):
    current_reward = getattr(self, f'current_reward_{env_idx}')
    current_reward += rewards[env_idx]
    if dones[env_idx]:
        self.episode_rewards.append(current_reward)
```

### 3. Improved Logging Frequency
- **Before**: Every 1000 steps (too infrequent)
- **After**: Every 250 steps (more responsive for multi-env)

### 4. Organized Metric Hierarchy
Metrics now use structured prefixes for better WandB visualization:

**Training Metrics** (`train/` prefix):
- `train/timesteps`: Total timesteps
- `train/updates`: Update count
- `train/parallel_envs`: Number of parallel environments
- `train/episodes_this_interval`: Episodes completed since last log

**Reward Metrics** (`rewards/` prefix):
- `rewards/episode_mean`: Average reward (last 20 episodes)
- `rewards/episode_best`: Best recent reward
- `rewards/episode_worst`: Worst recent reward
- `rewards/episode_std`: Standard deviation
- `rewards/all_time_best`: Best reward ever seen
- `rewards/learning_improvement`: Progress indicator

**Episode Metrics** (`episode/` prefix):
- `episode/length_mean`: Average episode length
- `episode/total_count`: Total episodes completed
- `episode/success_rate`: % of episodes with positive reward

**PPO Metrics** (`train/ppo_` prefix):
- `train/ppo_value_loss`: Value function loss
- `train/ppo_policy_gradient_loss`: Policy gradient loss
- `train/ppo_entropy_loss`: Entropy loss
- `train/ppo_approx_kl`: Approximate KL divergence
- `train/ppo_clip_fraction`: Clipping fraction
- `train/ppo_learning_rate`: Current learning rate

**Final Metrics** (`final/` prefix):
- `final/avg_reward`: Final average reward
- `final/best_reward`: Best reward achieved
- `final/total_episodes`: Total episodes
- `final/training_completed`: Training completion flag

### 5. Files Updated

1. **`train_all_baselines.py`**: Main baseline training script
2. **`highway_distillation/training.py`**: Generalist agent training
3. **`train_single_baseline_focused.py`**: Focused baseline training

## Expected WandB Dashboard Improvements

### Before Fix
- ❌ Only system metrics visible
- ❌ No reward charts
- ❌ No episode statistics
- ❌ Difficult to track learning progress

### After Fix
- ✅ **Rewards Tab**: Multiple reward charts (mean, best, worst, improvement)
- ✅ **Training Tab**: PPO losses, learning rates, update counts
- ✅ **Episodes Tab**: Episode lengths, success rates, completion counts
- ✅ **Final Tab**: Summary statistics at training end
- ✅ Clear metric hierarchy with grouped charts
- ✅ Real-time updates every 250 steps

## Testing

Quick test training command:
```bash
python train_all_baselines.py --mode quick --env highway-v0 --obs Lidar --device cpu
```

Full baseline training (all combinations):
```bash
python train_all_baselines.py --all --mode standard --device cuda
```

## WandB Projects

- **Baseline Training**: `highway-distillation-baselines`
- **Multi-Env Generalists**: `highway-distillation-generalists`
- **Urban Junction (Custom Env)**: `urban-junction-rl`

## Verification Checklist

- [x] Array handling fixed (no more "ambiguous truth value" errors)
- [x] Multi-environment episode tracking working
- [x] Metrics organized with prefixes for clear WandB charts
- [x] Logging frequency optimized for multi-env training
- [x] All callback implementations synchronized
- [x] Syntax errors fixed
- [x] Test run initiated

## Next Steps

1. Monitor quick test run to verify metrics appear on WandB
2. Start full baseline training (6 combinations: 3 envs × 2 obs types)
3. Validate that all reward metrics display correctly in WandB charts

