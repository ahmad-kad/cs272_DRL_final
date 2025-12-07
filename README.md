````markdown
# CS272 DRL – Highway & CopChase Environments

Highway-style driving tasks (highway / merge / intersection)
Including Team 12's challenging CopChase “crazy driver” scenario

---

## Installation

```bash
pip install -r requirements.txt
````

(Optional) Set up Weights & Biases if you want logging.

---

## Environments

* **Highway / Merge / Intersection**
  Gymnasium IDs: `highway-v0`, `merge-v0`, `intersection-v0` (from `highway_env`).

  * Actions: 5 discrete controls – `LANE_LEFT`, `LANE_RIGHT`, `FASTER`, `SLOWER`, `IDLE`.
  * Observations:

    * `lidar`: compact vector with nearby vehicles.
    * `grayscale`: rendered top-down image.

* **MultiScenarioHighwayEnv** (`multi_scenario_env.py`)
  Single env that randomly switches between highway / merge / intersection with adjustable traffic aggressiveness.

* **CopChase-v0 (Crazy Driver)** (`team2_env/`)
  Custom environment with pursuing police cars and dense traffic.

  * Continuous actions (steering, throttle).
  * Kinematic observation with a Transformer feature extractor and reward wrapper to discourage crash exploitation.

---

## Training & Evaluation Commands

Recommended run order, numbered for reference.

### Single-scenario PPO (Highway / Merge / Intersection)

```bash
# 1. Train highway-v0, lidar
python train_gpu.py highway lidar --steps 500000

# 2. Evaluate highway-v0, lidar
python evaluate_single.py highway lidar \
  --model outputs/models/highway-v0_lidar_500k.zip \
  --plot_id 2 --episodes 500

# 3. Train highway-v0, grayscale
python train_gpu.py highway grayscale --steps 500000

# 4. Evaluate highway-v0, grayscale
python evaluate_single.py highway grayscale \
  --model outputs/models/highway-v0_grayscale_500k.zip \
  --plot_id 4 --episodes 500

# 5. Train merge-v0, lidar
python train_gpu.py merge lidar --steps 500000

# 6. Evaluate merge-v0, lidar
python evaluate_single.py merge lidar \
  --model outputs/models/merge-v0_lidar_500k.zip \
  --plot_id 6 --episodes 500

# 7. Train merge-v0, grayscale
python train_gpu.py merge grayscale --steps 500000

# 8. Evaluate merge-v0, grayscale
python evaluate_single.py merge grayscale \
  --model outputs/models/merge-v0_grayscale_500k.zip \
  --plot_id 8 --episodes 500

# 9. Train intersection-v0, lidar
python train_gpu.py intersection lidar --steps 500000

# 10. Evaluate intersection-v0, lidar
python evaluate_single.py intersection lidar \
  --model outputs/models/intersection-v0_lidar_500k.zip \
  --plot_id 10 --episodes 500

# 11. Train intersection-v0, grayscale
python train_gpu.py intersection grayscale --steps 500000

# 12. Evaluate intersection-v0, grayscale
python evaluate_single.py intersection grayscale \
  --model outputs/models/intersection-v0_grayscale_500k.zip \
  --plot_id 12 --episodes 500
```

### Multi-scenario PPO

```bash
# 13. Train multi-scenario PPO (lidar)
python train_multi_scenario.py

# 14. Evaluate multi-scenario PPO
python evaluate_multi.py \
  --model outputs/models/multi_scenario_lidar_500k.zip \
  --plot_id 14 --episodes 500 --aggressiveness 1.0
```

### Crazy Driver / CopChase SAC

```bash
# 15. Train SAC + Transformer on CopChase-v0
python train_crazy_driver_sac.py --n_envs 4 --steps 500000

# 16. Visualize CopChase with a trained model
python visualize_copchase.py \
  --model outputs/models/crazy_driver_sac_safe_navigation-500k.zip
```

---

## Project Layout

```text
.
├── train.py                      # Simple single-env PPO training (CPU)
├── train_gpu.py                  # Vectorized PPO training for highway/merge/intersection
├── train_multi_scenario.py       # PPO on MultiScenarioHighwayEnv
├── evaluate_single.py            # Evaluation + plotting for single scenarios
├── evaluate_multi.py             # Evaluation + plotting for multi-scenario PPO
├── visualize.py                  # Visualize single-scenario PPO policies
├── visualize_multi_scenario.py   # Visualize multi-scenario PPO policy
├── custom_policies.py            # Transformer feature extractor for CopChase
├── train_crazy_driver_sac.py     # SAC training for CopChase-v0
├── visualize_copchase.py         # Rollout / GIFs for CopChase-v0
├── multi_scenario_env.py         # MultiScenarioHighwayEnv definition
├── intersection_helpers.py       # Intersection safety wrapper
├── multi_wandb_helpers.py        # W&B callbacks for multi-scenario training
├── wandb_single_helpers.py       # W&B callbacks for single-scenario training
├── team2_env/
│   ├── crazy_driver_enviornment.py  # CopChase-v0 environment
│   └── reward_wrapper.py            # Reward shaping for CopChase-v0
└── requirements.txt
```

Models and plots are saved under `outputs/` (created automatically); W&B logging can be toggled with the `--no_wandb` flags in the training scripts.
