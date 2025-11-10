# Urban Junction Environment

**Goal**: Train autonomous vehicle to navigate urban scenarios safely.

## What Agent Sees
- **8 nearby vehicles**: position, speed, presence (40 features)
- **Optional context**: highway/merge/intersection mode (3 features)
- **Frame stacking**: 2 frames history (80-86 total features)
- **Normalized**: [-1, 1] range

## Actions (5 discrete)
- LANE_LEFT, LANE_RIGHT, FASTER, SLOWER, IDLE

## Rewards
- **Collision**: -1.0 (episode ends)
- **Good speed** (20-30 mph): +0.4
- **Bad speed**: -0.3
- **Progress**: +0.02 × speed
- **Red light violation**: -0.4
- **Off-road**: -0.3
- **Stage complete**: +0.5
- **Episode success**: +2.0

## Constraints
- **Terminate**: Collision only (off-road recovery allowed)
- **Truncate**: 200 steps or all stages complete
- **Traffic**: Normal + antagonistic vehicles (configurable difficulty)
- **Physics**: Realistic vehicle dynamics