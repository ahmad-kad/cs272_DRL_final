# Urban Junction Autonomous Driving Environment

## Overview
The Urban Junction Environment is a reinforcement learning benchmark for training autonomous vehicles to navigate complex urban driving scenarios. It combines highway cruising, lane merging, and traffic light intersections with configurable adversarial traffic, providing a comprehensive testbed for developing robust driving policies.

## Objective
Train an autonomous vehicle to safely and efficiently navigate through multi-stage urban driving sequences consisting of:
1. **Highway segments** - Lane changes and speed maintenance
2. **Merge scenarios** - Lane merging with other traffic
3. **Intersection crossings** - Traffic light compliance and right-of-way

**Success**: Complete all stages without collision within time limits.

## State Space (What the Agent Observes)

### Primary Observation: Kinematics
- **8 vehicles** tracked simultaneously
- **5 features per vehicle**: [presence, relative_x, relative_y, relative_vx, relative_vy]
- **Temporal stacking**: 2 frames of history for motion prediction
- **Total dimensions**: 8 vehicles × 5 features × 2 frames = 80 features
- **Normalization**: All values scaled to [-1, 1] range

### Context Information (Optional)
- **3-dimensional one-hot encoding**: [is_highway, is_merge, is_intersection]
- **Purpose**: Enables scenario-specific behavior adaptation
- **Total dimensions**: 83 (80 kinematics + 3 context)

### Coordinate System
- **Relative positioning**: All coordinates relative to agent
- **Forward axis**: Positive x-direction is agent's heading
- **Normalization**: Positions and velocities scaled for neural network compatibility

## Action Space

### 5 Discrete Actions
1. **LANE_LEFT**: Change to left lane
2. **LANE_RIGHT**: Change to right lane  
3. **FASTER**: Increase target speed
4. **SLOWER**: Decrease target speed
5. **IDLE**: Maintain current lane and speed

### Action Execution
- Actions specify high-level driving intentions
- Continuous low-level control (steering, acceleration) handled by physics simulation
- Actions take effect over multiple timesteps as vehicle maneuvers

## Reward Structure

### Dense Reward Design
Rewards provided every timestep to enable stable learning:

| Situation | Reward | Notes |
|-----------|--------|-------|
| **Collision** | -1.0 | Terminal failure - episode ends |
| **Optimal Speed** (20-30 mph) | +0.4 | Efficient traffic flow |
| **Poor Speed** (too slow/fast) | -0.3 | Traffic disruption penalty |
| **Forward Progress** | +0.0 to +0.2 | Scaled by current speed |
| **Red Light Violation** | -0.4 | Safety and legal violation |
| **Green Light Compliance** | +0.1 | Proper traffic behavior |
| **Off-Road** | -0.3 | Lane discipline violation |
| **Stage Completion** | +0.5 | Milestone achievement |
| **Episode Success** | +2.0 | Full completion bonus |

### Reward Range: -1.0 to +0.7 per timestep

### Design Principles
- **Normalized magnitudes**: All rewards scaled to similar ranges for stable training
- **Dense feedback**: Continuous learning signals rather than sparse end-of-episode rewards
- **Safety prioritization**: Collision penalty dominates all other rewards
- **Multi-objective**: Balances safety, efficiency, and traffic compliance

## Constraints and Termination

### Episode Termination
- **Collision**: Any contact with other vehicles (reward = -1.0)
- **Success**: Complete all stages without crashing (reward = +2.0 bonus)

### Episode Truncation
- **Time Limit**: Maximum 200 timesteps
- **Distance Limit**: Complete all stages in sequence

### Vehicle Constraints
- **Physics**: Realistic car dynamics (acceleration limits, turning radius, momentum)
- **Lane Boundaries**: Off-road driving allowed but heavily penalized (-0.3/step)
- **Traffic Rules**: Must obey traffic lights and right-of-way conventions
- **Recovery**: Off-road incidents allow continued driving (not immediately terminal)

### Environmental Constraints
- **Road Layout**: 2-lane highway with junctions
- **Stage Lengths**: 100-200 meters per stage (configurable)
- **Traffic Density**: 8 vehicles total (ego + 7 others)
- **Traffic Light Timing**: Deterministic cycles (25 steps green, 3 yellow, 30 red)

## Traffic and Adversarial Behavior

### Vehicle Types
1. **Normal Traffic**: Intelligent Driver Model (IDM) vehicles following standard rules
2. **Antagonistic Vehicles**: Three types with increasing aggression levels:
   - **Swervers**: Unpredictable lane changes
   - **Cutters**: Aggressive merging and close following
   - **Random Drivers**: Erratic acceleration/deceleration

### Difficulty Scaling
- **Annoyance Level**: 0.0 (polite) to 1.0 (extremely aggressive)
- **Adaptive Curriculum**: Difficulty increases as agent improves
- **Configurable Ratios**: Mix of normal vs. antagonistic traffic

## Environment Configuration

### Stage Generation
- **Mode**: Deterministic (training) or random (generalization)
- **Sequence Length**: 2-5 stages per episode
- **Stage Types**: Highway, merge, intersection (configurable combinations)

### Simulation Parameters
- **Timestep**: 0.1 seconds (realistic physics)
- **Vehicle Dynamics**: Mass-based physics with realistic constraints
- **Sensor Range**: Observes vehicles within detection radius
- **Coordinate Frame**: Ego-centric with forward-facing bias