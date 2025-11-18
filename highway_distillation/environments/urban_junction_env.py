"""
Urban Junction Environment: Production-quality benchmark for autonomous driving.
Combines highway, merge, and intersection scenarios with configurable difficulty.

Best practices:
- Clear separation of concerns
- Dependency injection over discovery
- Deterministic and reproducible (with optional randomization)
- Comprehensive logging and debugging
- Gymnasium API compliance
- Physics-respecting vehicle control
- Normalized dense reward structure for stable DRL training
- Randomized stage sequences for better generalization
"""

import logging
import numpy as np
from gymnasium import spaces
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.behavior import IDMVehicle, AggressiveVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.road import Road

logger = logging.getLogger(__name__)


class AntagonisticVehicle(AggressiveVehicle):
    """Single antagonistic vehicle with configurable behavior types."""

    def __init__(self, road, position, heading=0, speed=0, target_lane_index=None,
                 target_speed=None, route=None, enable_lane_change=True, timer=None,
                 data=None, behavior_type='random', annoyance_level=0.5):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed,
                        route, enable_lane_change, timer, data)
        # Enhanced behavior types for more realistic traffic
        self.available_behaviors = [
            'tailgater',      # Follows too closely
            'weaving',        # Unpredictable lane changes
            'sudden_brake',   # Emergency braking
            'speed_demon',    # Excessive speeding
            'random',         # Original erratic behavior
            'swerve',         # Original swerve behavior
            'cutoff'          # Original cutoff behavior
        ]
        self.behavior_type = behavior_type if behavior_type in self.available_behaviors else 'random'
        self.annoyance_level = annoyance_level
        self.behavior_timer = 0
        self.brake_cooldown = 0  # Prevent constant braking

    def act(self, action=None):
        """Enhanced antagonistic behavior logic with more realistic traffic patterns."""
        if self.annoyance_level < 0.1:
            return super().act(action)  # Normal behavior

        # Update cooldown timers
        if self.brake_cooldown > 0:
            self.brake_cooldown -= 1

        # Route to specific behavior implementations
        behavior_map = {
            'tailgater': self._act_tailgater,
            'weaving': self._act_weaving,
            'sudden_brake': self._act_sudden_brake,
            'speed_demon': self._act_speed_demon,
            'swerve': self._act_swerve,
            'cutoff': self._act_cutoff,
            'random': self._act_random
        }

        behavior_func = behavior_map.get(self.behavior_type, self._act_random)
        return behavior_func(action)

    def _act_swerve(self, action=None):
        """Unpredictable lane changes."""
        self.behavior_timer += 1
        swerve_freq = max(50, int(200 / (self.annoyance_level + 0.1)))

        if self.behavior_timer >= swerve_freq:
            self.behavior_timer = 0
            # Simple lane change attempt
            try:
                current_lane = self.target_lane_index[2] if self.target_lane_index else 0
                new_lane = 1 if current_lane == 0 else 0  # Toggle between lanes
                if hasattr(self.road, 'network') and self.road.network:
                    graph_key = self.lane_index[:2] if hasattr(self, 'lane_index') else (0, 0)
                    lanes = self.road.network.graph.get(graph_key, {})
                    if 0 <= new_lane < len(lanes):
                        self.target_lane_index = (self.lane_index[0], self.lane_index[1], new_lane)
            except:
                pass  # Skip if lane change fails

        return super().act(action)

    def _act_cutoff(self, action=None):
        """Aggressive merging behavior."""
        self.behavior_timer += 1
        cutoff_freq = max(30, int(150 / (self.annoyance_level + 0.1)))

        if self.behavior_timer >= cutoff_freq:
            self.behavior_timer = 0
            # Speed up to pass vehicles ahead
            self.target_speed = min(self.target_speed + self.annoyance_level * 3, 35)

        return super().act(action)

    def _act_random(self, action=None):
        """Erratic acceleration behavior."""
        self.behavior_timer += 1
        behavior_freq = max(20, int(100 / (self.annoyance_level + 0.1)))

        if self.behavior_timer >= behavior_freq:
            self.behavior_timer = 0
            # Random speed change
            speed_change = np.random.uniform(-8, 8) * self.annoyance_level
            self.target_speed = np.clip(self.target_speed + speed_change, 5, 40)

        return super().act(action)

    def _act_tailgater(self, action=None):
        """Follows too closely, creating dangerous situations."""
        self.behavior_timer += 1
        tailgate_freq = max(100, int(300 / (self.annoyance_level + 0.1)))

        if self.behavior_timer >= tailgate_freq:
            self.behavior_timer = 0
            # Speed up to close gap with vehicle ahead
            self.target_speed = min(self.target_speed + self.annoyance_level * 5, 35)

        return super().act(action)

    def _act_weaving(self, action=None):
        """Unpredictable lane weaving."""
        self.behavior_timer += 1
        weave_freq = max(30, int(120 / (self.annoyance_level + 0.1)))

        if self.behavior_timer >= weave_freq:
            self.behavior_timer = 0
            # Rapid lane changes
            try:
                current_lane = self.target_lane_index[2] if self.target_lane_index else 0
                new_lane = 1 if current_lane == 0 else 0
                if hasattr(self.road, 'network') and self.road.network:
                    graph_key = self.lane_index[:2] if hasattr(self, 'lane_index') else (0, 0)
                    lanes = self.road.network.graph.get(graph_key, {})
                    if 0 <= new_lane < len(lanes):
                        self.target_lane_index = (self.lane_index[0], self.lane_index[1], new_lane)
            except:
                pass

        return super().act(action)

    def _act_sudden_brake(self, action=None):
        """Emergency braking behavior."""
        if self.brake_cooldown == 0:
            brake_chance = self.annoyance_level * 0.1  # 10% chance at max annoyance
            if np.random.random() < brake_chance:
                # Emergency brake
                self.target_speed = max(self.target_speed * 0.3, 2)  # Sudden stop
                self.brake_cooldown = 50  # Cooldown before next brake

        return super().act(action)

    def _act_speed_demon(self, action=None):
        """Excessive speeding."""
        self.behavior_timer += 1
        speed_freq = max(200, int(500 / (self.annoyance_level + 0.1)))

        if self.behavior_timer >= speed_freq:
            self.behavior_timer = 0
            # Speed up dangerously
            self.target_speed = min(self.target_speed + self.annoyance_level * 8, 45)

        return super().act(action)


class TrafficLight:
    """Deterministic traffic light with configurable timing."""
    
    def __init__(self, green_time=25, yellow_time=3, red_time=30, start_state="green"):
        self.states = ['red', 'yellow', 'green']
        self.timers = {'red': red_time, 'yellow': yellow_time, 'green': green_time}
        self.current_state = start_state
        self.timer = self.timers[start_state]
        self.time_step = 0

    def update(self):
        """Progress to next state."""
        self.timer -= 1
        self.time_step += 1
        
        if self.timer <= 0:
            current_idx = self.states.index(self.current_state)
            self.current_state = self.states[(current_idx + 1) % 3]
            self.timer = self.timers[self.current_state]
        
        return self.get_state()

    def get_state(self):
        """Return state: 0=red, 1=yellow, 2=green."""
        return self.states.index(self.current_state)

    def reset(self, start_state="green"):
        """Reset to initial state."""
        self.current_state = start_state
        self.timer = self.timers[start_state]
        self.time_step = 0


class StageGenerator:
    """
    Generates randomized or deterministic stage sequences.
    
    Stage types:
    - highway: Standard cruising with traffic
    - merge: Lane merge scenario with aggressive merging
    - intersection: Traffic light with crossing vehicles
    """
    
    STAGE_TYPES = ['highway', 'merge', 'intersection']
    
    def __init__(self, mode='random', min_stages=2, max_stages=5, 
                 stage_length_range=(100, 200), seed=None):
        """
        Args:
            mode: 'random', 'deterministic', or 'curriculum'
            min_stages: Minimum number of stages per episode
            max_stages: Maximum number of stages per episode
            stage_length_range: (min, max) length in meters for each stage
            seed: Random seed for reproducibility
        """
        self.mode = mode
        self.min_stages = min_stages
        self.max_stages = max_stages
        self.stage_length_range = stage_length_range
        self.rng = np.random.RandomState(seed)
    
    def generate_sequence(self):
        """
        Generate a stage sequence.
        
        Returns:
            List of tuples: [(stage_type, length_meters), ...]
        """
        if self.mode == 'deterministic':
            return self._generate_deterministic()
        elif self.mode == 'curriculum':
            return self._generate_curriculum()
        else:  # random
            return self._generate_random()
    
    def _generate_deterministic(self):
        """Classic sequence: highway -> merge -> intersection."""
        return [
            ('highway', 300),
            ('merge', 300),
            ('intersection', 400),
        ]
    
    def _generate_random(self):
        """Fully randomized sequence."""
        num_stages = self.rng.randint(self.min_stages, self.max_stages + 1)
        sequence = []

        for _ in range(num_stages):
            stage_type = self.rng.choice(self.STAGE_TYPES)
            length = self.rng.randint(*self.stage_length_range)
            sequence.append((stage_type, length))
        
        return sequence
    
    def _generate_curriculum(self):
        """
        Curriculum: start easy, progressively add harder stages.
        
        Logic:
        - Always start with highway (easy)
        - Randomly add merge or intersection
        - Longer sequences = more difficulty
        """
        num_stages = self.rng.randint(self.min_stages, self.max_stages + 1)
        sequence = [('highway', self.rng.randint(*self.stage_length_range))]
        
        # Add progressively harder stages
        remaining_types = ['merge', 'intersection']
        
        for i in range(1, num_stages):
            # Bias toward harder stages as sequence progresses
            if i < num_stages - 1:
                stage_type = self.rng.choice(self.STAGE_TYPES)
            else:
                # End with a hard stage
                stage_type = self.rng.choice(remaining_types)
            
            length = self.rng.randint(*self.stage_length_range)
            sequence.append((stage_type, length))
        
        return sequence
    
    def set_seed(self, seed):
        """Update random seed."""
        self.rng = np.random.RandomState(seed)


class UrbanJunctionEnv(AbstractEnv):
    """
    Production-quality benchmark environment with randomized stage sequences.
    
    Key features:
    - Randomized or deterministic stage sequences
    - Highway, merge, and intersection scenarios
    - Configurable antagonistic traffic
    - Curriculum learning via annoyance parameter
    - Normalized dense reward structure
    - Support for multiple observation types
    """

    @classmethod
    def default_config(cls):
        """Simplified configuration - essential parameters only."""
        config = super().default_config()
        config.update({
            # Core environment settings
            "vehicles_count": 8,              # Traffic density (reduced for speed)
            "duration": 200,                  # Episode length
            "lanes_count": 2,

            # Observation settings
            "observation": {
                "type": "Kinematics",
                "multi_modal": False,         # Enable lidar + visual
                "lidar_rays": 64,
                "lidar_range": 50.0,
                "visual_width": 84,
                "visual_height": 84,
                "vehicles_count": 8,
                "features": ["presence", "x", "y", "vx", "vy"],
                "normalize": True,
            },

            # Action space
            "action": {"type": "DiscreteMetaAction"},

            # Stage generation
            "stage_mode": "random",          # 'random', 'deterministic', 'curriculum'
            "min_stages": 2,
            "max_stages": 4,

            # Rewards (normalized dense)
            "collision_reward": -1.0,        # Terminal failure
            "speed_reward": 0.4,             # Optimal speed (20-30 mph)
            "speed_penalty": -0.3,           # Poor speed
            "progress_reward": 0.2,          # Forward movement
            "traffic_light_penalty": -0.4,   # Red light violation
            "off_road_penalty": -0.3,        # Lane discipline
            "success_reward": 2.0,           # Episode completion bonus
            "stage_success_bonus": 5.0,     # Bonus for completing all three stages

            # Traffic difficulty
            "antagonistic_vehicles": False,  # Enable hard traffic
            "annoyance_level": 0.0,          # Difficulty scale (0.0-1.0)
            "modality_dropout": 0.0,         # Random modality dropout (0.0-1.0) for robustness

            # Technical defaults (hidden)
            **cls._technical_defaults()
        })
        return config

    @classmethod
    def _technical_defaults(cls):
        """Technical parameters users don't need to see."""
        return {
            "vehicles_density": 1.0,
            "ego_spacing": 7,
            "stage_length_range": [100, 200],
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            "traffic_light_green": 25,
            "traffic_light_yellow": 3,
            "traffic_light_red": 30,
            "crossing_vehicle_probability": 0.05,
            "crossing_vehicle_speed_range": [15, 25],
            "merge_aggression": 0.7,
        }

    def define_spaces(self):
        """Override to support multi-modal observations with consistent spaces."""
        # Always call parent to set up action space and basic attributes
        super().define_spaces()

        # For multi-modal, we need to temporarily create an environment to get the actual kinematics size
        if self.config["observation"].get("multi_modal", False):
            # Create a temporary environment to get the actual kinematics observation size
            import copy
            temp_config = copy.deepcopy(self.config)
            temp_config["observation"]["multi_modal"] = False  # Disable multi-modal temporarily
            temp_env = type(self)(temp_config)
            temp_obs, _ = temp_env.reset()
            kinematics_size = temp_obs.flatten().shape[0]
            temp_env.close()

            lidar_size = self.config["observation"]["lidar_rays"]
            visual_size = (self.config["observation"]["visual_height"] *
                          self.config["observation"]["visual_width"])

            total_size = kinematics_size + lidar_size + visual_size

            # Override with consistent observation space for multi-modal
            self._observation_space = spaces.Box(
                low=-np.inf,  # All modalities normalized to similar ranges
                high=np.inf,
                shape=(total_size,),
                dtype=np.float32
            )

    @property
    def observation_space(self):
        """Override observation_space property to ensure correct space for multi-modal."""
        if self.config["observation"].get("multi_modal", False):
            # Use the stored observation space from define_spaces
            return self._observation_space
        else:
            return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        """Setter for observation_space."""
        self._observation_space = value

    def _get_lidar_observation(self):
        """Generate simulated lidar observation."""
        rays = self.config["observation"]["lidar_rays"]
        max_range = self.config["observation"]["lidar_range"]

        # Simulate lidar by casting rays in all directions
        angles = np.linspace(0, 2*np.pi, rays, endpoint=False)
        distances = np.full(rays, max_range, dtype=np.float32)

        # Check distance to all vehicles and road boundaries
        ego_pos = self.vehicle.position
        ego_heading = self.vehicle.heading

        for i, angle in enumerate(angles):
            # Rotate angle by ego heading
            world_angle = angle + ego_heading

            # Cast ray and find closest intersection

            # Check intersections with vehicles
            for vehicle in self.road.vehicles:
                if vehicle is self.vehicle:
                    continue

                # Simple circle approximation for vehicle detection
                vehicle_dist = np.linalg.norm(vehicle.position - ego_pos)
                vehicle_angle = np.arctan2(vehicle.position[1] - ego_pos[1],
                                         vehicle.position[0] - ego_pos[0])

                # Check if vehicle is in this ray's cone
                angle_diff = abs(vehicle_angle - world_angle)
                angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Handle wraparound

                if angle_diff < np.pi / rays and vehicle_dist < distances[i]:
                    distances[i] = vehicle_dist

            # Check road boundaries (simplified)
            # This would need more complex geometry in a real implementation

        return distances

    def _get_visual_observation(self):
        """Generate simulated visual observation."""
        height = self.config["observation"]["visual_height"]
        width = self.config["observation"]["visual_width"]

        # Create a simple top-down grayscale rendering
        # This is a simplified simulation - real implementation would use pygame rendering
        image = np.zeros((height, width, 1), dtype=np.uint8)

        # Convert world coordinates to pixel coordinates
        pixels_per_meter = 4  # Approximate scaling
        ego_x, ego_y = self.vehicle.position
        ego_heading = self.vehicle.heading

        # Draw ego vehicle (bright white)
        ego_pixel_x = width // 2
        ego_pixel_y = height // 2
        if 0 <= ego_pixel_x < width and 0 <= ego_pixel_y < height:
            image[ego_pixel_y, ego_pixel_x] = 255

        # Draw other vehicles
        for vehicle in self.road.vehicles:
            if vehicle is self.vehicle:
                continue

            # Convert to ego-centric coordinates
            rel_x = vehicle.position[0] - ego_x
            rel_y = vehicle.position[1] - ego_y

            # Rotate by ego heading to get forward-facing view
            cos_h = np.cos(ego_heading)
            sin_h = np.sin(ego_heading)
            rotated_x = rel_x * cos_h + rel_y * sin_h
            rotated_y = -rel_x * sin_h + rel_y * cos_h

            # Convert to pixel coordinates (simplified perspective)
            pixel_x = int(width // 2 + rotated_x * pixels_per_meter)
            pixel_y = int(height // 2 + rotated_y * pixels_per_meter)

            # Draw if within bounds
            if 0 <= pixel_x < width and 0 <= pixel_y < height:
                image[pixel_y, pixel_x] = 200  # Gray for other vehicles

        # Draw road boundaries (simplified)
        # Horizontal road lines
        for y in [height//2 - 10, height//2 + 10]:
            if 0 <= y < height:
                image[y, :] = 100

        return image

    def _reset(self):
        """Initialize environment for new episode."""
        # Generate stage sequence
        self.stage_generator = StageGenerator(
            mode=self.config.get("stage_mode", "random"),
            min_stages=self.config.get("min_stages", 2),
            max_stages=self.config.get("max_stages", 5),
            stage_length_range=tuple(self.config.get("stage_length_range", [100, 200])),
            seed=None  # Use default seeding for now
        )
        self.stage_sequence = self.stage_generator.generate_sequence()
        
        # Build position boundaries for each stage
        self.stage_boundaries = []
        current_pos = 0
        for stage_type, length in self.stage_sequence:
            self.stage_boundaries.append({
                'type': stage_type,
                'start': current_pos,
                'end': current_pos + length,
                'completed': False
            })
            current_pos += length
        
        self.total_distance = current_pos

        # Initialize annoyance_level before vehicle creation
        self.annoyance_level = self.config.get("annoyance_level", 0.5)

        self._create_road()
        self._create_vehicles()
        
        # Traffic light (only active in intersection stages)
        self.traffic_light = TrafficLight(
            green_time=self.config["traffic_light_green"],
            yellow_time=self.config["traffic_light_yellow"],
            red_time=self.config["traffic_light_red"],
        )
        
        # Phase tracking
        self.current_stage_idx = 0
        self.phase = self.stage_boundaries[0]['type']
        self.previous_phase = self.phase
        self.stages_completed_count = 0
        
        # Difficulty tracking
        self.adaptive_difficulty = self.config.get("adaptive_difficulty", False)
        self.episode_reward = 0.0
        
        # Success tracking
        self.success = False
        self.distance_traveled = 0.0
        self.initial_position = None

        # Note: observation post-processing happens in step() and reset() wrappers

    def reset(self, **kwargs):
        """Override reset to handle multi-modal observations with dropout support."""
        obs, info = super().reset(**kwargs)

        # Post-process observation if multi-modal
        if self.config["observation"].get("multi_modal", False):
            # Get additional observations
            lidar_obs = self._get_lidar_observation()
            visual_obs = self._get_visual_observation()

            # Flatten and concatenate all modalities
            kinematics_flat = obs.flatten().astype(np.float32)
            lidar_norm = lidar_obs.astype(np.float32) / self.config["observation"]["lidar_range"]  # Normalize to [0, 1]
            visual_norm = visual_obs.flatten().astype(np.float32) / 255.0  # Normalize to [0, 1]


            # Apply dropout during training if enabled (for robustness)
            dropout_rate = self.config.get("modality_dropout", 0.0)
            kinematics_dropout_allowed = self.config.get("kinematics_dropout_allowed", False)

            # Check for evaluation-specific dropout masks
            dropout_mask = self.config.get("observation", {}).get("dropout_mask", {})

            if dropout_mask:
                # Evaluation mode: specific modality dropout
                if dropout_mask.get('kinematics', False):
                    kinematics_flat *= 0.0
                if dropout_mask.get('lidar', False):
                    lidar_norm *= 0.0
                if dropout_mask.get('visual', False):
                    visual_norm *= 0.0
            elif dropout_rate > 0.0 and np.random.random() < dropout_rate:
                # Training mode: random dropout for robustness
                dropout_choice = np.random.random()
                if dropout_choice < 0.25:
                    lidar_norm *= 0.0  # Drop lidar (sensor failure/occlusion)
                elif dropout_choice < 0.5:
                    visual_norm *= 0.0  # Drop visual (camera failure/dirt)
                elif dropout_choice < 0.75 and kinematics_dropout_allowed:
                    kinematics_flat *= 0.0  # Drop kinematics (force sensor independence)
                # If kinematics dropout not allowed or chosen, keep all modalities

            # Concatenate into single tensor
            obs = np.concatenate([kinematics_flat, lidar_norm, visual_norm])

        return obs, info

    def _create_road(self):
        """Create straight road with length based on stage sequence."""
        from highway_env.road.road import RoadNetwork
        
        # Road length = total stage lengths + buffer
        road_length = self.total_distance + 200
        
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                lanes=self.config["lanes_count"],
                length=road_length
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self):
        """Spawn ego vehicle and antagonistic traffic."""
        # Ego vehicle
        vehicle = Vehicle.create_random(
            self.road,
            speed=20.0,
            lane_id=0,
            spacing=self.config.get("ego_spacing", 7),
        )
        self.vehicle = self.action_type.vehicle_class(
            self.road, vehicle.position, vehicle.heading, vehicle.speed
        )
        self.controlled_vehicles = [self.vehicle]
        self.road.vehicles = [self.vehicle]
        
        # Store initial position for distance calculation
        self.initial_position = self.vehicle.position[0]
        
        # Traffic
        if self.config.get("antagonistic_vehicles", True):
            self._create_antagonistic_vehicles()
        else:
            self._create_standard_vehicles()

    def _create_antagonistic_vehicles(self):
        """Spawn antagonistic vehicles with different behavior types."""
        total_vehicles = self.config["vehicles_count"]

        # Simple distribution: mix of behavior types
        behaviors = ['swerve', 'cutoff', 'random']
        vehicle_behaviors = []

        # Distribute evenly across behavior types
        for i in range(total_vehicles):
            vehicle_behaviors.append(behaviors[i % len(behaviors)])
        
        # Create vehicles with injected annoyance
        created_count = 0
        failed_count = 0

        for behavior_type in vehicle_behaviors:
            try:
                # Create vehicle with random parameters
                lane_index = self.rng.choice(self.road.network.get_lanes())
                longitudinal = self.rng.uniform(0, self.road.network.length)
                speed = self.rng.uniform(15, 35)

                vehicle = AntagonisticVehicle(
                    road=self.road,
                    position=np.array([longitudinal, lane_index[1]]),
                    heading=0,
                    speed=speed,
                    target_lane_index=lane_index,
                    behavior_type=behavior_type,
                    annoyance_level=self.annoyance_level
                )

                self.road.vehicles.append(vehicle)
                created_count += 1

            except Exception as e:
                failed_count += 1

                # Fallback to IDM
                try:
                    fallback = IDMVehicle.create_random(
                        self.road,
                        spacing=1.0 / self.config.get("vehicles_density", 1.0)
                    )
                    fallback.randomize_behavior()
                    self.road.vehicles.append(fallback)
                    created_count += 1
                except Exception as fb_e:
                    logger.error(f"Fallback creation failed: {fb_e}")
        

    def _create_standard_vehicles(self):
        """Fallback: create only IDM vehicles."""
        for _ in range(self.config["vehicles_count"]):
            vehicle = IDMVehicle.create_random(
                self.road,
                spacing=1.0 / self.config.get("vehicles_density", 1.0)
            )
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

    def _update_phase(self):
        """Determine current stage based on ego position."""
        pos = self.vehicle.position[0]
        
        # Find current stage
        for idx, stage in enumerate(self.stage_boundaries):
            if stage['start'] <= pos < stage['end']:
                old_phase = self.phase
                self.phase = stage['type']
                self.current_stage_idx = idx
                
                # Mark previous stage as completed
                if idx > 0 and not self.stage_boundaries[idx - 1]['completed']:
                    self.stage_boundaries[idx - 1]['completed'] = True
                    self.stages_completed_count += 1
                
                # Log phase transition
                if self.phase != old_phase:
                    self.previous_phase = old_phase
                
                break
        
        # Check if past all stages
        if pos >= self.stage_boundaries[-1]['end']:
            if not self.stage_boundaries[-1]['completed']:
                self.stage_boundaries[-1]['completed'] = True
                self.stages_completed_count += 1

    def _spawn_crossing_vehicle(self):
        """Spawn a vehicle crossing perpendicular (intersection phase only)."""
        try:
            lanes = self.road.network.all_lanes()
            if not lanes:
                return
            
            lane = self.np_random.choice(lanes)
            crossing_vehicle = IDMVehicle.make_on_lane(
                self.road, lane,
                longitudinal=self.vehicle.position[0] + self.np_random.uniform(-30, 30),
                speed=self.np_random.uniform(*self.config["crossing_vehicle_speed_range"])
            )
            self.road.vehicles.append(crossing_vehicle)
            
        except Exception as e:
            pass

    def _get_min_vehicle_distance(self):
        """Get minimum distance to any other vehicle."""
        min_distance = float('inf')
        ego_pos = self.vehicle.position

        for vehicle in self.road.vehicles:
            if vehicle is self.vehicle:
                continue
            distance = np.linalg.norm(vehicle.position - ego_pos)
            min_distance = min(min_distance, distance)

        return min_distance if min_distance != float('inf') else 50.0  # Default max range

    def _is_lane_changing(self):
        """Check if vehicle is currently changing lanes."""
        if not hasattr(self, 'previous_lane'):
            self.previous_lane = getattr(self.vehicle, 'target_lane_index', None)
            return False

        current_lane = getattr(self.vehicle, 'target_lane_index', None)
        if current_lane and self.previous_lane:
            lane_changed = current_lane != self.previous_lane
            self.previous_lane = current_lane
            return lane_changed
        return False

    def _get_lane_change_safety_distance(self):
        """Get minimum distance during lane change."""
        # Simplified: check distance to vehicles in target lane
        min_distance = float('inf')
        ego_pos = self.vehicle.position

        for vehicle in self.road.vehicles:
            if vehicle is self.vehicle:
                continue
            # Check if vehicle is in similar lateral position (same lane area)
            lateral_distance = abs(vehicle.position[1] - ego_pos[1])
            if lateral_distance < 4.0:  # Within lane width
                longitudinal_distance = abs(vehicle.position[0] - ego_pos[0])
                min_distance = min(min_distance, longitudinal_distance)

        return min_distance if min_distance != float('inf') else 30.0

    def _get_merge_gap_ahead(self):
        """Get available gap for merging."""
        ego_pos = self.vehicle.position
        ego_lane = getattr(self.vehicle, 'target_lane_index', (0, 0, 0))

        # Find vehicles in the target lane ahead
        vehicles_ahead = []
        for vehicle in self.road.vehicles:
            if vehicle is self.vehicle:
                continue
            vehicle_lane = getattr(vehicle, 'target_lane_index', (0, 0, 0))
            if vehicle_lane[2] == ego_lane[2]:  # Same lane
                if vehicle.position[0] > ego_pos[0]:  # Ahead
                    vehicles_ahead.append(vehicle)

        if not vehicles_ahead:
            return 50.0  # No vehicles ahead

        # Find closest vehicle ahead
        closest_ahead = min(vehicles_ahead, key=lambda v: v.position[0])
        gap = closest_ahead.position[0] - ego_pos[0]

        return max(gap, 0)  # Ensure non-negative

    def _get_traffic_speed_ahead(self):
        """Get average speed of traffic ahead."""
        ego_pos = self.vehicle.position
        ego_lane = getattr(self.vehicle, 'target_lane_index', (0, 0, 0))

        vehicles_ahead = []
        for vehicle in self.road.vehicles:
            if vehicle is self.vehicle:
                continue
            vehicle_lane = getattr(vehicle, 'target_lane_index', (0, 0, 0))
            if vehicle_lane[2] == ego_lane[2]:  # Same lane
                if vehicle.position[0] > ego_pos[0]:  # Ahead
                    vehicles_ahead.append(vehicle.speed)

        if vehicles_ahead:
            return sum(vehicles_ahead) / len(vehicles_ahead)
        return 25.0  # Default traffic speed

    def _is_stage_success(self):
        """Check if all three stages (highway, merge, intersection) were completed."""
        if not hasattr(self, 'stages_completed') or not self.stages_completed:
            return False

        required_stages = {'highway', 'merge', 'intersection'}
        completed_stages = set(self.stages_completed.keys())
        return required_stages.issubset(completed_stages)

    def _reward(self, action):
        """Enhanced reward function with harsh penalties for real-world driving."""
        # Terminal crash penalty - highest priority
        if self.vehicle.crashed:
            return -5.0  # Much harsher crash penalty

        reward = 0.0
        speed = self.vehicle.speed

        # ===== SAFETY FIRST =====

        # Critical proximity penalties (most important)
        min_distance = self._get_min_vehicle_distance()
        if min_distance < 3.0:
            reward -= 2.5  # Critical proximity - almost crash
        elif min_distance < 5.0:
            reward -= 1.2  # Danger zone
        elif min_distance < 8.0:
            reward -= 0.5  # Warning distance
        elif min_distance > 15.0:
            reward += 0.1  # Good following distance

        # Lane change safety (extremely important)
        if self._is_lane_changing():
            lane_change_distance = self._get_lane_change_safety_distance()
            if lane_change_distance < 4.0:
                reward -= 2.0  # Unsafe lane change
            elif lane_change_distance < 6.0:
                reward -= 0.8  # Risky lane change

        # ===== HIGHWAY-SPECIFIC RULES =====

        if self.phase == "highway":
            # Stricter following distances at high speeds
            if speed > 25 and min_distance < 10.0:
                reward -= 1.0  # Highway tailgating
            elif speed > 20 and min_distance < 8.0:
                reward -= 0.6

            # Speed stability on highway
            if hasattr(self, 'previous_speed'):
                speed_change = abs(speed - self.previous_speed)
                if speed_change > 5.0:
                    reward -= 0.4  # Erratic speed changes

        # ===== MERGING RULES =====

        elif self.phase == "merge":
            # Gap acceptance and yielding
            gap_ahead = self._get_merge_gap_ahead()
            if gap_ahead < 12.0 and speed > 18:
                reward -= 1.5  # Too fast for available gap
            elif gap_ahead < 8.0:
                reward -= 2.0  # Impossible merge

            # Check for cutting off traffic
            traffic_speed = self._get_traffic_speed_ahead()
            if speed > traffic_speed + 8:
                reward -= 1.0  # Aggressive merging

        # ===== INTERSECTION RULES =====

        elif self.phase == "intersection":
            light_state = self.traffic_light.get_state()
            if light_state == 0:  # Red light
                if speed > 1.0:
                    reward -= 2.0  # Much harsher red light penalty
                else:
                    reward += 0.2  # Small bonus for stopping
            elif light_state == 1:  # Yellow light
                if speed > 5.0:
                    reward -= 1.2  # Yellow light caution
            elif light_state == 2:  # Green light
                if speed > 8.0:
                    reward += 0.15  # Green light compliance bonus

        # ===== BASIC DRIVING =====

        # Speed optimization with safety bounds
        if 18 <= speed <= 32:  # Slightly wider optimal range
            reward += 0.4
        elif speed > 35:
            reward -= 1.0  # Excessive speed penalty
        elif speed < 15 and self.phase != "intersection":
            reward -= 0.5  # Too slow penalty
        else:
            reward -= 0.4  # Mild speed penalty

        # Progress bonus (capped to prevent speed demons)
        progress_bonus = min(speed * 0.015, 0.5)
        reward += progress_bonus

        # Lane discipline - harsher
        if not self.vehicle.on_road:
            reward -= 1.0  # Much harsher off-road penalty

        # ===== LONG-TERM INCENTIVES =====

        # Survival bonus for long episodes
        survival_bonus = min(self.steps * 0.0003, 0.3)
        reward += survival_bonus

        # Track speed for next step
        self.previous_speed = speed

        return reward

    def _is_terminated(self):
        """Episode terminates on collision only (allow off-road recovery)."""
        return self.vehicle.crashed

    def _is_truncated(self):
        """Episode truncates at time limit or when all stages completed."""
        # Time limit truncation
        time_truncated = self.steps >= self.config["duration"]

        # Stage completion truncation
        pos = self.vehicle.position[0]
        stages_truncated = pos >= self.total_distance

        truncated = time_truncated or stages_truncated

        if truncated:
            # Success: completed ALL THREE stages without crashing and staying on road
            self.success = (self._is_stage_success() and
                          not self.vehicle.crashed and
                          self.vehicle.on_road)
            if self.success:
                # Higher success bonus for completing all stages
                stage_success_bonus = self.config.get("stage_success_bonus", 5.0)
                self.episode_reward += stage_success_bonus

        return truncated

    def step(self, action):
        """Override step to handle multi-modal observations with dropout support."""
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)

        # Post-process observation if multi-modal
        if self.config["observation"].get("multi_modal", False):
            # Get additional observations
            lidar_obs = self._get_lidar_observation()
            visual_obs = self._get_visual_observation()

            # Flatten and concatenate all modalities
            kinematics_flat = obs.flatten().astype(np.float32)
            lidar_norm = lidar_obs.astype(np.float32) / self.config["observation"]["lidar_range"]  # Normalize to [0, 1]
            visual_norm = visual_obs.flatten().astype(np.float32) / 255.0  # Normalize to [0, 1]

            # Apply dropout during training if enabled (for robustness)
            dropout_rate = self.config.get("modality_dropout", 0.0)
            kinematics_dropout_allowed = self.config.get("kinematics_dropout_allowed", False)

            # Check for evaluation-specific dropout masks
            dropout_mask = self.config.get("observation", {}).get("dropout_mask", {})

            if dropout_mask:
                # Evaluation mode: specific modality dropout
                if dropout_mask.get('kinematics', False):
                    kinematics_flat *= 0.0
                if dropout_mask.get('lidar', False):
                    lidar_norm *= 0.0
                if dropout_mask.get('visual', False):
                    visual_norm *= 0.0
            elif dropout_rate > 0.0 and np.random.random() < dropout_rate:
                # Training mode: random dropout for robustness
                dropout_choice = np.random.random()
                if dropout_choice < 0.25:
                    lidar_norm *= 0.0  # Drop lidar (sensor failure/occlusion)
                elif dropout_choice < 0.5:
                    visual_norm *= 0.0  # Drop visual (camera failure/dirt)
                elif dropout_choice < 0.75 and kinematics_dropout_allowed:
                    kinematics_flat *= 0.0  # Drop kinematics (force sensor independence)
                # If kinematics dropout not allowed or chosen, keep all modalities

            # Concatenate into single tensor
            obs = np.concatenate([kinematics_flat, lidar_norm, visual_norm])

        return obs, reward, terminated, truncated, info

    def _step(self, action):
        """Execute one environment step."""
        # Track stage completions for rewards
        prev_completed = self.stages_completed_count
        
        # Update traffic light
        self.traffic_light.update()
        
        # Update phase based on position
        self._update_phase()
        
        # Spawn crossing vehicles in intersection stages
        if self.phase == "intersection" and \
           self.np_random.random() < self.config["crossing_vehicle_probability"]:
            self._spawn_crossing_vehicle()
        
        # Execute parent step
        obs, reward, terminated, truncated, info = super()._step(action)
        
        # Add stage completion bonus
        stages_completed_this_step = self.stages_completed_count - prev_completed
        if stages_completed_this_step > 0:
            stage_bonus = self.config["stage_completion_reward"] * stages_completed_this_step
            reward += stage_bonus
        
        # Track cumulative reward
        self.episode_reward += reward
        
        # Track distance traveled
        if self.initial_position is not None:
            self.distance_traveled = self.vehicle.position[0] - self.initial_position
        
        # Adaptive difficulty
        if self.adaptive_difficulty and not terminated and not truncated:
            self._check_adaptive_difficulty()
        
        # Update info dict
        info.update({
            "phase": self.phase,
            "stage_index": self.current_stage_idx,
            "stages_completed": self.stages_completed_count,
            "total_stages": len(self.stage_boundaries),
            "stage_progress": self._get_stage_progress(),
            "traffic_light": self.traffic_light.current_state,
            "traffic_light_state": self.traffic_light.get_state(),
            "annoyance_level": self.annoyance_level,
            "episode_reward": self.episode_reward,
            "distance_traveled": self.distance_traveled,
            "success": self.success if (terminated or truncated) else False,
            "stage_sequence": [s['type'] for s in self.stage_boundaries],
        })
        
        # Post-process observation if multi-modal
        if self.config["observation"].get("multi_modal", False):
            # Get additional observations
            lidar_obs = self._get_lidar_observation()
            visual_obs = self._get_visual_observation()

            # Flatten and concatenate all modalities
            kinematics_flat = obs.flatten().astype(np.float32)
            lidar_norm = lidar_obs.astype(np.float32) / self.config["observation"]["lidar_range"]  # Normalize to [0, 1]
            visual_norm = visual_obs.flatten().astype(np.float32) / 255.0  # Normalize to [0, 1]

            # Apply dropout during training if enabled (for robustness)
            dropout_rate = self.config.get("modality_dropout", 0.0)
            kinematics_dropout_allowed = self.config.get("kinematics_dropout_allowed", False)

            # Check for evaluation-specific dropout masks
            dropout_mask = self.config.get("observation", {}).get("dropout_mask", {})

            if dropout_mask:
                # Evaluation mode: specific modality dropout
                if dropout_mask.get('kinematics', False):
                    kinematics_flat *= 0.0
                if dropout_mask.get('lidar', False):
                    lidar_norm *= 0.0
                if dropout_mask.get('visual', False):
                    visual_norm *= 0.0
            elif dropout_rate > 0.0 and np.random.random() < dropout_rate:
                # Training mode: random dropout for robustness
                dropout_choice = np.random.random()
                if dropout_choice < 0.25:
                    lidar_norm *= 0.0  # Drop lidar (sensor failure/occlusion)
                elif dropout_choice < 0.5:
                    visual_norm *= 0.0  # Drop visual (camera failure/dirt)
                elif dropout_choice < 0.75 and kinematics_dropout_allowed:
                    kinematics_flat *= 0.0  # Drop kinematics (force sensor independence)
                # If kinematics dropout not allowed or chosen, keep all modalities

            # Concatenate into single tensor
            obs = np.concatenate([kinematics_flat, lidar_norm, visual_norm])

        return obs, reward, terminated, truncated, info

    def _get_stage_progress(self):
        """Calculate progress through current stage (0.0 to 1.0)."""
        if self.current_stage_idx >= len(self.stage_boundaries):
            return 1.0
        
        stage = self.stage_boundaries[self.current_stage_idx]
        pos = self.vehicle.position[0]
        stage_length = stage['end'] - stage['start']
        
        if stage_length == 0:
            return 1.0
        
        progress = (pos - stage['start']) / stage_length
        return np.clip(progress, 0.0, 1.0)

    def _check_adaptive_difficulty(self):
        """Increase difficulty if agent is performing well."""
        threshold = self.config.get("performance_threshold", 20.0)
        max_annoyance = self.config.get("max_annoyance", 1.0)
        
        if self.episode_reward > threshold and self.annoyance_level < max_annoyance:
            # Increase annoyance gradually
            increase = 0.05
            old_annoyance = self.annoyance_level
            self.annoyance_level = min(max_annoyance, self.annoyance_level + increase)
            
            # Update all antagonistic vehicles
            for vehicle in self.road.vehicles[1:]:
                if isinstance(vehicle, AntagonisticVehicle):
                    vehicle.annoyance_level = self.annoyance_level
            
            logger.info(f"Difficulty increased: {old_annoyance:.2f} → {self.annoyance_level:.2f}")
            
            # Reset threshold for next tier
            self.config["performance_threshold"] += 10

    def _info(self, obs, action):
        """Additional environment info for logging and analysis."""
        info = super()._info(obs, action)
        return info