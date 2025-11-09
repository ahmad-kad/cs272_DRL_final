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


class AntagonisticVehicle:
    """Mixin for antagonistic behavior. Injected annoyance level determines intensity."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Injected by environment at creation
        self.annoyance_level = 0.5
        self.behavior_state = "normal"  # Track current behavior mode
        self.behavior_timer = 0


class SwervingVehicle(AntagonisticVehicle, AggressiveVehicle):
    """Unpredictable lane changes. Intensity controlled by annoyance_level."""
    
    def __init__(self, road, position, heading=0, speed=0, target_lane_index=None,
                 target_speed=None, route=None, enable_lane_change=True, timer=None, data=None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed,
                        route, enable_lane_change, timer, data)
        self.swerve_timer = 0
        self.swerve_direction = 1

    def act(self, action=None):
        """Execute swerve behavior based on annoyance level."""
        if self.behavior_timer <= 0:
            # Swerve initiation chance increases with annoyance
            swerve_chance = 0.08 * (0.5 + self.annoyance_level)  # 4-12%
            
            if np.random.random() < swerve_chance and self.enable_lane_change:
                self.behavior_state = "swerving"
                self.behavior_timer = int(15 + 25 * (1 - self.annoyance_level))  # 15-40 steps
                self.swerve_direction = np.random.choice([-1, 1])
        
        # Execute swerve: use lane changes instead of position manipulation
        if self.behavior_timer > 0:
            self.behavior_timer -= 1
            current_lane = self.lane_index[2]
            new_lane = current_lane + self.swerve_direction
            
            # Check lane bounds with safety
            try:
                num_lanes = len(self.road.network.graph[self.lane_index[:2]])
                if 0 <= new_lane < num_lanes:
                    self.target_lane_index = (self.lane_index[0], self.lane_index[1], new_lane)
                    # Higher annoyance = more forceful lane changes
                    if np.random.random() < 0.3 * self.annoyance_level:
                        self.change_lane_policy()
            except (KeyError, IndexError):
                # Graceful fallback if lane structure is unexpected
                pass
        
        return super().act(action)


class CutoffVehicle(AntagonisticVehicle, AggressiveVehicle):
    """Aggressive merging and cutoffs. Safety margins decrease with annoyance."""
    
    def __init__(self, road, position, heading=0, speed=0, target_lane_index=None,
                 target_speed=None, route=None, enable_lane_change=True, timer=None, data=None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed,
                        route, enable_lane_change, timer, data)
        self.last_cutoff_time = -100

    def act(self, action=None):
        """Execute cutoff behavior. Aggression controlled by annoyance_level."""
        current_time = getattr(self.road, 'simulation_time', 0)
        
        # Cutoff frequency increases with annoyance
        cooldown = int(40 * (1 - 0.5 * self.annoyance_level))  # 20-40 steps
        
        if current_time - self.last_cutoff_time > cooldown:
            vehicles_ahead = [v for v in self.road.vehicles
                            if v is not self and v.position[0] > self.position[0]]
            
            for vehicle in vehicles_ahead:
                gap = vehicle.position[0] - self.position[0]
                
                # Safety margin shrinks with annoyance
                min_gap = 5 + (10 * (1 - self.annoyance_level))  # 5-15 meters
                max_gap = 15 + (10 * self.annoyance_level)  # 15-25 meters
                
                cutoff_prob = 0.05 * (0.5 + self.annoyance_level)  # 2.5-7.5%
                
                if min_gap < gap < max_gap and np.random.random() < cutoff_prob:
                    # Aggression: how much faster to go
                    speed_factor = 1.05 + (0.15 * self.annoyance_level)  # 1.05-1.2x
                    self.target_speed = min(self.MAX_SPEED, vehicle.speed * speed_factor)
                    self.last_cutoff_time = current_time
                    
                    # More likely to force lane change when very annoying
                    lane_offset_diff = np.linalg.norm(vehicle.lane_offset - self.lane_offset)
                    if lane_offset_diff < 1 and \
                       np.random.random() < (0.4 + 0.4 * self.annoyance_level):
                        self.change_lane_policy()
                    break
        
        return super().act(action)


class RandomMovementVehicle(AntagonisticVehicle, IDMVehicle):
    """Unpredictable acceleration/deceleration. Randomness controlled by annoyance."""
    
    def __init__(self, road, position, heading=0, speed=0, target_lane_index=None,
                 target_speed=None, route=None, enable_lane_change=True, timer=None, data=None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed,
                        route, enable_lane_change, timer)

    def act(self, action=None):
        """Execute random movement behavior."""
        # Random acceleration frequency increases with annoyance
        accel_prob = 0.03 * (0.5 + self.annoyance_level)  # 1.5-4.5%
        
        if np.random.random() < accel_prob:
            # Severity increases with annoyance
            speed_range = 0.7 + (0.4 * self.annoyance_level)  # 0.7-1.1
            random_factor = np.random.uniform(1 - speed_range, 1 + speed_range)
            self.target_speed = self.target_speed * random_factor
            self.target_speed = np.clip(self.target_speed, self.MIN_SPEED, self.MAX_SPEED)
        
        # Random lane changes (less frequent)
        lane_change_prob = 0.01 * (0.5 + self.annoyance_level)  # 0.5-1.5%
        if np.random.random() < lane_change_prob and self.enable_lane_change:
            self.change_lane_policy()
        
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
        config = super().default_config()
        config.update({
            # Observation - configurable by user for different DRL experiments
            "observation": {
                "type": "Kinematics",  # Base observation: "Kinematics", "LidarObservation", "GrayscaleObservation"
                "multi_modal": False,  # Enable multi-modal with lidar + visual
                "lidar_rays": 64,
                "lidar_range": 50.0,
                "visual_width": 84,
                "visual_height": 84,
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "normalize": True,
            },
            # Action
            "action": {
                "type": "DiscreteMetaAction",
            },
            # Environment
            "lanes_count": 2,
            "vehicles_count": 15,
            "vehicles_density": 1.0,
            "duration": 200,  # Steps per episode
            "ego_spacing": 7,
            
            # Stage generation
            "stage_mode": "random",  # 'random', 'deterministic', 'curriculum'
            "min_stages": 2,
            "max_stages": 5,
            "stage_length_range": [100, 200],  # Meters per stage
            
            # Normalized dense reward structure
            "collision_reward": 1.0,           # Terminal failure
            "speed_reward": 0.4,                # Optimal speed maintenance
            "speed_penalty_scale": 0.3,         # Sub-optimal speed
            "progress_reward": 0.2,             # Forward progress scaling
            "traffic_light_penalty": 0.4,       # Red light violation
            "traffic_light_reward": 0.1,        # Green light compliance
            "off_road_penalty": 0.3,            # Lane discipline
            "success_reward": 2.0,              # Episode completion bonus
            "stage_completion_reward": 0.5,     # Per-stage completion bonus
            
            # Speed targets
            "reward_speed_range": [20, 30],  # m/s optimal range
            
            # Antagonistic traffic
            "antagonistic_vehicles": True,
            "swerving_vehicle_ratio": 0.25,
            "cutoff_vehicle_ratio": 0.20,
            "random_vehicle_ratio": 0.15,
            
            # Difficulty curriculum
            "annoyance_level": 0.5,  # 0.0 (mild) to 1.0 (extreme)
            "adaptive_difficulty": False,
            "performance_threshold": 20.0,  # Trigger difficulty increase
            "max_annoyance": 1.0,
            
            # Traffic light (for intersection stages)
            "traffic_light_green": 25,
            "traffic_light_yellow": 3,
            "traffic_light_red": 30,
            
            # Cross-traffic (for intersection stages)
            "crossing_vehicle_probability": 0.05,
            "crossing_vehicle_speed_range": [15, 25],
            
            # Merge scenario
            "merge_aggression": 0.7,  # How aggressively vehicles merge
            
            "normalize_reward": True,
            "offroad_terminal": False,  # Allow recovery from off-road
        })
        return config

    def define_spaces(self):
        """Override to support multi-modal observations."""
        super().define_spaces()

        # Check if multi-modal observation is requested
        if self.config["observation"].get("multi_modal", False):
            # Create multi-modal observation space
            lidar_space = spaces.Box(
                low=0,
                high=self.config["observation"]["lidar_range"],
                shape=(self.config["observation"]["lidar_rays"],),
                dtype=np.float32
            )

            visual_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.config["observation"]["visual_height"],
                       self.config["observation"]["visual_width"], 1),
                dtype=np.uint8
            )

            # Get the original observation space for kinematics
            original_space = self.observation_space

            self.observation_space = spaces.Dict({
                'kinematics': original_space,
                'lidar': lidar_space,
                'visual': visual_space,
            })

    def _step(self, action):
        """Override step to handle multi-modal observations."""
        # Call parent step
        obs, reward, terminated, truncated, info = super()._step(action)

        # Post-process observation if multi-modal
        if self.config["observation"].get("multi_modal", False):
            # Get additional observations
            lidar_obs = self._get_lidar_observation()
            visual_obs = self._get_visual_observation()

            # Replace the observation with multi-modal version
            obs = {
                'kinematics': obs,  # Original observation becomes kinematics
                'lidar': lidar_obs,
                'visual': visual_obs,
            }

        return obs, reward, terminated, truncated, info

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
        
        logger.debug(f"Environment reset | Stages: {[s['type'] for s in self.stage_boundaries]} | "
                    f"Total distance: {self.total_distance}m | Annoyance: {self.annoyance_level:.2f}")

        # Note: observation post-processing happens in step() and reset() wrappers

    def reset(self, **kwargs):
        """Override reset to handle multi-modal observations."""
        obs, info = super().reset(**kwargs)

        # Post-process observation if multi-modal
        if self.config["observation"].get("multi_modal", False):
            # Get additional observations
            lidar_obs = self._get_lidar_observation()
            visual_obs = self._get_visual_observation()

            # Create multi-modal observation
            obs = {
                'kinematics': obs,  # Original observation becomes kinematics
                'lidar': lidar_obs,
                'visual': visual_obs,
            }

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
        """Spawn mix of antagonistic vehicle types with injected annoyance."""
        type_specs = [
            (SwervingVehicle, "swerving_vehicle_ratio", 0.25),
            (CutoffVehicle, "cutoff_vehicle_ratio", 0.20),
            (RandomMovementVehicle, "random_vehicle_ratio", 0.15),
        ]
        
        total_vehicles = self.config["vehicles_count"]
        vehicle_types = []
        total_allocated = 0
        
        # Allocate vehicles by type
        for vehicle_class, config_key, default_ratio in type_specs:
            ratio = self.config.get(config_key, default_ratio)
            count = int(total_vehicles * ratio)
            vehicle_types.extend([vehicle_class] * count)
            total_allocated += count
        
        # Fill remainder with IDM vehicles
        remaining = total_vehicles - total_allocated
        vehicle_types.extend([IDMVehicle] * remaining)
        
        # Shuffle for variety
        np.random.shuffle(vehicle_types)
        
        # Create vehicles with injected annoyance
        created_count = 0
        failed_count = 0
        
        for vehicle_class in vehicle_types:
            try:
                vehicle = vehicle_class.create_random(
                    self.road,
                    spacing=1.0 / self.config.get("vehicles_density", 1.0)
                )
                
                # Inject annoyance level for antagonistic vehicles
                if isinstance(vehicle, AntagonisticVehicle):
                    vehicle.annoyance_level = self.annoyance_level
                
                if hasattr(vehicle, 'randomize_behavior'):
                    vehicle.randomize_behavior()
                
                self.road.vehicles.append(vehicle)
                created_count += 1
                
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to create {vehicle_class.__name__}: {e}")
                
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
        
        logger.debug(f"Created {created_count} vehicles | Failed: {failed_count}")

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
                    logger.debug(f"Stage {idx - 1} completed: {self.stage_boundaries[idx - 1]['type']}")
                
                # Log phase transition
                if self.phase != old_phase:
                    logger.debug(f"Phase transition: {old_phase} -> {self.phase}")
                    self.previous_phase = old_phase
                
                break
        
        # Check if past all stages
        if pos >= self.stage_boundaries[-1]['end']:
            if not self.stage_boundaries[-1]['completed']:
                self.stage_boundaries[-1]['completed'] = True
                self.stages_completed_count += 1
                logger.debug(f"Final stage completed: {self.stage_boundaries[-1]['type']}")

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
            logger.debug(f"Could not spawn crossing vehicle: {e}")

    def _reward(self, action):
        """
        Calculate normalized dense reward for stable DRL training.
        All reward components scaled to similar magnitudes.
        
        Reward structure:
        - Collision: -1.0 (terminal)
        - Speed maintenance: +0.4 (optimal) to -0.3 (sub-optimal)
        - Progress: 0.0 to +0.2 (scaled by speed)
        - Traffic light compliance: -0.4 (violation) to +0.1 (compliance)
        - Lane discipline: -0.3 (off-road)
        - Stage completion: +0.5 per stage
        
        Total per-step range: approximately [-1.0, +0.7]
        """
        # Terminal collision penalty
        if self.vehicle.crashed:
            return -self.config["collision_reward"]
        
        reward = 0.0
        speed = self.vehicle.speed
        min_speed, max_speed = self.config["reward_speed_range"]
        
        # Speed reward: encourage optimal speed range
        if min_speed <= speed <= max_speed:
            reward += self.config["speed_reward"]
        else:
            # Penalty for sub-optimal speed (too slow or too fast)
            distance_from_range = min(abs(speed - min_speed), abs(speed - max_speed))
            penalty = (distance_from_range / 10.0) * self.config["speed_penalty_scale"]
            reward -= min(penalty, self.config["speed_penalty_scale"])
        
        # Progress reward: scaled by speed to encourage fast movement
        speed_ratio = np.clip(speed / max_speed, 0.0, 1.0)
        reward += self.config["progress_reward"] * speed_ratio
        
        # Traffic light compliance (intersection phase only)
        if self.phase == "intersection":
            light_state = self.traffic_light.get_state()
            if light_state == 0 and speed > 1.0:  # Running red light
                reward -= self.config["traffic_light_penalty"]
            elif light_state == 2 and speed > 5.0:  # Obeying green light with movement
                reward += self.config["traffic_light_reward"]
        
        # Lane discipline: penalty for going off-road
        if not self.vehicle.on_road:
            reward -= self.config["off_road_penalty"]
        
        # Stage completion bonus (awarded once per stage)
        # This is handled in _step() to avoid double counting
        
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
        
        if truncated and not self.vehicle.crashed:
            # Success: completed episode without crashing
            self.success = True
            # Add success bonus
            self.episode_reward += self.config["success_reward"]
            logger.debug(f"Episode success | Stages: {self.stages_completed_count}/{len(self.stage_boundaries)} | "
                        f"Total reward: {self.episode_reward:.2f}")
        
        return truncated

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
            logger.debug(f"Stage completion bonus: +{stage_bonus:.2f}")
        
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
            lidar_obs = self._get_lidar_observation()
            visual_obs = self._get_visual_observation()
            obs = {
                'kinematics': obs,
                'lidar': lidar_obs,
                'visual': visual_obs,
            }
        
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