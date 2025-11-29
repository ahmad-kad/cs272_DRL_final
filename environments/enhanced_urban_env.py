"""
Enhanced Urban Junction Environment with Improved Reward Structure

This environment extends UrbanJunctionEnv with scenario-aware rewards that provide
denser feedback and better learning signals for multi-scenario autonomous driving.

Enhanced features:
- Scenario-aware speed optimization
- Lane position rewards for better lane keeping
- Progress tracking and completion bonuses
- Dense reward landscape for stable learning
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List, Callable
from environments.urban_junction_env import UrbanJunctionEnv


class EnhancedUrbanJunctionEnv(UrbanJunctionEnv):
    """
    Enhanced Urban Junction Environment with improved reward structure.

    This environment provides dense, scenario-aware rewards that help agents
    learn better driving policies across highway, merge, and intersection scenarios.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        scenario: str = "highway",
        modality: str = "both",
        render_mode: Optional[str] = None,
        reward_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize the enhanced environment.

        Args:
            config: Environment configuration
            scenario: Driving scenario
            modality: Observation modality
            render_mode: Rendering mode
            reward_weights: Custom reward component weights
        """
        super().__init__(config, scenario, modality, render_mode)

        # Enhanced reward weights with collision avoidance focus
        self.reward_weights = reward_weights or {
            # Core safety (balanced to allow emergency maneuvers)
            "collision_reward": -15.0,  # Reduced to allow emergency maneuvers
            "proximity_penalty": -1.5,  # Reduced proximity penalty
            "offroad_penalty": -6.0,    # Increased for better lane keeping
            "on_road_reward": 0.5,

            # Speed optimization
            "speed_reward": 2.0,
            "collision_avoidance_speed_bonus": 3.0,

            # Lane management with empty lane incentives
            "lane_position_reward": 0.5,
            "lane_change_penalty": -0.05,
            "safe_lane_change_bonus": 1.0,
            "empty_lane_bonus": 1.5,       # NEW: Reward choosing lanes with fewer vehicles
            "lane_traffic_penalty": -0.8,  # NEW: Penalty for heavy traffic lanes
            "passing_success_bonus": 1.0,  # NEW: Reward for successfully passing

            # Progress and completion
            "progress_reward": 0.3,
            "completion_bonus": 2.0,

            # Scenario-specific bonuses and penalties
            "merge_success_bonus": 1.5,
            "intersection_completion": 2.5,
            "intersection_offroad_penalty": -8.0,  # NEW: Harsher offroad penalty at intersections
            "corner_cutting_penalty": -2.0,        # NEW: Penalty for cutting corners

            # Behavioral shaping
            "time_penalty": -0.01
        }

        # Track previous state for progress and collision avoidance calculation
        self.prev_position = None
        self.prev_lane_index = None
        self.prev_speed = None
        self.episode_step_count = 0
        self.max_episode_steps = 1000

        # Safety constraint tracking
        self.safety_override_count = 0
        self.last_safety_override = None

    def _rewards(self, action: Union[int, np.ndarray]) -> Dict[str, float]:
        """
        Calculate enhanced reward components with dense feedback.

        This method provides scenario-aware rewards that guide learning
        toward optimal driving behavior in each scenario type.
        """
        rewards = {}

        # Core safety rewards with proximity awareness
        rewards["collision_reward"] = 1.0 if self.vehicle.crashed else 0.0
        rewards["proximity_penalty"] = self._get_proximity_penalty()  # NEW: Proactive collision avoidance
        is_offroad = self._is_offroad()
        rewards["on_road_reward"] = 1.0 if not is_offroad else 0.0
        rewards["offroad_penalty"] = 1.0 if is_offroad else 0.0

        # Enhanced speed reward (scenario-aware with collision avoidance)
        rewards["speed_reward"] = self._get_scenario_speed_reward()
        rewards["collision_avoidance_speed_bonus"] = 0.0  # Will be set in speed reward method

        # Lane position reward for better lane keeping
        rewards["lane_position_reward"] = self._get_lane_position_reward()

        # Progress reward for encouraging forward movement
        rewards["progress_reward"] = self._get_progress_reward()

        # Scenario-specific completion rewards
        completion_rewards = self._get_scenario_completion_rewards()
        rewards.update(completion_rewards)

        # Lane management (less restrictive for collision avoidance)
        rewards["lane_change_penalty"] = self._get_lane_change_penalty()
        rewards["safe_lane_change_bonus"] = self._get_safe_maneuver_bonus()

        # Empty lane and passing incentives
        rewards["empty_lane_bonus"] = self._get_empty_lane_bonus()
        rewards["lane_traffic_penalty"] = self._get_lane_traffic_penalty()
        rewards["passing_success_bonus"] = self._get_passing_success_bonus()

        # Intersection-specific penalties
        rewards["intersection_offroad_penalty"] = self._get_intersection_offroad_penalty()
        rewards["corner_cutting_penalty"] = self._get_corner_cutting_penalty()

        # Time penalty (encourage efficiency)
        rewards["time_penalty"] = -0.01

        # Update tracking variables
        self.prev_position = self.vehicle.position.copy() if hasattr(self.vehicle, 'position') else None
        self.prev_lane_index = self.vehicle.lane_index
        self.prev_speed = self.vehicle.speed
        self.episode_step_count += 1

        return rewards

    def _get_scenario_speed_reward(self) -> float:
        """
        Calculate scenario-aware speed reward with collision avoidance incentives.

        Returns optimal speed ranges for each scenario type, with bonuses for
        collision-avoiding speed adjustments.
        """
        if self._is_offroad() or self.vehicle.crashed:
            return -1.0  # Strong penalty for unsafe states

        speed = self.vehicle.speed
        base_reward = 0.0

        # Base speed optimization by scenario
        if self.current_scenario == "highway":
            # Highway: reward cruising speeds (20-30 km/h)
            optimal_speed = 25.0
            speed_range = 5.0  # Acceptable deviation
            if abs(speed - optimal_speed) <= speed_range:
                base_reward = 0.5 * (1 - abs(speed - optimal_speed) / speed_range)
            else:
                base_reward = -0.1  # Penalty for too slow/fast

        elif self.current_scenario == "merge":
            # Merge: reward moderate speeds for gap finding (15-25 km/h)
            optimal_speed = 20.0
            speed_range = 5.0
            if abs(speed - optimal_speed) <= speed_range:
                base_reward = 0.4 * (1 - abs(speed - optimal_speed) / speed_range)
            else:
                base_reward = -0.15

        elif self.current_scenario == "intersection":
            # Intersection: reward caution (8-15 km/h)
            optimal_speed = 12.0
            speed_range = 4.0
            if abs(speed - optimal_speed) <= speed_range:
                base_reward = 0.6 * (1 - abs(speed - optimal_speed) / speed_range)
            else:
                base_reward = -0.2  # Harsh penalty for reckless intersection speed

        else:
            # Default case
            base_reward = 0.0

        # Store collision avoidance bonus separately
        collision_avoidance_bonus = 0.0
        proximity_penalty = self._get_proximity_penalty()
        if proximity_penalty < -1.0:  # In dangerous proximity
            # Reward slowing down in dangerous situations
            if self.prev_speed is not None and speed < self.prev_speed:
                speed_reduction = self.prev_speed - speed
                if speed_reduction > 2.0:  # Significant slowing
                    collision_avoidance_bonus = min(1.0, speed_reduction / 10.0)

        # Update the collision avoidance bonus in rewards dict (hack for separate weighting)
        if hasattr(self, '_rewards_dict'):
            self._rewards_dict["collision_avoidance_speed_bonus"] = collision_avoidance_bonus

        return max(-1.0, min(1.0, base_reward))  # Clamp base reward

    def _get_lane_position_reward(self) -> float:
        """
        Calculate reward for proper lane positioning.

        Encourages staying in appropriate lanes for each scenario.
        """
        if not self.vehicle.lane_index or self._is_offroad():
            return -0.3  # Penalty for poor lane awareness

        try:
            # Get available lanes for current lane
            neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)

            if len(neighbours) <= 1:
                return 0.1  # Neutral for single-lane roads

            # Find current lane's position among neighbors
            current_lane_id = self.vehicle.lane_index[1] if len(self.vehicle.lane_index) > 1 else 0
            lane_idx = 0

            # Find our position in the neighbor list
            for i, lane in enumerate(neighbours):
                if lane[1] == current_lane_id:  # Compare lane IDs
                    lane_idx = i
                    break

            # Scenario-specific lane preferences
            if self.current_scenario == "highway":
                # Highway: prefer rightmost lanes (safer, easier passing)
                target_lane = len(neighbours) - 1
                lane_preference = 1.0 - abs(lane_idx - target_lane) / max(1, len(neighbours) - 1)
                return 0.3 * lane_preference

            elif self.current_scenario == "merge":
                # Merge: prefer lanes that lead to main road
                if not self._is_on_main_road():
                    return 0.2  # Small bonus for staying on ramp
                else:
                    # On main road, prefer center lanes
                    target_lane = len(neighbours) // 2
                    lane_preference = 1.0 - abs(lane_idx - target_lane) / max(1, len(neighbours) - 1)
                    return 0.25 * lane_preference

            elif self.current_scenario == "intersection":
                # Intersection: prefer lanes that lead through intersection
                # Center lanes usually go straight through
                target_lane = len(neighbours) // 2
                lane_preference = 1.0 - abs(lane_idx - target_lane) / max(1, len(neighbours) - 1)
                return 0.4 * lane_preference

            else:
                return 0.0

        except (IndexError, TypeError, AttributeError, KeyError):
            return -0.1  # Penalty for lane detection issues

    def _get_progress_reward(self) -> float:
        """
        Calculate progress reward based on forward movement and scenario goals.
        """
        if not hasattr(self.vehicle, 'position') or self.prev_position is None:
            return 0.0

        current_pos = np.array(self.vehicle.position)
        prev_pos = np.array(self.prev_position)

        if self.current_scenario == "highway":
            # Highway: reward longitudinal progress
            progress = current_pos[0] - prev_pos[0]
            return 0.1 * progress  # Small reward per unit of forward progress

        elif self.current_scenario == "merge":
            # Merge: reward progress toward main road
            if self._is_on_main_road():
                # On main road, reward forward progress
                progress = current_pos[0] - prev_pos[0]
                return 0.15 * progress
            else:
                # On ramp, reward any forward progress toward merge point
                progress = current_pos[0] - prev_pos[0]
                return 0.05 * progress

        elif self.current_scenario == "intersection":
            # Intersection: reward progress through intersection
            progress = current_pos[0] - prev_pos[0]
            # Extra reward for clearing intersection area
            if current_pos[0] > 15 and abs(current_pos[1]) < 5:
                progress *= 1.5  # Bonus multiplier
            return 0.08 * progress

        return 0.0

    def _get_scenario_completion_rewards(self) -> Dict[str, float]:
        """
        Calculate scenario-specific completion rewards.

        These provide clear goals for each scenario type.
        """
        rewards = {
            "merge_success_bonus": 0.0,
            "intersection_completion": 0.0,
            "completion_bonus": 0.0
        }

        if self.current_scenario == "merge":
            # Enhanced merge completion detection
            if self._is_successfully_merged():
                rewards["merge_success_bonus"] = 1.5
                rewards["completion_bonus"] = 1.0

        elif self.current_scenario == "intersection":
            # Enhanced intersection completion detection
            if self._has_cleared_intersection():
                rewards["intersection_completion"] = 2.5
                rewards["completion_bonus"] = 1.5

        return rewards

    def _get_lane_change_penalty(self) -> float:
        """
        Lightly penalize unnecessary lane changes to encourage stable driving.
        Much reduced to allow collision avoidance maneuvers.
        """
        if self.prev_lane_index is None or self.vehicle.lane_index is None:
            return 0.0

        # Check if lane changed (compare lane indices)
        if self.vehicle.lane_index != self.prev_lane_index:
            # Very small penalty for lane changes (reduced from -0.2)
            return -0.05
        else:
            # Small bonus for lane stability
            return 0.02

    def _get_proximity_penalty(self) -> float:
        """
        Penalize dangerous proximity to other vehicles based on lidar observations.
        This provides proactive collision avoidance incentives.
        """
        if self._is_offroad() or self.vehicle.crashed:
            return 0.0

        # Try to use lidar observations for proximity detection
        if hasattr(self, 'observation') and self.observation is not None:
            try:
                # For combined modality, extract lidar part
                if self.modality == "both":
                    # Assuming lidar comes first in the flattened observation
                    # This is a simplified version - would need proper observation parsing
                    lidar_size = 32 * 2  # 32 cells * 2 features (presence, x)
                    if len(self.observation) >= lidar_size:
                        lidar_obs = self.observation[:lidar_size].reshape(32, 2)
                        return self._calculate_lidar_proximity_penalty(lidar_obs)
                elif self.modality == "lidar":
                    if len(self.observation.shape) >= 2:
                        return self._calculate_lidar_proximity_penalty(self.observation)
            except Exception as e:
                # Fallback if observation parsing fails
                pass

        return 0.0

    def _calculate_lidar_proximity_penalty(self, lidar_obs) -> float:
        """
        Calculate proximity penalty from lidar observations with improved detection.
        """
        try:
            min_distance = float('inf')
            close_vehicle_count = 0
            very_close_count = 0

            # Check lidar cells for vehicle detections
            for i in range(len(lidar_obs)):
                if len(lidar_obs[i]) < 2:
                    continue

                presence = lidar_obs[i][0]  # Presence feature
                distance = abs(lidar_obs[i][1])  # Distance feature

                if presence > 0.3:  # Vehicle detected (lower threshold for sensitivity)
                    min_distance = min(min_distance, distance)

                    if distance < 20.0:  # Count vehicles within 20m
                        close_vehicle_count += 1
                        if distance < 8.0:  # Very close vehicles
                            very_close_count += 1

            if min_distance < float('inf'):
                # Exponential penalty for dangerously close vehicles
                if min_distance < 2.5:  # Critical danger zone (< 2.5 seconds at 30 km/h)
                    penalty = -8.0 * np.exp(-(min_distance - 1.0))  # Strong exponential penalty
                elif min_distance < 5.0:  # High danger zone (< 5 seconds at 30 km/h)
                    penalty = -4.0 * (1 - min_distance/5.0)**2  # Quadratic penalty
                elif min_distance < 10.0:  # Warning zone (< 10 seconds at 30 km/h)
                    penalty = -1.5 * (1 - min_distance/10.0)  # Linear penalty
                elif min_distance < 15.0:  # Caution zone
                    penalty = -0.3 * (1 - min_distance/15.0)  # Light penalty
                else:
                    penalty = 0.0
            else:
                penalty = 0.0

            # Additional penalties for traffic density
            if very_close_count > 0:
                penalty -= very_close_count * 1.0  # Extra penalty per very close vehicle

            if close_vehicle_count > 3:
                penalty -= 0.5 * (close_vehicle_count - 3)  # Crowd penalty

            # Scenario-specific adjustments
            if self.current_scenario == "intersection":
                penalty *= 1.5  # More cautious at intersections
            elif self.current_scenario == "merge":
                penalty *= 1.2  # More cautious during merging

            return penalty

        except Exception as e:
            # Fallback if lidar processing fails
            print(f"Warning: Lidar proximity calculation failed: {e}")
            return 0.0

    def _get_safe_maneuver_bonus(self) -> float:
        """
        Reward safe collision-avoidance maneuvers (lane changes and speed adjustments).
        Only rewards maneuvers when the vehicle was actually in danger.
        """
        bonus = 0.0

        # Only consider maneuvers after episode has started
        if self.episode_step_count < 5:
            return 0.0

        # Check if we were in a dangerous situation
        proximity_penalty = self._get_proximity_penalty()
        was_in_danger = proximity_penalty < -1.0  # Significant proximity penalty

        if was_in_danger:
            # Reward speed adjustments for collision avoidance
            if self.prev_speed is not None:
                speed_change = abs(self.vehicle.speed - self.prev_speed)
                if speed_change > 1.5:  # Significant speed adjustment
                    if self.vehicle.speed < self.prev_speed:  # Slowing down
                        bonus += 0.8  # Reward slowing down in danger
                    else:  # Speeding up
                        bonus += 0.4  # Reward speeding up to pass if appropriate

            # Reward lane changes for collision avoidance
            if (self.prev_lane_index is not None and
                self.vehicle.lane_index != self.prev_lane_index):
                bonus += 1.0  # Reward lane change when in danger

        return min(bonus, 2.0)  # Cap the bonus to prevent exploitation

    def _get_empty_lane_bonus(self) -> float:
        """
        Reward choosing lanes with fewer vehicles ahead.
        Uses lidar observations to detect traffic density in different lanes.
        """
        if self._is_offroad() or self.vehicle.crashed:
            return 0.0

        # Get vehicle counts per lane
        lane_vehicle_counts = self._count_vehicles_per_lane()
        if not lane_vehicle_counts:
            return 0.0

        current_lane = self.vehicle.lane_index
        if current_lane not in lane_vehicle_counts:
            return 0.0

        current_count = lane_vehicle_counts[current_lane]
        avg_count = sum(lane_vehicle_counts.values()) / len(lane_vehicle_counts)

        # Reward if current lane has fewer vehicles than average
        if current_count < avg_count:
            bonus = 1.5 * (avg_count - current_count) / max(avg_count, 1)
            return min(bonus, 2.0)  # Cap the bonus

        return 0.0

    def _get_lane_traffic_penalty(self) -> float:
        """
        Penalize choosing lanes with heavy traffic ahead.
        """
        if self._is_offroad() or self.vehicle.crashed:
            return 0.0

        lane_vehicle_counts = self._count_vehicles_per_lane()
        if not lane_vehicle_counts:
            return 0.0

        current_lane = self.vehicle.lane_index
        if current_lane not in lane_vehicle_counts:
            return 0.0

        current_count = lane_vehicle_counts[current_lane]
        max_count = max(lane_vehicle_counts.values())

        # Penalty proportional to traffic density in current lane
        if current_count > 2:  # Only penalize if lane has more than 2 vehicles
            penalty = -0.8 * (current_count - 2) / max(max_count, 3)
            return max(penalty, -2.0)  # Cap the penalty

        return 0.0

    def _get_passing_success_bonus(self) -> float:
        """
        Reward successfully passing slower vehicles.
        Detects when vehicle changes lanes and gains speed advantage.
        """
        bonus = 0.0

        # Need previous state for comparison
        if (self.episode_step_count < 10 or
            self.prev_lane_index is None or
            self.prev_speed is None):
            return 0.0

        # Check if we just changed lanes
        lane_changed = self.vehicle.lane_index != self.prev_lane_index

        if lane_changed:
            # Check if we're now going faster (successful pass)
            speed_improved = self.vehicle.speed > self.prev_speed + 2.0  # Significant speed gain

            # Check if we were behind slower vehicles before
            was_in_traffic = self._was_in_slow_traffic()

            if speed_improved and was_in_traffic:
                bonus = 1.0  # Reward successful passing maneuver

        return bonus

    def _get_intersection_offroad_penalty(self) -> float:
        """
        Harsher penalty for going offroad at intersections.
        """
        if self.current_scenario != "intersection":
            return 0.0

        is_offroad = self._is_offroad()
        if is_offroad:
            return 1.0  # Additional penalty at intersections (weighted by -8.0)

        return 0.0

    def _get_corner_cutting_penalty(self) -> float:
        """
        Penalize cutting corners at intersections by detecting sharp turns offroad.
        """
        if self.current_scenario != "intersection":
            return 0.0

        # Simple corner cutting detection: sharp lateral movement while offroad
        if (self._is_offroad() and
            hasattr(self.vehicle, 'position') and
            self.prev_position is not None):

            current_pos = np.array(self.vehicle.position)
            prev_pos = np.array(self.prev_position)

            # Calculate lateral movement (y-direction in intersection)
            lateral_movement = abs(current_pos[1] - prev_pos[1])

            # Penalize large lateral movements while offroad (corner cutting)
            if lateral_movement > 2.0:  # Significant lateral movement
                return 1.0  # Penalty for corner cutting (weighted by -2.0)

        return 0.0

    def _collision_imminent(self, threshold_distance: float = 5.0) -> bool:
        """
        Check if collision is imminent within threshold distance.

        Args:
            threshold_distance: Distance in meters to check for imminent collision

        Returns:
            True if collision is likely within the threshold distance
        """
        proximity_penalty = self._get_proximity_penalty()
        # Strong proximity penalty indicates very close vehicles
        return proximity_penalty < -4.0

    def _get_max_safe_speed(self) -> float:
        """Get maximum safe speed for current scenario."""
        base_speeds = {
            "highway": 35.0,      # Highway speed limit
            "merge": 25.0,        # Merging requires caution
            "intersection": 15.0  # Intersection caution
        }
        return base_speeds.get(self.current_scenario, 25.0)

    def _enforce_hard_safety_constraints(self, action: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Apply hard safety constraints that override RL actions when necessary.
        This provides a safety net while still allowing RL learning.
        Includes emergency maneuvers for merge scenarios.
        """
        original_action = action
        safety_override = False

        # Constraint 1: Emergency braking for imminent collision
        if self._collision_imminent(threshold_distance=3.0):
            # For merge scenarios, allow emergency lane changes/offroad first
            if self.current_scenario == "merge" and self._emergency_maneuver_available():
                action = self._emergency_maneuver_action()
                safety_override = "emergency_maneuver"
                print("🚨 SAFETY OVERRIDE: Emergency lane change/offroad for merge!")
            else:
                action = self._emergency_brake_action()
                safety_override = "emergency_brake"
                print("🚨 SAFETY OVERRIDE: Emergency braking for imminent collision!")

        # Constraint 2: Speed limiting
        elif hasattr(self.vehicle, 'speed') and self.vehicle.speed > self._get_max_safe_speed():
            action = self._speed_limit_action(action)
            safety_override = "speed_limit"
            print(".1f")

        # Constraint 3: Prevent complete road departure (except during emergency maneuvers)
        elif (hasattr(self.vehicle, 'lane_index') and
              self.vehicle.lane_index is None and
              self.last_safety_override != "emergency_maneuver"):
            action = self._road_recovery_action()
            safety_override = "road_recovery"
            print("🛣️ SAFETY OVERRIDE: Road departure prevention!")

        # Constraint 4: Lane keeping (prevent extreme lane deviation, relaxed for intersections)
        elif (hasattr(self.vehicle, 'lane_index') and
              self.vehicle.lane_index is not None and
              len(self.vehicle.lane_index) > 2):
            lane_deviation = abs(self.vehicle.lane_index[2])
            max_deviation = 6.0 if self.current_scenario == "intersection" else 4.0  # More lenient at intersections
            if lane_deviation > max_deviation:  # Way off lane center
                action = self._lane_correction_action()
                safety_override = "lane_correction"
                print(".1f")

        # Track safety overrides
        if safety_override:
            self.safety_override_count += 1
            self.last_safety_override = safety_override

        return action

    def _emergency_brake_action(self) -> Union[int, np.ndarray]:
        """Return emergency braking action."""
        # This depends on your action space - adjust accordingly
        # Assuming discrete actions where lower values = braking
        if hasattr(self.action_space, 'n'):  # Discrete action space
            return 0  # Assume 0 is maximum braking
        else:  # Continuous action space
            # Return action for maximum braking, minimal steering
            return np.array([0.0, 0.0])  # [steering, acceleration]

    def _speed_limit_action(self, original_action: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """Modify action to enforce speed limits."""
        if hasattr(self.action_space, 'n'):  # Discrete
            # Choose a less aggressive acceleration action
            return min(original_action, 2) if isinstance(original_action, (int, np.integer)) else 2
        else:  # Continuous
            # Reduce acceleration component
            if isinstance(original_action, np.ndarray) and len(original_action) >= 2:
                # Assume [steering, acceleration] format
                steering = original_action[0]
                acceleration = min(original_action[1], 0.2)  # Limit acceleration
                return np.array([steering, acceleration])
            return original_action

    def _road_recovery_action(self) -> Union[int, np.ndarray]:
        """Return action to recover to road."""
        if hasattr(self.action_space, 'n'):  # Discrete
            return 3  # Assume centering action
        else:  # Continuous
            # Try to center steering and maintain speed
            return np.array([0.0, 0.3])  # [steering, acceleration]

    def _lane_correction_action(self) -> Union[int, np.ndarray]:
        """Return action to correct lane position."""
        if hasattr(self.action_space, 'n'):  # Discrete
            return 4  # Assume lane correction action
        else:  # Continuous
            # Adjust steering toward lane center
            current_lane_deviation = self.vehicle.lane_index[2] if self.vehicle.lane_index else 0
            steering_correction = -np.sign(current_lane_deviation) * 0.5  # Steer toward center
            return np.array([steering_correction, 0.2])  # [steering, acceleration]

    def _emergency_maneuver_available(self) -> bool:
        """
        Check if emergency maneuver (lane change or offroad) is available.
        Only allows this in merge scenarios when no safe lane change exists.
        """
        if self.current_scenario != "merge":
            return False

        # Check if we have alternative lanes available
        try:
            if hasattr(self.road, 'network') and self.vehicle.lane_index:
                neighbors = self.road.network.all_side_lanes(self.vehicle.lane_index)
                return len(neighbors) > 1  # Can change lanes or go offroad
        except:
            pass

        return False  # No alternatives available

    def _emergency_maneuver_action(self) -> Union[int, np.ndarray]:
        """
        Return emergency maneuver action (lane change or controlled offroad).
        Prioritizes lane changes over offroad maneuvers.
        """
        if hasattr(self.action_space, 'n'):  # Discrete action space
            # Try lane change first (left or right)
            return 1  # Assume 1 is left lane change, or try different action
        else:  # Continuous action space
            # Emergency steering to change lanes or go offroad
            # Sharp steering maneuver with maintained speed
            emergency_steering = 1.0  # Max steering to change lanes
            return np.array([emergency_steering, 0.5])  # [steering, acceleration]

    def get_safety_stats(self):
        """Get safety constraint statistics."""
        return {
            "safety_override_count": self.safety_override_count,
            "last_safety_override": self.last_safety_override,
            "override_rate": self.safety_override_count / max(1, self.episode_step_count)
        }

    # Helper methods for scenario-specific logic

    def _is_on_main_road(self) -> bool:
        """Check if vehicle is on the main highway (not on ramp)."""
        if not self.vehicle.lane_index:
            return False
        try:
            # Main road lanes start with "a" or "b" in highway-env merge scenario
            return str(self.vehicle.lane_index[0]) in ["a", "b"]
        except (IndexError, TypeError):
            return False

    def _is_on_merge_ramp(self) -> bool:
        """Check if vehicle is on the merge ramp."""
        return not self._is_on_main_road()

    def _is_successfully_merged(self) -> bool:
        """Check if merge was completed successfully."""
        if not hasattr(self.vehicle, 'position'):
            return False

        # Check if on main road and has been there for several steps
        # This is a simplified version - could be enhanced with more sophisticated logic
        return self._is_on_main_road() and self.episode_step_count > 20

    def _has_cleared_intersection(self) -> bool:
        """Check if intersection has been cleared successfully."""
        if not hasattr(self.vehicle, 'position'):
            return False

        pos = self.vehicle.position
        # Check if vehicle has moved beyond intersection area
        # This assumes intersection center at (0,0) and clearing requires x > 15, |y| < 5
        return pos[0] > 15 and abs(pos[1]) < 5

    def reset(self, **kwargs):
        """Reset environment and tracking variables."""
        # Reset tracking variables
        self.prev_position = None
        self.prev_lane_index = None
        self.prev_speed = None
        self.episode_step_count = 0

        return super().reset(**kwargs)

    def step(self, action):
        """
        Enhanced step with hard safety constraints and comprehensive info tracking.
        """
        # Apply hard safety constraints BEFORE executing action
        safe_action = self._enforce_hard_safety_constraints(action)

        # Get reward breakdown before executing step
        reward_breakdown = self.get_reward_breakdown(action)

        # Execute the safe action
        result = super().step(safe_action)

        # Enhance info with comprehensive metrics
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            info.update(self._get_comprehensive_info(reward_breakdown))
            return obs, reward, terminated, truncated, info
        else:
            obs, reward, done, info = result
            info.update(self._get_comprehensive_info(reward_breakdown))
            return obs, reward, done, info

    def _get_comprehensive_info(self, reward_breakdown: Dict[str, float]) -> Dict[str, Any]:
        """
        Get comprehensive episode information for analysis and logging.
        """
        # Basic environment info
        info = {
            "scenario": self.current_scenario,
            "modality": self.modality,
            "episode_step": self.episode_step_count,
            "crashed": self.vehicle.crashed if hasattr(self.vehicle, 'crashed') else False,
        }

        # Safety override statistics
        safety_stats = self.get_safety_stats()
        info.update({
            f"safety/{k}": v for k, v in safety_stats.items()
        })

        # Vehicle state information
        if hasattr(self.vehicle, 'speed'):
            info["vehicle/speed"] = self.vehicle.speed
        if hasattr(self.vehicle, 'position') and self.vehicle.position is not None:
            info["vehicle/position_x"] = self.vehicle.position[0]
            info["vehicle/position_y"] = self.vehicle.position[1]
        if hasattr(self.vehicle, 'lane_index') and self.vehicle.lane_index is not None:
            info["vehicle/lane_index"] = str(self.vehicle.lane_index)

        # Reward component breakdown
        info.update({
            f"rewards/{k}": v for k, v in reward_breakdown.items()
        })

        # Scenario-specific progress metrics
        if self.current_scenario == "merge":
            info["scenario/on_main_road"] = self._is_on_main_road()
            info["scenario/successfully_merged"] = self._is_successfully_merged()
        elif self.current_scenario == "intersection":
            info["scenario/cleared_intersection"] = self._has_cleared_intersection()

        # Episode completion metrics
        info["episode/completed"] = self._episode_completed(self.current_scenario, self.episode_step_count, info["crashed"])

        return info

    def _reward(self, action: Union[int, np.ndarray]) -> float:
        """
        Calculate total reward with enhanced components.

        Uses weighted sum of reward components for dense feedback.
        """
        rewards = self._rewards(action)

        # Apply reward weights
        total_reward = 0.0
        for component, value in rewards.items():
            weight = self.reward_weights.get(component, 0.0)
            total_reward += weight * value

        # Clip reward for stability (important for PPO)
        total_reward = np.clip(total_reward, -5.0, 5.0)

        return float(total_reward)

    def get_reward_breakdown(self, action: Union[int, np.ndarray]) -> Dict[str, float]:
        """
        Get detailed breakdown of reward components for analysis.

        Useful for debugging and understanding agent behavior.
        """
        rewards = self._rewards(action)
        weighted_rewards = {}

        for component, value in rewards.items():
            weight = self.reward_weights.get(component, 0.0)
            weighted_rewards[f"{component}_raw"] = value
            weighted_rewards[f"{component}_weighted"] = weight * value

        weighted_rewards["total_reward"] = sum(weighted_rewards[k] for k in weighted_rewards.keys() if k.endswith("_weighted"))

        return weighted_rewards

    def _episode_completed(self, scenario: str, steps: int, crashed: bool) -> bool:
        """Check if episode represents successful completion."""
        if crashed:
            return False

        if scenario == "highway":
            return steps >= 150
        elif scenario == "merge":
            return steps >= 100
        elif scenario == "intersection":
            return steps >= 80
        return steps >= 100

    def _count_vehicles_per_lane(self) -> dict:
        """
        Count vehicles in each lane using lidar observations.
        Returns a dictionary mapping lane indices to vehicle counts.
        """
        lane_counts = {}

        # Get all available lanes
        if hasattr(self.road, 'network') and self.vehicle.lane_index:
            try:
                all_lanes = self.road.network.all_lanes()
                for lane in all_lanes:
                    lane_counts[lane.index] = 0
            except:
                # Fallback: estimate lanes from current lane
                if self.vehicle.lane_index not in lane_counts:
                    lane_counts[self.vehicle.lane_index] = 0

        # Use lidar to count vehicles in each lane
        if hasattr(self, 'observation') and self.observation is not None:
            try:
                # Extract lidar data (assuming lidar comes first in observation)
                if self.modality == "both":
                    lidar_size = 32 * 2  # 32 cells * 2 features
                    if len(self.observation) >= lidar_size:
                        lidar_obs = self.observation[:lidar_size].reshape(32, 2)
                        lane_counts = self._count_from_lidar(lidar_obs, lane_counts)
                elif self.modality == "lidar":
                    if len(self.observation.shape) >= 2:
                        lane_counts = self._count_from_lidar(self.observation, lane_counts)
            except:
                pass

        return lane_counts

    def _count_from_lidar(self, lidar_obs, lane_counts: dict) -> dict:
        """
        Count vehicles per lane from lidar observations using angle-based lane detection.
        """
        # Simple lane detection based on angle ranges
        # This is a simplified version - in practice you'd use more sophisticated lane detection
        lane_angles = {
            (0,): (-10, 10),    # Center lane
            (1,): (-30, -10),   # Left lane
            (2,): (10, 30),     # Right lane
        }

        for i in range(len(lidar_obs)):
            if len(lidar_obs[i]) < 2:
                continue

            presence = lidar_obs[i][0]
            angle = lidar_obs[i][1] if len(lidar_obs[i]) > 1 else 0

            if presence > 0.3:  # Vehicle detected
                # Determine which lane the vehicle is in based on angle
                for lane_idx, (min_angle, max_angle) in lane_angles.items():
                    if min_angle <= angle <= max_angle:
                        if lane_idx in lane_counts:
                            lane_counts[lane_idx] += 1
                        break

        return lane_counts

    def _was_in_slow_traffic(self) -> bool:
        """
        Check if vehicle was previously in slow traffic (behind slower vehicles).
        """
        if self.prev_speed is None:
            return False

        # Check if previous speed was significantly below optimal
        optimal_speed = self._get_scenario_speed_optimal()
        return self.prev_speed < optimal_speed * 0.7  # Below 70% of optimal speed

    def _get_scenario_speed_optimal(self) -> float:
        """Get optimal speed for current scenario."""
        if self.current_scenario == "highway":
            return 25.0
        elif self.current_scenario == "merge":
            return 20.0
        elif self.current_scenario == "intersection":
            return 12.0
        return 20.0
