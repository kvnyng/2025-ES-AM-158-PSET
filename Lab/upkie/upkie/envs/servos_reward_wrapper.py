#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward and termination wrapper for Upkie-Servos environment.

This wrapper adds reward shaping and termination conditions to the base
UpkieServos environment, similar to how UpkiePendulum wraps UpkieEnv.
"""

from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from upkie.logging import logger

from .upkie_servos import UpkieServos


class ServosRewardWrapper(gym.Wrapper):
    r"""!
    Wrapper that adds reward shaping and termination conditions to UpkieServos.
    
    This wrapper computes rewards based on the robot's balance state and
    applies termination conditions for falls and unsafe states.
    """
    
    def __init__(
        self,
        env: UpkieServos,
        fall_pitch: float = 1.0,
        max_position_drift: float = 5.0,
        max_angular_velocity: float = 10.0,
        max_linear_velocity: float = 2.0,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        r"""!
        Initialize the reward wrapper.
        
        \param env UpkieServos environment to wrap.
        \param fall_pitch Fall detection pitch angle threshold in radians.
        \param max_position_drift Maximum allowed position drift before termination (meters).
        \param max_angular_velocity Maximum allowed angular velocity before termination (rad/s).
        \param max_linear_velocity Maximum allowed linear velocity before termination (m/s).
        \param reward_weights Dictionary of reward weights for different terms.
        """
        super().__init__(env)
        
        self.fall_pitch = fall_pitch
        self.max_position_drift = max_position_drift
        self.max_angular_velocity = max_angular_velocity
        self.max_linear_velocity = max_linear_velocity
        
        # Default reward weights (can be overridden)
        default_weights = {
            "pitch": 0.5,
            "pitch_velocity": 0.1,
            "linear_velocity": 0.1,
            "action": 0.05,
        }
        if reward_weights is not None:
            default_weights.update(reward_weights)
        self.reward_weights = default_weights
        
        # Track initial position for drift detection
        self.initial_position: Optional[float] = None
        self.time_stamp = 0
        
        # Track previous action magnitude for smoothness penalty
        self._previous_action: Optional[float] = None
        self._previous_action_change: float = 0.0
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        """Reset the environment and track initial position."""
        observation, info = self.env.reset(seed=seed, options=options)
        
        # Extract initial position from spine observation
        spine_obs = info.get("spine_observation", {})
        if "wheel_odometry" in spine_obs:
            self.initial_position = spine_obs["wheel_odometry"].get("position", 0.0)
        else:
            self.initial_position = 0.0
        
        self.time_stamp = 0
        self._previous_action = None
        self._previous_action_change = 0.0
        
        return observation, info
    
    def _detect_fall(self, spine_observation: dict) -> bool:
        """Detect if the robot has fallen based on pitch angle."""
        if "base_orientation" not in spine_observation:
            return False
        
        pitch = spine_observation["base_orientation"].get("pitch", 0.0)
        if abs(pitch) > self.fall_pitch:
            logger.warning(
                "Fall detected (pitch=%.2f rad, fall_pitch=%.2f rad)",
                abs(pitch),
                self.fall_pitch,
            )
            return True
        return False
    
    def _extract_state_from_spine(self, spine_observation: dict) -> Dict[str, float]:
        """Extract relevant state variables from spine observation."""
        state = {
            "pitch": 0.0,
            "pitch_velocity": 0.0,
            "position": 0.0,
            "linear_velocity": 0.0,
        }
        
        # Extract pitch and pitch velocity from base_orientation
        if "base_orientation" in spine_observation:
            base_orient = spine_observation["base_orientation"]
            state["pitch"] = base_orient.get("pitch", 0.0)
            
            # Angular velocity is in base_orientation.angular_velocity
            # Pitch velocity is the y-component (lateral axis, index 1)
            if "angular_velocity" in base_orient:
                ang_vel = base_orient["angular_velocity"]
                if isinstance(ang_vel, (list, np.ndarray)) and len(ang_vel) >= 2:
                    state["pitch_velocity"] = float(ang_vel[1])
                elif isinstance(ang_vel, dict):
                    state["pitch_velocity"] = ang_vel.get("y", 0.0)
        
        # Extract position and linear velocity from wheel odometry
        if "wheel_odometry" in spine_observation:
            odom = spine_observation["wheel_odometry"]
            state["position"] = odom.get("position", 0.0)
            state["linear_velocity"] = odom.get("velocity", 0.0)
        
        return state
    
    def _extract_action_magnitude(self, action) -> float:
        """Extract action magnitude from action (handles both Box and Dict actions)."""
        if isinstance(action, dict):
            # Action is a Dict (from action wrapper) - extract magnitudes from servo commands
            total_mag = 0.0
            for servo_name, servo_action in action.items():
                if isinstance(servo_action, dict):
                    # Sum magnitudes of velocity and position commands
                    vel = servo_action.get("velocity", 0.0)
                    pos = servo_action.get("position", 0.0)
                    if not np.isnan(pos):
                        total_mag += abs(float(pos))
                    total_mag += abs(float(vel))
            return total_mag
        elif isinstance(action, np.ndarray):
            # Action is a numpy array (original policy action)
            return float(np.abs(action).sum())
        elif isinstance(action, (list, tuple)):
            return float(np.abs(np.asarray(action, dtype=np.float32)).sum())
        else:
            return 0.0
    
    def _compute_reward(
        self, state: Dict[str, float], action_mag: float
    ) -> float:
        """Compute reward based on state and action magnitude."""
        # Extract state variables
        pitch = state["pitch"]
        pitch_velocity = state["pitch_velocity"]
        linear_velocity = state["linear_velocity"]
        
        # Compute action smoothness penalty
        action_change = 0.0
        action_accel = 0.0
        
        if self._previous_action is not None:
            prev_mag = self._previous_action
            action_change = abs(action_mag - prev_mag)
            action_accel = abs(action_change - self._previous_action_change)
            self._previous_action_change = action_change
        
        self._previous_action = action_mag
        
        # Compute reward: start from baseline and subtract penalties
        reward = 1.0 - (
            self.reward_weights["pitch"] * pitch**2
            + self.reward_weights["pitch_velocity"] * pitch_velocity**2
            + self.reward_weights["linear_velocity"] * linear_velocity**2
            + self.reward_weights["action"] * action_mag**2
        )
        
        # Add smoothness penalty (smaller weight)
        smoothness_penalty = 0.01 * (action_change**2 + 0.5 * action_accel**2)
        reward -= smoothness_penalty
        
        # Clamp to reasonable range (VecNormalize will handle normalization)
        return float(np.clip(reward, -10.0, 10.0))
    
    def _check_termination_conditions(
        self, state: Dict[str, float], spine_observation: dict
    ) -> bool:
        """Check additional termination conditions beyond fall detection."""
        # Check position drift
        if self.initial_position is not None:
            position_drift = abs(state["position"] - self.initial_position)
            if position_drift > self.max_position_drift:
                logger.warning(
                    "Termination: excessive position drift (%.2f m > %.2f m)",
                    position_drift,
                    self.max_position_drift,
                )
                return True
        
        # Check angular velocity
        pitch_velocity_abs = abs(state["pitch_velocity"])
        if pitch_velocity_abs > self.max_angular_velocity:
            logger.warning(
                "Termination: excessive angular velocity (%.2f rad/s > %.2f rad/s)",
                pitch_velocity_abs,
                self.max_angular_velocity,
            )
            return True
        
        # Check linear velocity
        linear_velocity_abs = abs(state["linear_velocity"])
        if linear_velocity_abs > self.max_linear_velocity:
            logger.warning(
                "Termination: excessive linear velocity (%.2f m/s > %.2f m/s)",
                linear_velocity_abs,
                self.max_linear_velocity,
            )
            return True
        
        return False
    
    def step(
        self, action
    ) -> Tuple[dict, float, bool, bool, dict]:
        """Run one timestep with reward computation and termination checks."""
        # The action wrapper wraps this reward wrapper, so we receive the Dict action.
        # We need to extract action magnitude from the Dict for reward computation.
        action_mag = self._extract_action_magnitude(action)
        
        # Step the base environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract state from spine observation
        spine_observation = info.get("spine_observation", {})
        state = self._extract_state_from_spine(spine_observation)
        
        # Compute reward based on state and action magnitude
        reward = self._compute_reward(state, action_mag)
        
        # Check termination conditions
        if self._detect_fall(spine_observation):
            terminated = True
        elif self._check_termination_conditions(state, spine_observation):
            terminated = True
        
        # Update time stamp
        self.time_stamp += 1
        
        return observation, reward, terminated, truncated, info

