#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria

## \namespace upkie.envs.upkie_pendulum
## \brief Environment where Upkie behaves like a wheeled inverted pendulum.

import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from upkie.config import ROBOT_CONFIG
from upkie.envs.upkie_env import UpkieEnv
from upkie.exceptions import UpkieException
from upkie.logging import logger
from upkie.utils.clamp import clamp_and_warn
from upkie.utils.filters import low_pass_filter


class UpkiePendulum(gym.Wrapper):
    r"""!
    Wrapper to make Upkie act as a wheeled inverted pendulum.

    \anchor upkie_pendulum_description

    When this wrapper is applied, Upkie keeps its legs straight and actions
    only affect wheel velocities. This way, it behaves like a <a
    href="https://scaron.info/robotics/wheeled-inverted-pendulum-model.html">wheeled
    inverted pendulum</a>.

    \note For reinforcement learning with neural-network policies: the
    observation space and action space are not normalized.

    ### Action space

    The action corresponds to the ground velocity resulting from wheel
    velocities. The action vector is simply:

    \f[
    a =\begin{bmatrix} \dot{p}^* \end{bmatrix}
    \f]

    where we denote by \f$\dot{p}^*\f$ the commanded ground velocity in m/s,
    which is internally converted into wheel velocity commands. Note that,
    while this action is not normalized, [-1, 1] m/s is a reasonable range for
    ground velocities.

    ### Observation space

    Vectorized observations have the following structure:

    \f[
    \begin{align*}
    o &= \begin{bmatrix} \theta \\ p \\ \dot{\theta} \\ \dot{p} \end{bmatrix}
    \end{align*}
    \f]

    where we denote by:

    - \f$\theta\f$ the pitch angle of the base with respect to the world
      vertical, in radians. This angle is positive when the robot leans
      forward.
    - \f$p\f$ the position of the average wheel contact point, in meters.
    - \f$\dot{\theta}\f$ the body angular velocity of the base frame along its
      lateral axis, in radians per seconds.
    - \f$\dot{p}\f$ the velocity of the average wheel contact point, in meters
      per seconds.

    As with all Upkie environments, full observations from the spine (detailed
    in \ref observations) are also available in the `info` dictionary
    returned by the reset and step functions.
    """

    ## \var action_space
    ## Action space.
    action_space: gym.spaces.Box

    ## \var env
    ## Internal \ref upkie.envs.upkie_env.UpkieEnv environment.
    env: UpkieEnv

    ## \var fall_pitch
    ## Fall detection pitch angle, in radians.
    fall_pitch: float

    ## \var left_wheeled
    ## Set to True (default) if the robot is left wheeled, that is, a positive
    ## turn of the left wheel results in forward motion. Set to False for a
    ## right-wheeled variant.
    left_wheeled: bool

    ## \var observation_space
    ## Observation space.
    observation_space: gym.spaces.Box

    def __init__(
        self,
        env: UpkieEnv,
        fall_pitch: float = 1.0,
        left_wheeled: bool = True,
        max_ground_velocity: float = 1.0,
        max_position_drift: float = 5.0,
        max_angular_velocity: float = 10.0,
        max_linear_velocity: float = 2.0,
    ):
        r"""!
        Initialize environment.

        \param env Upkie environment to command servomotors.
        \param fall_pitch Fall detection pitch angle, in radians.
        \param left_wheeled Set to True (default) if the robot is left wheeled,
            that is, a positive turn of the left wheel results in forward
            motion. Set to False for a right-wheeled variant.
        \param max_ground_velocity Maximum commanded ground velocity in m/s.
            The default value of 1 m/s is conservative, don't hesitate to
            increase it once you feel confident in your agent.
        \param max_position_drift Maximum allowed position drift from origin
            before termination, in meters.
        \param max_angular_velocity Maximum allowed angular velocity before
            termination, in rad/s.
        \param max_linear_velocity Maximum allowed linear velocity before
            termination, in m/s.
        """
        super().__init__(env)
        if env.frequency is None:
            raise UpkieException("This environment needs a loop frequency")

        MAX_BASE_PITCH: float = np.pi
        MAX_GROUND_POSITION: float = float("inf")
        MAX_BASE_ANGULAR_VELOCITY: float = 1000.0  # rad/s
        observation_limit = np.array(
            [
                MAX_BASE_PITCH,
                MAX_GROUND_POSITION,
                MAX_BASE_ANGULAR_VELOCITY,
                max_ground_velocity,
            ],
            dtype=np.float32,
        )
        action_limit = np.array([max_ground_velocity], dtype=np.float32)

        # gymnasium.Env: observation_space
        self.observation_space = gym.spaces.Box(
            -observation_limit,
            +observation_limit,
            shape=observation_limit.shape,
            dtype=observation_limit.dtype,
        )

        # gymnasium.Env: action_space
        self.action_space = gym.spaces.Box(
            -action_limit,
            +action_limit,
            shape=action_limit.shape,
            dtype=action_limit.dtype,
        )

        # Instance attributes
        self.__leg_servo_action = env.get_neutral_action()
        self.env = env
        self.fall_pitch = fall_pitch
        self.left_wheeled = left_wheeled
        self.max_position_drift = max_position_drift
        self.max_angular_velocity = max_angular_velocity
        self.max_linear_velocity = max_linear_velocity
        self.env.max_time_steps = 300
        
        # Track previous actions for smoothness penalty (velocity and acceleration)
        self.__previous_action = np.array([0.0])
        self.__previous_action_change = 0.0  # Track action velocity for acceleration penalty

    def __get_env_observation(self, spine_observation: dict) -> np.ndarray:
        r"""!
        Extract environment observation from spine observation dictionary.

        \param spine_observation Spine observation dictionary.
        \return Environment observation vector.
        """
        base_orientation = spine_observation["base_orientation"]
        pitch_base_in_world = base_orientation["pitch"]
        angular_velocity_base_in_base = base_orientation["angular_velocity"]
        ground_position = spine_observation["wheel_odometry"]["position"]
        ground_velocity = spine_observation["wheel_odometry"]["velocity"]

        obs = np.empty(4, dtype=np.float32)
        obs[0] = pitch_base_in_world
        obs[1] = ground_position
        obs[2] = angular_velocity_base_in_base[1]
        obs[3] = ground_velocity
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        r"""!
        Resets the environment and get an initial observation.

        \param seed Number used to initialize the environment's internal random
            number generator.
        \param options Currently unused.
        \return
            - `observation`: Initial vectorized observation, i.e. an element
              of the environment's `observation_space`.
            - `info`: Dictionary with auxiliary diagnostic information. For
              Upkie this is the full observation dictionary sent by the spine.
        """
        self.time_stamp = 0
        self.initial_position = None
        # Reset previous action for smoothness penalty
        self.__previous_action = np.array([0.0])
        self.__previous_action_change = 0.0  # Reset action velocity tracking
        _, info = self.env.reset(seed=seed, options=options)
        spine_observation = info["spine_observation"]
        for joint in self.env.model.upper_leg_joints:
            position = spine_observation["servo"][joint.name]["position"]
            self.__leg_servo_action[joint.name]["position"] = position
        observation = self.__get_env_observation(spine_observation)
        # Store initial position for drift detection
        self.initial_position = observation[1]
        return observation, info

    def __get_leg_servo_action(self) -> Dict[str, Dict[str, float]]:
        r"""!
        Get servo actions for both hip and knee joints.

        \return Servo action dictionary.
        """
        for joint in self.env.model.upper_leg_joints:
            prev_position = self.__leg_servo_action[joint.name]["position"]
            new_position = low_pass_filter(
                prev_output=prev_position,
                new_input=0.0,  # go to neutral configuration
                cutoff_period=1.0,  # in roughly one second
                dt=self.env.dt,
            )
            self.__leg_servo_action[joint.name]["position"] = new_position
        return self.__leg_servo_action

    def __get_wheel_servo_action(
        self, left_wheel_velocity: float
    ) -> Dict[str, Dict[str, float]]:
        r"""!
        Get servo actions for wheel joints.

        \param[in] left_wheel_velocity Left-wheel velocity, in rad/s.
        \return Servo action dictionary.
        """
        right_wheel_velocity = -left_wheel_velocity
        servo_action = {
            "left_wheel": {
                "position": math.nan,
                "velocity": left_wheel_velocity,
            },
            "right_wheel": {
                "position": math.nan,
                "velocity": right_wheel_velocity,
            },
        }
        for joint in self.env.model.wheel_joints:
            servo_action[joint.name]["maximum_torque"] = joint.limit.effort
        return servo_action

    def __get_spine_action(self, action: np.ndarray) -> Dict[str, dict]:
        r"""!
        Convert environment action to a spine action dictionary.

        \param action Environment action.
        \return Spine action dictionary.
        """
        ground_velocity = clamp_and_warn(
            action[0],
            self.action_space.low[0],
            self.action_space.high[0],
            label="ground_velocity",
        )
        wheel_velocity = ground_velocity / ROBOT_CONFIG["wheel_radius"]
        left_wheel_sign = 1.0 if self.left_wheeled else -1.0
        left_wheel_velocity = left_wheel_sign * wheel_velocity
        leg_servo_action = self.__get_leg_servo_action()
        wheel_servo_action = self.__get_wheel_servo_action(left_wheel_velocity)
        return leg_servo_action | wheel_servo_action  # wheel comes second

    def __detect_fall(self, spine_observation: dict) -> bool:
        r"""!
        Detect a fall based on the base-to-world pitch angle.

        \param spine_observation Spine observation dictionary.
        \return True if and only if a fall is detected.

        Spine observations should have a "base_orientation" key. This requires
        the \ref upkie::cpp::observers::BaseOrientation observer in the spine's
        observer pipeline.
        """
        pitch = spine_observation["base_orientation"]["pitch"]
        if abs(pitch) > self.fall_pitch:
            logger.warning(
                "Fall detected (pitch=%.2f rad, fall_pitch=%.2f rad)",
                abs(pitch),
                self.fall_pitch,
            )
            return True
        return False

    def __compute_reward(
        self, observation: np.ndarray, action: np.ndarray
    ) -> float:
        r"""!
        Compute reward based on observation and action.

        The reward encourages stability by penalizing deviations from the
        ideal balanced state (θ=0, ẋ=0, ẏ=0), large control actions, and
        rapid action changes (jitter). Uses a shifted reward scale to provide
        better learning signal.

        \param observation Current observation vector [θ, p, ẋ, ẏ].
        \param action Current action [ground_velocity].
        \return Reward value, typically in [0, 1] range.
        """
        theta = observation[0]  # pitch angle (rad)
        theta_dot = observation[2]  # angular velocity (rad/s)
        p_dot = observation[3]  # linear velocity (m/s)
        action_mag = abs(action[0])  # action magnitude (m/s)
        
        # Compute action smoothness penalties
        # First derivative: action velocity (rate of change)
        action_change = action[0] - self.__previous_action[0]
        action_change_abs = abs(action_change)
        
        # Second derivative: action acceleration (change in rate of change)
        action_accel = abs(action_change - self.__previous_action_change)
        
        # Update tracking variables
        self.__previous_action_change = action_change
        self.__previous_action = action.copy()

        # Reward weights (tuned for stability and smoothness)
        # Increased smoothness penalties significantly to reduce jitter
        w_theta = 2.0  # penalty for pitch deviation (increased for better balance)
        w_theta_dot = 0.5  # penalty for angular velocity
        w_p_dot = 0.5  # penalty for linear velocity
        w_action = 1.0  # penalty for large actions (significantly increased to reduce jitter)
        w_smoothness = 2.0  # penalty for action velocity (first derivative, increased)
        w_accel = 1.5  # penalty for action acceleration (second derivative, NEW)

        # Compute reward: start from a positive baseline and subtract penalties
        # This provides a clearer learning signal - balanced state gives high reward
        # The baseline of 1.0 means perfect balance (all terms = 0) gives reward = 1.0
        reward = 1.0 - (
            w_theta * theta**2
            + w_theta_dot * theta_dot**2
            + w_p_dot * p_dot**2
            + w_action * action_mag**2
            + w_smoothness * action_change_abs**2
            + w_accel * action_accel**2
        )

        # Note: No clamping needed here - VecNormalize in train_pendulum.py will
        # normalize and clip rewards appropriately. The reward is naturally bounded
        # by physics limits, and extreme values are rare in practice.

        return reward

    def __check_termination_conditions(
        self, observation: np.ndarray, spine_observation: dict
    ) -> bool:
        r"""!
        Check additional termination conditions beyond fall detection.

        \param observation Current observation vector [θ, p, ẋ, ẏ].
        \param spine_observation Full spine observation dictionary.
        \return True if termination condition is met.
        """
        # Check position drift
        if self.initial_position is not None:
            position = observation[1]
            position_drift = abs(position - self.initial_position)
            if position_drift > self.max_position_drift:
                logger.warning(
                    "Termination: excessive position drift (%.2f m > %.2f m)",
                    position_drift,
                    self.max_position_drift,
                )
                return True

        # Check angular velocity
        theta_dot = abs(observation[2])
        if theta_dot > self.max_angular_velocity:
            logger.warning(
                "Termination: excessive angular velocity (%.2f rad/s > %.2f rad/s)",
                theta_dot,
                self.max_angular_velocity,
            )
            return True

        # Check linear velocity
        p_dot = abs(observation[3])
        if p_dot > self.max_linear_velocity:
            logger.warning(
                "Termination: excessive linear velocity (%.2f m/s > %.2f m/s)",
                p_dot,
                self.max_linear_velocity,
            )
            return True

        return False

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        r"""!
        Run one timestep of the environment's dynamics.

        When the end of the episode is reached, you are responsible for calling
        `reset()` to reset the environment's state.

        \param action Action from the agent.
        \return
            - `observation`: Observation of the environment, i.e. an element
              of its `observation_space`.
            - `reward`: Reward returned after taking the action.
            - `terminated`: Whether the agent reached a terminal state,
              which may be a good or a bad thing. When true, the user needs to
              call `reset()`.
            - `truncated`: Whether the episode is reaching max number of
              steps. This boolean can signal a premature end of the episode,
              i.e. before a terminal state is reached. When true, the user
              needs to call `reset()`.
            - `info`: Dictionary with additional information, reporting in
              particular the full observation dictionary coming from the spine.
        """
        spine_action = self.__get_spine_action(action)
        _, _, terminated, truncated, info = self.env.step(spine_action)
        spine_observation = info["spine_observation"]
        observation = self.__get_env_observation(spine_observation)

        # Compute reward based on observation and action
        reward = self.__compute_reward(observation, action)

        # Check termination conditions
        if self.__detect_fall(spine_observation):
            terminated = True
        elif self.__check_termination_conditions(observation, spine_observation):
            terminated = True

        # Check time limit
        self.time_stamp += 1
        if self.env.max_time_steps is not None:
            if self.time_stamp >= self.env.max_time_steps:
                truncated = True

        return observation, reward, terminated, truncated, info
