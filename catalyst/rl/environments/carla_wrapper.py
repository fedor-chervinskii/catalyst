#!/usr/bin/env python

import time
import numpy as np
from srunner.challenge.random_target_runner import RandomTargetRunner


class CarlaWrapper:
    def __init__(
        self,
        host="0.0.0.0",
        port=2000,
        num_vehicles=20,
        frame_skip=1,
        visualize=False,
        reward_scale=1,
        step_delay=0.0
    ):
        self.runner = RandomTargetRunner(host, port, num_vehicles)

        self.visualize = visualize
        self.frame_skip = frame_skip
        self.reward_scale = reward_scale
        self.step_delay = step_delay

        self.observation_shape = (5,)
        self.action_shape = (3,)

        self.time_step = 0
        self.total_reward = 0

    def reset(self):
        self.time_step = 0
        self.total_reward = 0
        return self.runner.reset()

    def calculate_reward(self, observation_dict, done):
        if done:
            return 1.
        else:
            return -0.01 + np.abs(observation_dict["speed"]) * 1e-4

    def state_to_vec(self, obs_dict):
        return [(obs_dict["goal"].x - obs_dict["location"].x) / 100.,
                (obs_dict["goal"].y - obs_dict["location"].y) / 100.,
                obs_dict["speed"] / 100.,
                np.sin(np.deg2rad(obs_dict["yaw"])),
                np.cos(np.deg2rad(obs_dict["yaw"]))]

    def step(self, action):
        time.sleep(self.step_delay)
        reward = 0
        for i in range(self.frame_skip):
            observation_dict, done = self.runner.step(action)
            r = self.calculate_reward(observation_dict, done)
            observation = self.state_to_vec(observation_dict)
            if self.visualize:
                "TBD"
            reward += r
            if done:
                break
        self.total_reward += reward
        self.time_step += 1
        info = observation_dict
        info["reward_origin"] = reward
        reward *= self.reward_scale
        return observation, reward, done, info
