# import gym
# from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, NormalizeObservation
# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros
# import numpy as np

# def process_frame(frame):
#     if frame is not None:
#         frame = frame/ 255.
#         return frame
#     else:
#         return np.zeros((1, 84, 84))

# class SkipFrame(gym.Wrapper):
#     def __init__(self, env, skip):
#         """Return only every `skip`-th frame"""
#         super().__init__(env)
#         self._skip = skip

#     def step(self, action):
#         """Repeat action, and sum reward"""
#         total_reward = 0.0
#         for i in range(self._skip):
#             # Accumulate reward and repeat the same action
#             obs, reward, done, trunk, info = self.env.step(action)
#             total_reward += reward
#             if done:
#                 break
#         return obs, total_reward, done, trunk, info

# class CustomReward(gym.Wrapper):
#     def __init__(self, env=None):
#         super(CustomReward, self).__init__(env)
#         self.curr_score = 0
    

#     def step(self, action):
        
#         state, reward, done, trunk  ,info = self.env.step(action)
#         state = process_frame(state)
#         reward += (info["score"] - self.curr_score) / 40.
#         self.curr_score = info["score"]
#         if done:
#             if info["flag_get"]:
#                 reward += 50
#             else:
#                 reward -= 50
#         return state, reward / 10., done, trunk,  info

#     def reset(self):
#         self.curr_score = 0
#         state , info = self.env.reset()
#         return process_frame(state) , info

    

# def create_train_env(world, stage, action_type, output_path = None):
#     env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0", render_mode='rgb_array', apply_api_compatibility=True)
#     env = JoypadSpace(env, action_type)
#     env = GrayScaleObservation(env)
#     env = ResizeObservation(env, shape=84)
#     env = CustomReward(env)
#     env = SkipFrame(env, skip=4)
#     env = FrameStack(env, num_stack=4)
    
#     return env , env.observation_space.shape[0], env.action_space.n



"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp



def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0

    def step(self, action):
        
        state, reward, done, _ ,info = self.env.step(action)
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        state , _ = self.env.reset()
        return process_frame(state)


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)[None, :, :, :]
        return states.astype(np.float32)


def create_train_env(world, stage, action_type):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage), apply_api_compatibility=True)

    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions)
    env = CustomReward(env)
    env = CustomSkipFrame(env)
    return env, env.observation_space.shape[0], len(actions)
