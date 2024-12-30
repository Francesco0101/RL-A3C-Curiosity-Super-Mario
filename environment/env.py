import gym
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from utils.constants import *


PALETTE_ACTIONS = [['NOOP'],
 ['up'],
 ['down'],
 ['left'],
 ['left', 'A'],
 ['left', 'B'],
 ['left', 'A', 'B'],
 ['right'],
 ['right', 'A'],
 ['right', 'B'],
 ['right', 'A', 'B'],
 ['A'],
 ['B'],
 ['A', 'B']
 ]



class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info

class CustomReward(Wrapper):
    def __init__(self, env=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84))
        self.prev_time = 400
        self.prev_stat = 0
        self.prev_score = 0
        self.prev_dist = 40
        self.counter = 0
        self.milestones = [i for i in range(150,3150,150)]
        
    def step(self, action):
        
        state, _, done, trunc ,info = self.env.step(action)    
        #normalize the state
        state = state / 255.
        if REWARD_TYPE == "dense":
            # Implementing custom rewards
            reward = max(min((info['x_pos'] - self.prev_dist - 0.05), 2), -2)
            self.prev_dist = info['x_pos']
            reward += (self.prev_time - info['time'])* -0.1
            self.prev_time = info['time']

            reward += (int(info['status']!='small')  - self.prev_stat) * 5
            self.prev_stat = int(info['status']!='small')

            reward += (info['score'] - self.prev_score) * 0.025
            self.prev_score = info['score']

            if done:
                if info['flag_get'] :
                    reward += 500
                else:
                    reward -= 50
        elif REWARD_TYPE == "sparse":
            reward = 0
            if (self.counter < len(self.milestones)) and (info['x_pos'] > self.milestones[self.counter])  : 
                reward = 10 
                self.counter = self.counter + 1
            if done :
                if info['flag_get'] :
                    reward = 50
                else:
                    reward = -10
        else:
            reward = 0
        return state, reward/10, done, trunc , info

    def reset(self):
        self.prev_time = 400
        self.prev_stat = 0
        self.prev_score = 0
        self.prev_dist = 40
        state , info = self.env.reset()
        state = state / 255.
        return state, info


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        if observation is not None:    # for future meta implementation
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

            unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
            unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

            return (observation - unbiased_mean) / (unbiased_std + 1e-8)
        
        else:
            return observation



def create_train_env(world=WORLD, stage=STAGE, render = False):
    if render:
        env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage),render_mode = 'human', apply_api_compatibility=True)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage),render_mode = "rgb_array" , apply_api_compatibility=True)

    env = JoypadSpace(env, PALETTE_ACTIONS)
    env = ResizeObservation(env, shape=(84, 84))
    # Convert to grayscale
    env = GrayScaleObservation(env, keep_dim=False)
    # Apply the CustomReward wrapper
    env = CustomReward(env)
    # Normalize the observations
    env = NormalizedEnv(env)
    # SkipFrame wrapper to skip frames for efficiency
    env = SkipFrame(env, skip=4)
    # Stack frames (e.g., stack 4 frames)
    env = FrameStack(env, num_stack=4)
    return env, env.observation_space.shape[0], len(PALETTE_ACTIONS)
