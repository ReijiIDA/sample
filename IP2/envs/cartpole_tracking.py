import gymnasium as gym
import numpy as np

class SinusoidTrackingWrapper(gym.Wrapper):
    """
    CartPoleのカート位置が正弦波(sin)を追いかけるようにするラッパー
    """
    def __init__(self, env):
        super().__init__(env)
        self.t = 0
        self.frequency = 0.1 

    def reset(self, seed=None, options=None):
        self.t = 0
        obs, info = self.env.reset(seed=seed, options=options)
        return self._modify_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += 1
        
        target_x = np.sin(self.t * self.frequency)
        
        # 観測データの1つ目(x)を「誤差」に書き換え
        modified_obs = self._modify_obs(obs, target_x)
        
        # LLM報酬計算用に生データとターゲットをinfoに入れる
        info["target_x"] = target_x
        info["real_x"] = obs[0]
        info["x_error"] = obs[0] - target_x
        info["theta"] = obs[2]
        info["action"] = action # ラッパー対策
        
        base_reward = 1.0 - abs(obs[0] - target_x)
        if terminated:
            base_reward = -10.0
            
        return modified_obs, base_reward, terminated, truncated, info

    def _modify_obs(self, obs, target_x=0.0):
        modified_obs = obs.copy()
        modified_obs[0] = obs[0] - target_x 
        return modified_obs