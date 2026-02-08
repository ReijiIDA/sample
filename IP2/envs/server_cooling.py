import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ServerCoolingEnv(gym.Env):
    """
    データセンターのサーバー冷却環境
    State: [温度(20-100), 負荷(0-100)]
    Action: 0(停止) ~ 4(強風)
    """
    def __init__(self):
        super(ServerCoolingEnv, self).__init__()
        
        self.observation_space = spaces.Box(
            low=np.array([20.0, 0.0]), 
            high=np.array([100.0, 100.0]), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        
        self.target_temp_low = 50.0
        self.target_temp_high = 60.0
        self.ambient_temp = 25.0
        self.state = None
        self.steps = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([
            np.random.uniform(40, 70),
            np.random.uniform(20, 80)
        ], dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        temp, load = self.state
        
        heat_gain = 0.05 * load + np.random.normal(0, 0.5)
        cooling_power = 1.5 * action 
        natural_decay = 0.05 * (temp - self.ambient_temp)
        
        next_temp = np.clip(temp + heat_gain - cooling_power - natural_decay, 20, 100)
        
        load_change = np.random.randint(-10, 11)
        next_load = np.clip(load + load_change, 0, 100)
        
        self.state = np.array([next_temp, next_load], dtype=np.float32)
        self.steps += 1
        
        terminated = False
        truncated = (self.steps >= self.max_steps)
        
        if next_temp >= 95.0 or next_temp <= 20.0:
            terminated = True
            
        # デフォルト報酬
        if self.target_temp_low <= next_temp <= self.target_temp_high:
            reward = 1.0 - (0.1 * action)
        else:
            reward = -0.1 * abs(next_temp - 55.0)
            
        if terminated:
            reward = -100.0

        # wrappers.py が action を引数に取らないため、info に action を入れて渡す
        info = {
            'temp': next_temp, 
            'load': next_load, 
            'action': action
        }
        
        return self.state, reward, terminated, truncated, info