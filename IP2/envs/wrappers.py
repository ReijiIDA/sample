
# envs/wrappers.py
import gymnasium as gym

class LLMRewardWrapper(gym.Wrapper):
    """LLMが生成した報酬関数でデフォルト報酬を上書き"""
    def __init__(self, env, llm_code_string):
        super().__init__(env)
        local_scope = {}
        try:
            exec(llm_code_string, {}, local_scope)
            self.reward_fn = local_scope.get("compute_reward")
            if not callable(self.reward_fn):
                print("Warning: compute_reward function not found.")
                self.reward_fn = None
        except Exception as e:
            print(f"Code compilation failed: {e}")
            self.reward_fn = None

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        if self.reward_fn:
            try:
                new_reward = self.reward_fn(obs, terminated, truncated, info)
            except Exception as e:
                print(f"Reward function error: {e}")
                new_reward = original_reward
        else:
            new_reward = original_reward
        return obs, new_reward, terminated, truncated, info
