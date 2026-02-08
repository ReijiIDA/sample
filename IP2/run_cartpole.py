# run_cartpole_new.py
import gymnasium as gym
import numpy as np
from envs.cartpole_tracking import SinusoidTrackingWrapper
from experiment_runner import ExperimentRunner

# --- 1. 環境固有の設定 ---

class CartPoleDiscretizer:
    def __init__(self):
        self.bins_num = 6
        self.bins = [
            np.linspace(-2.4, 2.4, self.bins_num),  # x error
            np.linspace(-3.0, 3.0, self.bins_num),  # x dot
            np.linspace(-0.2, 0.2, self.bins_num),  # theta
            np.linspace(-2.0, 2.0, self.bins_num)   # theta dot
        ]
        self.shape = tuple([self.bins_num] * 4)

    def __call__(self, obs):
        idxs = []
        for i in range(4):
            idx = np.digitize(obs[i], self.bins[i]) - 1
            idx = min(max(idx, 0), self.bins_num - 1)
            idxs.append(idx)
        return tuple(idxs)

def calculate_tracking_error(episode_infos):
    """
    エピソード中の x_error の絶対値の平均を計算（低いほど良い）
    """
    if not episode_infos:
        return 0.0
    errors = [abs(i.get('x_error', 0)) for i in episode_infos]
    return np.mean(errors)

def make_env():
    """Wrapperを適用した環境を返すファクトリ関数"""
    base = gym.make("CartPole-v1")
    return SinusoidTrackingWrapper(base)

# --- 2. プロンプト定義 ---

CARTPOLE_PROMPT = """
You are an expert in Reinforcement Learning. Write a Python reward function for a modified CartPole task.
The cart must track a moving sinusoidal target.

Function signature:
def compute_reward(obs, terminated, truncated, info):
    return float_reward

Context:
- info['x_error']: (cart_x - target_x). We want this close to 0.
- obs[2]: theta (pole angle). We want to keep the pole upright (close to 0).
- terminated=True if theta is too large or cart moves too far.

Requirements:
- Reward should be HIGHER when abs(info['x_error']) is small.
- Reward should be HIGHER when abs(obs[2]) is small (stability).
- Return ONLY the function code inside ```python ... ```.
"""

# --- 3. メイン実行 ---

if __name__ == "__main__":
    runner = ExperimentRunner(
        env_factory=make_env,
        discretizer=CartPoleDiscretizer(),
        metric_fn=calculate_tracking_error,
        experiment_name="CartPole Tracking",
        cache_file="cache_cartpole.json"
    )

    # (A) ベースライン
    runner.add_manual_reward("Default", None)
    runner.add_manual_reward("Simple Stability", """
def compute_reward(obs, terminated, truncated, info):
    # obs[2] is theta
    theta = obs[2]
    reward = 1.0 - abs(theta)
    if terminated: reward = -10.0
    return reward
""")

    # (B) LLMモデル
    target_models = [
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
    ]
    runner.generate_llm_rewards(CARTPOLE_PROMPT, target_models)

    # (C) 実行
    runner.run_experiments(episodes=1000)

    # (D) 結果描画 (誤差の推移)
    runner.plot_results("result_cartpole_error.png")