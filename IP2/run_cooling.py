# run_cooling_new.py
import numpy as np
from envs.server_cooling import ServerCoolingEnv
from experiment_runner import ExperimentRunner

# --- 1. タスク固有の設定（離散化と評価関数） ---
class CoolingDiscretizer:
    def __init__(self):
        self.bins_temp = np.linspace(20, 100, 10)
        self.bins_load = np.linspace(0, 100, 5)
        self.shape = (len(self.bins_temp), len(self.bins_load))

    def __call__(self, obs):
        temp, load = obs
        t_idx = np.digitize(temp, self.bins_temp) - 1
        l_idx = np.digitize(load, self.bins_load) - 1
        return (
            min(max(t_idx, 0), len(self.bins_temp)-1),
            min(max(l_idx, 0), len(self.bins_load)-1)
        )

def calculate_temp_error(infos):
    """55度からの誤差平均（低いほうが良い）"""
    temps = [i.get('temp', 55.0) for i in infos]
    return np.mean(np.abs(np.array(temps) - 55.0))

# --- 2. プロンプト定義 ---
COOLING_PROMPT = """
You are an expert in Reinforcement Learning. Write a Python reward function for a server cooling task.
Function signature MUST be:
def compute_reward(obs, terminated, truncated, info):
    return float_reward

Context:
- Goal: Keep server temperature close to 55.0 degrees.
- Constraints: Minimize energy cost (fan speed).
- info['temp']: Current temperature (float, 20-100).
- info['action']: Fan speed (int, 0-4). Higher is more energy.
- terminated=True if temp < 20 or temp > 95 (Failure).

Requirements:
- Use info['temp'] and info['action'] to calculate reward.
- Give penalty for energy cost (action ** 2 is common).
- Give penalty for temp deviation from 55.0.
- Return ONLY the function code inside ```python ... ```.
"""

# --- 3. メイン実行 ---
if __name__ == "__main__":
    # Runnerの初期化
    runner = ExperimentRunner(
        env_factory=lambda: ServerCoolingEnv(),
        discretizer=CoolingDiscretizer(),
        metric_fn=calculate_temp_error,
        experiment_name="Server Cooling Task",
        cache_file="cache_cooling.json"  # 結果をここに保存
    )

    # (A) ベースライン（手動定義）の追加
    runner.add_manual_reward("Default", None) # None = 環境のデフォルト報酬
    runner.add_manual_reward("Baseline(Simple)", """
def compute_reward(obs, terminated, truncated, info):
    temp = info.get('temp', 55.0)
    if 50 <= temp <= 60: return 1.0
    return -1.0
""")

    # (B) LLMモデルの指定（OpenRouterのモデルID）
    # 比較したいモデルをここに並べるだけでOK
    target_models = [
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
        # "deepseek/deepseek-r1-distill-llama-70b" # 必要なら追加
    ]

    # プロンプトを投げてコード生成（キャッシュがあればそれを使う）
    runner.generate_llm_rewards(COOLING_PROMPT, target_models)

    # (C) 実験実行
    runner.run_experiments(episodes=1000)

    # (D) 結果描画
    runner.plot_results("result_cooling_comparison.png")