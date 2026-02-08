# run_gridworld_new.py
import numpy as np
from envs.abstract_sensor_gridworld import AbstractSensorGridWorld
from experiment_runner import ExperimentRunner

# --- 1. 環境固有の設定 ---

class GridWorldDiscretizer:
    def __init__(self):
        bins_num = 8
        self.bins = [
            np.linspace(0, 1, bins_num),   # s1: 壁距離
            np.linspace(0, 2, bins_num),   # s2: トラップ匂い
            np.linspace(-1, 1, bins_num),  # s3: ゴール方向
            np.linspace(0, 1, bins_num)    # s4: 危険度
        ]
        # shape属性はQテーブル作成時に必要
        self.shape = tuple([bins_num] * 4)

    def __call__(self, obs):
        idxs = []
        for i in range(4):
            idx = np.digitize(obs[i], self.bins[i]) - 1
            idx = min(max(idx, 0), len(self.bins[i]) - 1)
            idxs.append(idx)
        return tuple(idxs)

def calculate_success(episode_infos):
    """
    エピソード最後のinfoを見て、ゴール座標(3,3)にいれば成功(1.0)、それ以外は失敗(0.0)
    ExperimentRunnerで平滑化されることで「成功率」のグラフになる
    """
    if not episode_infos:
        return 0.0
    last_info = episode_infos[-1]
    # 環境側のゴール座標定義に合わせて (3, 3) を指定
    if last_info.get('position') == (3, 3):
        return 1.0
    return 0.0

# --- 2. プロンプト定義 ---

GRID_PROMPT = """
You are writing ONLY Python code for a reward function used in Gymnasium.
Write a function with EXACT signature:
def compute_reward(obs, terminated, truncated, info):
    \"\"\"Return a float reward.\"\"\"

Constraints:
- obs: numpy.ndarray of shape (4,) -> [s1 (wall dist), s2 (trap smell), s3 (goal dir), s4 (static risk)]
- terminated=True if goal or trap; truncated=True if time-limit
- Use ONLY standard Python and numpy; NO imports; NO I/O.

Design goals:
- +10 for reaching goal (terminated with info['position']==(3,3))
- -10 for trap termination; small time penalty each step.
- Encourage moving closer to goal via s3 (goal direction -1..1).
- Penalize high s2 (trap smell) and very small s1 (too close to wall).

Return ONLY the function code inside ```python ... ```.
"""

# --- 3. メイン実行 ---

if __name__ == "__main__":
    runner = ExperimentRunner(
        env_factory=lambda: AbstractSensorGridWorld(),
        discretizer=GridWorldDiscretizer(),
        metric_fn=calculate_success,
        experiment_name="GridWorld Navigation",
        cache_file="cache_gridworld.json"
    )

    # (A) ベースライン
    runner.add_manual_reward("Default", None)
    runner.add_manual_reward("Naive Distance", """
def compute_reward(obs, terminated, truncated, info):
    s1, s2, s3, s4 = obs
    if terminated and info.get('position') == (3, 3):
        return 10.0
    if terminated:
        return -10.0
    return float(s3 - 0.1)
""")

    # (B) LLMモデル
    target_models = [
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
    ]
    runner.generate_llm_rewards(GRID_PROMPT, target_models)

    # (C) 実行 (GridWorldは学習に時間がかかるのでエピソード数多め推奨)
    runner.run_experiments(episodes=3000)

    # (D) 結果描画 (成功率の推移)
    runner.plot_results("result_gridworld_success.png")