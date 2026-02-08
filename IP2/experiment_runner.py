# experiment_runner.py
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from envs.wrappers import LLMRewardWrapper
from training.q_learning import train_q_learning
from LLMapi_openrouter import call_llm
import re

class ExperimentRunner:
    def __init__(self, env_factory, discretizer, metric_fn, experiment_name="Experiment", cache_file="reward_cache.json"):
        """
        Args:
            env_factory: () -> gym.Env を返す関数
            discretizer: 観測値を離散化するクラス/関数
            metric_fn: infoリストを受け取り評価値を返す関数
            experiment_name: グラフ描画用のタイトル
            cache_file: 生成されたコードを保存するパス
        """
        self.env_factory = env_factory
        self.discretizer = discretizer
        self.metric_fn = metric_fn
        self.name = experiment_name
        self.cache_file = cache_file
        
        # { "ModelName": "def compute_reward... code" }
        self.reward_codes = {}
        self.results = {}
        
        # キャッシュがあれば読み込む
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.reward_codes = json.load(f)
                print(f"[Init] Loaded {len(self.reward_codes)} reward codes from {self.cache_file}")
            except Exception as e:
                print(f"[Init] Failed to load cache: {e}")

    def save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.reward_codes, f, indent=2, ensure_ascii=False)
            print(f"[Save] Cache saved to {self.cache_file}")
        except Exception as e:
            print(f"[Save] Failed to save cache: {e}")

    def add_manual_reward(self, name, code):
        """手動で報酬関数を追加"""
        self.reward_codes[name] = code

    def generate_llm_rewards(self, prompt, models, force_regenerate=False):
        """
        指定されたモデルリストを使ってLLMにコードを書かせる。
        既にキャッシュにある場合はスキップする（force_regenerate=Trueで強制上書き）。
        """
        for model in models:
            # 短い名前を作成 (例: openai/gpt-4o-mini -> gpt-4o-mini)
            short_name = model.split('/')[-1]
            key = f"LLM_{short_name}"

            if key in self.reward_codes and not force_regenerate:
                print(f"[Skip] {key} already exists in cache.")
                continue

            print(f"[Gen] Requesting to {model} ...")
            try:
                # API呼び出し
                raw_text = call_llm(prompt, model=model, temperature=0.5)
                code = self._strip_code(raw_text)
                
                # 簡易バリデーション
                if "def compute_reward" in code:
                    self.reward_codes[key] = code
                    print(f"  -> Success. Code length: {len(code)}")
                else:
                    print(f"  -> Failed. 'def compute_reward' not found.")
            except Exception as e:
                print(f"  -> API Error: {e}")
            
            # APIレート制限への配慮
            time.sleep(1)
        
        # 生成が終わったら保存
        self.save_cache()

    def _strip_code(self, text):
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        return m.group(1).strip() if m else text.strip()

    def run_experiments(self, episodes=1000):
        print(f"\n{'='*20} Starting Experiments: {self.name} {'='*20}")
        
        for name, code in self.reward_codes.items():
            print(f"--> Testing: {name}")
            
            # 環境作成
            base_env = self.env_factory()
            
            # ラッパー適用（コードがNoneならデフォルト環境のまま）
            if code:
                env = LLMRewardWrapper(base_env, code)
                # LLMコードが壊れていてコンパイルできなかった場合のチェック
                if env.reward_fn is None:
                    print(f"  [Warning] Reward function compilation failed for {name}. Skipping.")
                    env.close()
                    continue
            else:
                env = base_env

            # 学習実行 (汎用Q学習関数を使用)
            try:
                history = train_q_learning(
                    env, 
                    self.discretizer, 
                    episodes=episodes, 
                    metric_fn=self.metric_fn,
                    verbose=False
                )
            except Exception as e:
                print(f"  [Error] Training failed for {name}: {e}")
                env.close()
                continue
            
            env.close()
            
            # 結果の平滑化
            window = max(5, int(episodes * 0.05)) # エピソード数の5%で移動平均
            if len(history) >= window:
                smoothed = np.convolve(history, np.ones(window)/window, mode='valid')
            else:
                smoothed = history
            
            self.results[name] = smoothed
            # 最終スコアを表示
            print(f"    Final Score (Last {window} avg): {np.mean(history[-window:]):.4f}")

    def plot_results(self, filename="experiment_result.png"):
        if not self.results:
            print("No results to plot.")
            return

        plt.figure(figsize=(12, 7))
        
        # 色を見やすくするためのカラーサイクル
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        for i, (name, data) in enumerate(self.results.items()):
            color = colors[i % len(colors)]
            # メインの線
            plt.plot(data, label=name, color=color, linewidth=2, alpha=0.9)
        
        plt.xlabel('Episode')
        plt.ylabel('Metric (Smoothed)')
        plt.title(f'{self.name}: Performance Comparison')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150)
            print(f"\n[Plot] Saved to {filename}")
        plt.show()