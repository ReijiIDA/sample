import numpy as np

def train_q_learning(env, discretizer, episodes=2000, verbose=True, metric_fn=None):
    """
    汎用Q学習関数
    
    Args:
        env: Gymnasium環境
        discretizer: 観測(obs)を受け取り、タプルのインデックスを返す関数
        metric_fn: (オプション) 報酬以外に記録したい指標を計算する関数 func(info_history) -> float
    """
    # Qテーブルのサイズを自動特定するために一度ダミー実行してshapeを取得
    obs_dummy, _ = env.reset()
    state_dummy = discretizer(obs_dummy)
    
    # 状態のビン数(tupleの要素ごとの最大値+1)を知る必要があるが、
    # 簡易的にdiscretizerが持つ属性 bins_shape を参照するか、引数で渡す設計にする
    # ここでは discretizer に .shape 属性があると仮定する
    if not hasattr(discretizer, 'shape'):
        raise ValueError("discretizer function must have a 'shape' attribute (tuple of bin sizes).")
        
    q_table_shape = discretizer.shape + (env.action_space.n,)
    q_table = np.zeros(q_table_shape)

    lr = 0.1
    gamma = 0.95
    epsilon = 1.0
    eps_decay = 0.995
    min_eps = 0.01

    history = [] # 報酬またはメトリクスの履歴

    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretizer(obs)
        total_reward = 0
        done = False
        
        # メトリクス計算用のログ
        episode_infos = []

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = discretizer(next_obs)
            
            # Q値更新
            old_value = q_table[state + (action,)]
            next_max = np.max(q_table[next_state])
            new_value = (1 - lr) * old_value + lr * (reward + gamma * next_max)
            q_table[state + (action,)] = new_value

            state = next_state
            total_reward += reward
            episode_infos.append(info)

        if epsilon > min_eps:
            epsilon *= eps_decay
            
        # 記録: metric_fnがあればそれを使う（例：温度誤差）、なければ合計報酬
        if metric_fn:
            val = metric_fn(episode_infos)
            history.append(val)
        else:
            history.append(total_reward)

        if verbose and (episode + 1) % 200 == 0:
            avg_val = np.mean(history[-200:])
            print(f"Episode {episode+1}/{episodes}, Avg Metric: {avg_val:.2f}, Epsilon: {epsilon:.3f}")

    return history