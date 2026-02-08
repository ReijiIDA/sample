
# envs/abstract_sensor_gridworld.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class AbstractSensorGridWorld(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, grid_size=5, max_steps=30):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, -1.0, 0.0]),
            high=np.array([1.0, 2.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self._define_layout()
        self.agent_pos = None
        self.step_count = 0

    def _define_layout(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        self.grid[2, 2] = 1  # 内部壁

        self.goal = (3, 3)
        self.grid[self.goal[0], self.goal[1]] = 3

        self.traps = [(1, 3), (2, 1), (3, 2)]
        for trap in self.traps:
            r, c = trap
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                if self.grid[r, c] == 0:
                    self.grid[r, c] = 2

        np.random.seed(42)
        self.danger_map = np.random.uniform(0, 0.3, (self.grid_size, self.grid_size))
        for trap in self.traps:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r, c = trap[0] + dr, trap[1] + dc
                    if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                        self.danger_map[r, c] += 0.2
        self.danger_map = np.clip(self.danger_map, 0, 1)

        # 到達可能性チェック（必要ならログ）
        self._check_reachability()

    def _check_reachability(self):
        from collections import deque
        valid_starts = self._get_valid_start_positions()
        if not valid_starts:
            print("⚠️ WARNING: No valid start positions!")
            return
        start = valid_starts[0]
        visited = set([start])
        q = deque([start])
        while q:
            r, c = q.popleft()
            if (r, c) == self.goal:
                # print("✅ Goal is REACHABLE from starting positions")
                return
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size \
                   and (nr, nc) not in visited and self.grid[nr, nc] != 1:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        print("❌ WARNING: Goal is NOT REACHABLE from starting positions!")

    def _get_valid_start_positions(self):
        valid = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == 0:
                    valid.append((r, c))
        return valid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        valid_positions = self._get_valid_start_positions()
        self.agent_pos = valid_positions[np.random.randint(len(valid_positions))]
        self.step_count = 0
        observation = self._get_observation()
        return observation, {}

    def _get_observation(self) -> np.ndarray:
        r, c = self.agent_pos
        # s1: 壁距離（最短距離/正規化）
        wall_distances = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            dist = 0
            nr, nc = r, c
            while 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                if self.grid[nr, nc] == 1:
                    break
                dist += 1
                nr += dr; nc += dc
            wall_distances.append(dist)
        s1 = min(wall_distances) / self.grid_size

        # s2: トラップ匂い
        s2 = 0.0
        alpha = 1.5
        for trap in self.traps:
            dist = abs(r - trap[0]) + abs(c - trap[1])
            s2 += np.exp(-alpha * dist)
        s2 += np.random.normal(0, 0.05)
        s2 = max(0, s2)

        # s3: ゴール方向（ノイズ低減）
        goal_vec = np.array([self.goal[0] - r, self.goal[1] - c])
        if np.linalg.norm(goal_vec) > 0:
            goal_direction = np.dot(goal_vec, [1, 1]) / (np.linalg.norm(goal_vec) * np.sqrt(2))
        else:
            goal_direction = 1.0
        s3 = np.clip(goal_direction + np.random.normal(0, 0.1), -1, 1)

        # s4: 静的危険度
        s4 = self.danger_map[r, c]

        return np.array([s1, s2, s3, s4], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        r, c = self.agent_pos
        old_dist_to_goal = abs(r - self.goal[0]) + abs(c - self.goal[1])

        dr, dc = [(-1,0),(1,0),(0,-1),(0,1)][action]
        new_r, new_c = r + dr, c + dc

        reward = -0.1
        terminated = False

        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
            reward = -0.5
            new_r, new_c = r, c
        else:
            cell_type = self.grid[new_r, new_c]
            if cell_type == 1:  # 壁
                reward = -0.5
                new_r, new_c = r, c
            elif cell_type == 2:  # トラップ
                reward = -10.0
                terminated = True
                self.agent_pos = (new_r, new_c)
            elif cell_type == 3:  # ゴール
                reward = 10.0
                terminated = True
                self.agent_pos = (new_r, new_c)
            else:  # 通常セル
                self.agent_pos = (new_r, new_c)
                new_dist_to_goal = abs(new_r - self.goal[0]) + abs(new_c - self.goal[1])
                if new_dist_to_goal < old_dist_to_goal:
                    reward += 0.2
                elif new_dist_to_goal > old_dist_to_goal:
                    reward -= 0.1

        truncated = self.step_count >= self.max_steps
        observation = self._get_observation()
        info = {'position': self.agent_pos, 'step': self.step_count}
        return observation, reward, terminated, truncated, info

    def render(self):
        grid_display = self.grid.copy().astype(float)
        grid_display[self.agent_pos] = 4

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(grid_display, cmap='tab10', vmin=0, vmax=4)

        ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
        ax.tick_params(which='minor', size=0)

        cbar = plt.colorbar(im, ax=ax, ticks=[0,1,2,3,4])
        cbar.ax.set_yticklabels(['Empty', 'Wall', 'Trap', 'Goal', 'Agent'])
        ax.set_title(f'Step: {self.step_count}')
        plt.tight_layout()
        plt.show()
