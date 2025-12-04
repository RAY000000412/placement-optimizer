"""
配置最適化プログラム（グループ制約付き）
optimize_placement_1204.py

機能:
- 15×15の遷移確率行列に対応
- グループ分けによる配置位置の制限
- 目的関数: Σ(距離 × 遷移確率) を最小化（全体で計算）

作成日: 2024/12/04
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Optional
import random


class GroupConstrainedPlacementOptimizer:
    """
    グループ制約付き配置最適化クラス
    
    各状態をグループに分け、グループごとに配置可能な位置範囲を制限しながら
    全体の目的関数（Σ距離×遷移確率）を最小化する
    """
    
    def __init__(self, 
                 transition_matrix: np.ndarray, 
                 groups: Dict[str, Dict],
                 state_labels: List[str] = None):
        """
        Parameters:
        -----------
        transition_matrix : np.ndarray
            遷移確率行列 (n×n)
        groups : Dict[str, Dict]
            グループ定義。形式:
            {
                'group1': {
                    'states': [0, 1, 2, 3, 4],  # 状態インデックスのリスト
                    'positions': (0, 4)          # 配置可能な位置範囲 (start, end) ※0始まり
                },
                'group2': {
                    'states': [5, 6, 7, 8, 9],
                    'positions': (5, 9)
                },
                ...
            }
        state_labels : List[str], optional
            状態のラベル（指定しない場合はA, B, C, ...を自動生成）
        """
        self.transition_matrix = transition_matrix
        self.n_states = transition_matrix.shape[0]
        self.groups = groups
        
        # 状態ラベルの設定
        if state_labels is None:
            self.state_labels = [chr(65 + i) for i in range(self.n_states)]
        else:
            self.state_labels = state_labels
        
        # グループ設定の検証
        self._validate_groups()
        
        # 状態→グループのマッピングを作成
        self.state_to_group = {}
        for group_name, group_info in self.groups.items():
            for state in group_info['states']:
                self.state_to_group[state] = group_name
        
        # 統計情報の記録用
        self.history = {
            'costs': [],
            'temperatures': [],
            'acceptance_rates': []
        }
    
    def _validate_groups(self):
        """グループ設定の妥当性を検証"""
        all_states = set()
        all_positions = set()
        
        for group_name, group_info in self.groups.items():
            states = group_info['states']
            pos_start, pos_end = group_info['positions']
            
            # 状態数と位置範囲の一致を確認
            n_states_in_group = len(states)
            n_positions = pos_end - pos_start + 1
            if n_states_in_group != n_positions:
                raise ValueError(
                    f"グループ '{group_name}': 状態数({n_states_in_group})と"
                    f"位置数({n_positions})が一致しません"
                )
            
            # 状態の重複チェック
            for state in states:
                if state in all_states:
                    raise ValueError(f"状態 {state} が複数のグループに属しています")
                all_states.add(state)
            
            # 位置の重複チェック
            for pos in range(pos_start, pos_end + 1):
                if pos in all_positions:
                    raise ValueError(f"位置 {pos} が複数のグループに割り当てられています")
                all_positions.add(pos)
        
        # 全状態がグループに属しているか確認
        if len(all_states) != self.n_states:
            missing = set(range(self.n_states)) - all_states
            raise ValueError(f"グループに属していない状態があります: {missing}")
        
        print("✅ グループ設定の検証完了")
    
    def calculate_cost(self, placement: List[int]) -> float:
        """
        配置のコストを計算（全体で計算、グループ分けなし）
        
        Parameters:
        -----------
        placement : List[int]
            配置リスト。placement[i] = 位置iに配置された状態のインデックス
        
        Returns:
        --------
        float : 総コスト = Σ(距離 × 遷移確率)
        """
        cost = 0.0
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i != j and self.transition_matrix[i, j] > 0:
                    # 状態iと状態jの位置を取得
                    pos_i = placement.index(i)
                    pos_j = placement.index(j)
                    distance = abs(pos_i - pos_j)
                    cost += distance * self.transition_matrix[i, j]
        return cost
    
    def generate_initial_placement_greedy(self) -> List[int]:
        """
        貪欲法による初期配置生成（グループ制約付き）
        各グループ内で、遷移確率が高いペアを近くに配置
        """
        placement = [None] * self.n_states
        
        for group_name, group_info in self.groups.items():
            states = group_info['states']
            pos_start, pos_end = group_info['positions']
            positions = list(range(pos_start, pos_end + 1))
            
            # グループ内の遷移確率に基づいてソート
            # 他の状態との遷移確率の合計が高い順に中央から配置
            state_importance = []
            for state in states:
                total_prob = 0
                for other in range(self.n_states):
                    if state != other:
                        total_prob += self.transition_matrix[state, other]
                        total_prob += self.transition_matrix[other, state]
                state_importance.append((state, total_prob))
            
            # 重要度順にソート
            state_importance.sort(key=lambda x: x[1], reverse=True)
            sorted_states = [s[0] for s in state_importance]
            
            # 中央から配置（重要な状態を中央に）
            center = len(positions) // 2
            for i, state in enumerate(sorted_states):
                if i % 2 == 0:
                    pos_idx = center + i // 2
                else:
                    pos_idx = center - (i + 1) // 2
                
                if 0 <= pos_idx < len(positions):
                    placement[positions[pos_idx]] = state
        
        return placement
    
    def generate_initial_placement_random(self) -> List[int]:
        """
        ランダムな初期配置生成（グループ制約付き）
        """
        placement = [None] * self.n_states
        
        for group_name, group_info in self.groups.items():
            states = list(group_info['states'])
            pos_start, pos_end = group_info['positions']
            
            random.shuffle(states)
            for i, pos in enumerate(range(pos_start, pos_end + 1)):
                placement[pos] = states[i]
        
        return placement
    
    def get_neighbor(self, placement: List[int]) -> List[int]:
        """
        近傍解の生成（同一グループ内での交換のみ）
        
        Parameters:
        -----------
        placement : List[int]
            現在の配置
        
        Returns:
        --------
        List[int] : 近傍配置
        """
        new_placement = placement.copy()
        
        # ランダムにグループを選択
        group_name = random.choice(list(self.groups.keys()))
        group_info = self.groups[group_name]
        pos_start, pos_end = group_info['positions']
        
        # グループ内の位置からランダムに2つ選択して交換
        positions = list(range(pos_start, pos_end + 1))
        if len(positions) >= 2:
            pos1, pos2 = random.sample(positions, 2)
            new_placement[pos1], new_placement[pos2] = new_placement[pos2], new_placement[pos1]
        
        return new_placement
    
    def simulated_annealing(self,
                           initial_temp: float = 100.0,
                           cooling_rate: float = 0.995,
                           min_temp: float = 0.1,
                           max_iterations: int = 20000,
                           use_greedy_init: bool = True) -> Tuple[List[int], float]:
        """
        焼きなまし法による最適化
        
        Parameters:
        -----------
        initial_temp : float
            初期温度
        cooling_rate : float
            冷却率
        min_temp : float
            最小温度
        max_iterations : int
            最大反復回数
        use_greedy_init : bool
            True: 貪欲法で初期解生成, False: ランダム初期解
        
        Returns:
        --------
        Tuple[List[int], float] : (最良配置, 最良コスト)
        """
        # 初期解の生成
        if use_greedy_init:
            current = self.generate_initial_placement_greedy()
            print("初期解: 貪欲法で生成")
        else:
            current = self.generate_initial_placement_random()
            print("初期解: ランダム生成")
        
        current_cost = self.calculate_cost(current)
        
        best = current.copy()
        best_cost = current_cost
        
        temperature = initial_temp
        
        # 履歴の初期化
        self.history = {
            'costs': [current_cost],
            'temperatures': [temperature],
            'acceptance_rates': []
        }
        
        print(f"初期コスト: {current_cost:.4f}")
        
        accepted = 0
        total = 0
        
        iteration = 0
        while temperature > min_temp and iteration < max_iterations:
            # 近傍解の生成
            neighbor = self.get_neighbor(current)
            neighbor_cost = self.calculate_cost(neighbor)
            
            # 受理判定
            delta = neighbor_cost - current_cost
            total += 1
            
            if delta < 0:
                # 改善した場合は常に受理
                current = neighbor
                current_cost = neighbor_cost
                accepted += 1
            else:
                # 悪化した場合は確率的に受理
                probability = np.exp(-delta / temperature)
                if random.random() < probability:
                    current = neighbor
                    current_cost = neighbor_cost
                    accepted += 1
            
            # 最良解の更新
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost
            
            # 冷却
            temperature *= cooling_rate
            
            # 履歴の記録（100回ごと）
            if iteration % 100 == 0:
                self.history['costs'].append(current_cost)
                self.history['temperatures'].append(temperature)
                if total > 0:
                    self.history['acceptance_rates'].append(accepted / total)
                    accepted = 0
                    total = 0
            
            iteration += 1
        
        print(f"最終コスト: {best_cost:.4f}")
        print(f"反復回数: {iteration}")
        
        return best, best_cost
    
    def print_solution(self, placement: List[int], cost: float):
        """解の詳細表示"""
        print("\n" + "=" * 70)
        print("【最適化結果】")
        print("=" * 70)
        
        # グループごとの配置表示
        print("\n【グループ別配置】")
        for group_name, group_info in self.groups.items():
            pos_start, pos_end = group_info['positions']
            print(f"\n{group_name} (位置 {pos_start+1}〜{pos_end+1}):")
            print("  位置: ", end="")
            for pos in range(pos_start, pos_end + 1):
                print(f"{pos+1:>4}", end=" ")
            print("\n  状態: ", end="")
            for pos in range(pos_start, pos_end + 1):
                print(f"{self.state_labels[placement[pos]]:>4}", end=" ")
            print()
        
        # 全体配置
        print("\n【全体配置】")
        print("位置: ", end="")
        for i in range(self.n_states):
            print(f"{i+1:>4}", end=" ")
        print("\n状態: ", end="")
        for i in range(self.n_states):
            print(f"{self.state_labels[placement[i]]:>4}", end=" ")
        print(f"\n\n総コスト: {cost:.4f}")
        
        # 主要な遷移コスト
        print("\n【主要な遷移コスト（上位10件）】")
        contributions = []
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i != j and self.transition_matrix[i, j] > 0:
                    pos_i = placement.index(i)
                    pos_j = placement.index(j)
                    distance = abs(pos_i - pos_j)
                    contrib = distance * self.transition_matrix[i, j]
                    contributions.append((
                        self.state_labels[i],
                        self.state_labels[j],
                        self.transition_matrix[i, j],
                        distance,
                        contrib
                    ))
        
        contributions.sort(key=lambda x: x[4], reverse=True)
        
        print(f"{'From':<6} {'To':<6} {'Prob':<10} {'Distance':<10} {'Cost':<10}")
        print("-" * 50)
        for from_state, to_state, prob, dist, contrib in contributions[:10]:
            print(f"{from_state:<6} {to_state:<6} {prob:<10.4f} {dist:<10} {contrib:<10.4f}")
    
    def visualize_results(self, placement: List[int]) -> plt.Figure:
        """結果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 配置の可視化（グループ別に色分け）
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.groups)))
        group_colors = {}
        for i, group_name in enumerate(self.groups.keys()):
            group_colors[group_name] = colors[i]
        
        for pos in range(self.n_states):
            state = placement[pos]
            group = self.state_to_group[state]
            color = group_colors[group]
            ax1.barh(0, 1, left=pos, color=color, edgecolor='black', linewidth=1)
            ax1.text(pos + 0.5, 0, self.state_labels[state], 
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        # グループ境界線
        for group_info in self.groups.values():
            pos_end = group_info['positions'][1]
            if pos_end < self.n_states - 1:
                ax1.axvline(x=pos_end + 1, color='red', linewidth=2, linestyle='--')
        
        ax1.set_xlim(0, self.n_states)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('Position')
        ax1.set_title('Optimized Placement (Grouped)')
        ax1.set_yticks([])
        
        # 凡例
        legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=group_colors[g], 
                          edgecolor='black', label=g) for g in self.groups.keys()]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 2. 配置後の遷移確率行列
        ax2 = axes[0, 1]
        reordered_matrix = np.zeros_like(self.transition_matrix)
        for i in range(self.n_states):
            for j in range(self.n_states):
                reordered_matrix[i, j] = self.transition_matrix[placement[i], placement[j]]
        
        im = ax2.imshow(reordered_matrix, cmap='Blues', aspect='auto')
        ax2.set_title('Transition Matrix (Reordered by Placement)')
        ax2.set_xlabel('To Position')
        ax2.set_ylabel('From Position')
        
        # ラベル
        labels = [self.state_labels[placement[i]] for i in range(self.n_states)]
        ax2.set_xticks(range(self.n_states))
        ax2.set_yticks(range(self.n_states))
        ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_yticklabels(labels, fontsize=8)
        plt.colorbar(im, ax=ax2)
        
        # 3. コストの推移
        ax3 = axes[1, 0]
        ax3.plot(self.history['costs'], 'b-', linewidth=1)
        ax3.set_xlabel('Iteration (×100)')
        ax3.set_ylabel('Cost')
        ax3.set_title('Cost History')
        ax3.grid(True, alpha=0.3)
        
        # 4. 温度と受理率の推移
        ax4 = axes[1, 1]
        ax4_temp = ax4
        ax4_rate = ax4.twinx()
        
        line1, = ax4_temp.plot(self.history['temperatures'], 'r-', label='Temperature')
        ax4_temp.set_xlabel('Iteration (×100)')
        ax4_temp.set_ylabel('Temperature', color='r')
        ax4_temp.tick_params(axis='y', labelcolor='r')
        
        if self.history['acceptance_rates']:
            line2, = ax4_rate.plot(self.history['acceptance_rates'], 'g-', label='Acceptance Rate')
            ax4_rate.set_ylabel('Acceptance Rate', color='g')
            ax4_rate.tick_params(axis='y', labelcolor='g')
            ax4_rate.set_ylim(0, 1)
        
        ax4.set_title('Temperature and Acceptance Rate')
        
        plt.tight_layout()
        return fig


def create_sample_transition_matrix_15x15() -> np.ndarray:
    """
    15×15のサンプル遷移確率行列を生成
    """
    np.random.seed(42)
    n = 15
    
    # スパースな遷移確率行列を生成
    matrix = np.zeros((n, n))
    
    # 各状態から2〜4個の遷移を設定
    for i in range(n):
        n_transitions = np.random.randint(2, 5)
        targets = np.random.choice([j for j in range(n) if j != i], 
                                   size=n_transitions, replace=False)
        probs = np.random.dirichlet(np.ones(n_transitions))
        for t, p in zip(targets, probs):
            matrix[i, t] = p
    
    return matrix


def main():
    """メイン実行関数"""
    print("=" * 70)
    print("グループ制約付き配置最適化プログラム")
    print("=" * 70)
    
    # 15×15の遷移確率行列を生成（または読み込み）
    try:
        transition_matrix = np.load('/home/claude/transition_matrix_15x15.npy')
        print("遷移確率行列を読み込みました")
    except FileNotFoundError:
        print("遷移確率行列ファイルが見つかりません。サンプルを生成します。")
        transition_matrix = create_sample_transition_matrix_15x15()
        np.save('/home/claude/transition_matrix_15x15.npy', transition_matrix)
    
    print(f"行列サイズ: {transition_matrix.shape}")
    
    # グループ定義
    # A〜Oの15状態を3グループに分ける例
    groups = {
        'Group1': {
            'states': [0, 1, 2, 3, 4],      # A, B, C, D, E（インデックス0〜4）
            'positions': (0, 4)              # 位置1〜5（0始まり: 0〜4）
        },
        'Group2': {
            'states': [5, 6, 7, 8, 9],      # F, G, H, I, J（インデックス5〜9）
            'positions': (5, 9)              # 位置6〜10（0始まり: 5〜9）
        },
        'Group3': {
            'states': [10, 11, 12, 13, 14], # K, L, M, N, O（インデックス10〜14）
            'positions': (10, 14)            # 位置11〜15（0始まり: 10〜14）
        }
    }
    
    print("\n【グループ設定】")
    state_labels = [chr(65 + i) for i in range(15)]  # A〜O
    for group_name, group_info in groups.items():
        states_str = ', '.join([state_labels[s] for s in group_info['states']])
        pos_start, pos_end = group_info['positions']
        print(f"  {group_name}: 状態[{states_str}] → 位置[{pos_start+1}〜{pos_end+1}]")
    
    # 最適化の実行
    optimizer = GroupConstrainedPlacementOptimizer(
        transition_matrix, 
        groups,
        state_labels
    )
    
    # パラメータ設定
    params = {
        'initial_temp': 100.0,
        'cooling_rate': 0.997,      # 15状態なので少し遅く冷却
        'min_temp': 0.1,
        'max_iterations': 25000     # 反復回数を増やす
    }
    
    print("\n【焼きなまし法のパラメータ】")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()
    
    # 最適化実行
    start_time = time.time()
    best_placement, best_cost = optimizer.simulated_annealing(**params)
    elapsed_time = time.time() - start_time
    
    print(f"\n実行時間: {elapsed_time:.2f}秒")
    
    # 結果の詳細表示
    optimizer.print_solution(best_placement, best_cost)
    
    # 結果の可視化
    fig = optimizer.visualize_results(best_placement)
    plt.savefig('/home/claude/optimization_results_grouped.png', dpi=150, bbox_inches='tight')
    print("\n結果を optimization_results_grouped.png に保存しました")
    
    plt.show()


def example_custom_groups():
    """
    カスタムグループ設定の例
    グループのメンバーを任意に指定するデモ
    """
    print("\n" + "=" * 70)
    print("【カスタムグループ設定の例】")
    print("=" * 70)
    
    # 15×15の遷移確率行列を生成
    transition_matrix = create_sample_transition_matrix_15x15()
    state_labels = [chr(65 + i) for i in range(15)]  # A〜O
    
    # カスタムグループ: 順序を変更した例
    # グループ1: F, A, O, K, J → 位置1〜5
    # グループ2: B, M, C, N, D → 位置6〜10
    # グループ3: E, G, H, I, L → 位置11〜15
    custom_groups = {
        'CustomGroup1': {
            'states': [5, 0, 14, 10, 9],   # F, A, O, K, J
            'positions': (0, 4)
        },
        'CustomGroup2': {
            'states': [1, 12, 2, 13, 3],   # B, M, C, N, D
            'positions': (5, 9)
        },
        'CustomGroup3': {
            'states': [4, 6, 7, 8, 11],    # E, G, H, I, L
            'positions': (10, 14)
        }
    }
    
    print("\n【カスタムグループ設定】")
    for group_name, group_info in custom_groups.items():
        states_str = ', '.join([state_labels[s] for s in group_info['states']])
        pos_start, pos_end = group_info['positions']
        print(f"  {group_name}: 状態[{states_str}] → 位置[{pos_start+1}〜{pos_end+1}]")
    
    # 最適化実行
    optimizer = GroupConstrainedPlacementOptimizer(
        transition_matrix,
        custom_groups,
        state_labels
    )
    
    best_placement, best_cost = optimizer.simulated_annealing(
        initial_temp=100.0,
        cooling_rate=0.997,
        min_temp=0.1,
        max_iterations=25000
    )
    
    optimizer.print_solution(best_placement, best_cost)
    
    return optimizer, best_placement, best_cost


if __name__ == "__main__":
    main()
    
    # カスタムグループの例も実行する場合は以下のコメントを外す
    # example_custom_groups()
