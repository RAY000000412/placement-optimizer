"""
配置最適化プログラム（グループ制約付き・グループ順列対応版）
optimize_placement_1204.py

機能:
- 15×15の遷移確率行列に対応
- グループ分けによる配置位置の制限
- グループの並び順（順列）も最適化
- 目的関数: Σ(距離 × 遷移確率) を最小化（全体で計算）

作成日: 2024/12/04
更新日: 2024/12/09 - グループ順列対応
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Optional
from itertools import permutations
import random


class GroupConstrainedPlacementOptimizer:
    """
    グループ制約付き配置最適化クラス（グループ順列対応版）
    
    各状態をグループに分け、グループの並び順とグループ内の配置を
    同時に最適化して、全体の目的関数（Σ距離×遷移確率）を最小化する
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
                },
                'group2': {
                    'states': [5, 6, 7, 8, 9],
                },
                ...
            }
            ※ 'positions' は省略可能（グループ順列探索時は自動計算）
        state_labels : List[str], optional
            状態のラベル（指定しない場合はA, B, C, ...を自動生成）
        """
        self.transition_matrix = transition_matrix
        self.n_states = transition_matrix.shape[0]
        self.groups_original = groups
        
        # 状態ラベルの設定
        if state_labels is None:
            self.state_labels = [chr(65 + i) for i in range(self.n_states)]
        else:
            self.state_labels = state_labels
        
        # グループ名のリスト（順序保持）
        self.group_names = list(groups.keys())
        
        # 各グループのサイズを記録
        self.group_sizes = {name: len(info['states']) for name, info in groups.items()}
        
        # 基本検証
        self._validate_groups_basic()
        
        # 現在のグループ設定（positions付き）- 初期化時は元の順序
        self.groups = self._assign_positions(self.group_names)
        
        # 状態→グループのマッピングを作成
        self._update_state_to_group()
        
        # 統計情報の記録用
        self.history = {
            'costs': [],
            'temperatures': [],
            'acceptance_rates': []
        }
        
        # グループ順列探索の結果
        self.group_permutation_results = []
    
    def _validate_groups_basic(self):
        """グループ設定の基本検証（状態の重複・網羅性のみ）"""
        all_states = set()
        
        for group_name, group_info in self.groups_original.items():
            states = group_info['states']
            
            # 状態の重複チェック
            for state in states:
                if state in all_states:
                    raise ValueError(f"状態 {state} が複数のグループに属しています")
                all_states.add(state)
        
        # 全状態がグループに属しているか確認
        if len(all_states) != self.n_states:
            missing = set(range(self.n_states)) - all_states
            raise ValueError(f"グループに属していない状態があります: {missing}")
        
        print("✅ グループ設定の検証完了")
    
    def _assign_positions(self, group_order: List[str]) -> Dict[str, Dict]:
        """
        グループ順序に基づいて位置を割り当てる
        
        Parameters:
        -----------
        group_order : List[str]
            グループ名の順序リスト
        
        Returns:
        --------
        Dict : positions付きのグループ定義
        """
        groups_with_positions = {}
        current_pos = 0
        
        for group_name in group_order:
            states = self.groups_original[group_name]['states']
            n_states = len(states)
            groups_with_positions[group_name] = {
                'states': states,
                'positions': (current_pos, current_pos + n_states - 1)
            }
            current_pos += n_states
        
        return groups_with_positions
    
    def _update_state_to_group(self):
        """状態→グループのマッピングを更新"""
        self.state_to_group = {}
        for group_name, group_info in self.groups.items():
            for state in group_info['states']:
                self.state_to_group[state] = group_name
    
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
            
            state_importance = []
            for state in states:
                total_prob = 0
                for other in range(self.n_states):
                    if state != other:
                        total_prob += self.transition_matrix[state, other]
                        total_prob += self.transition_matrix[other, state]
                state_importance.append((state, total_prob))
            
            state_importance.sort(key=lambda x: x[1], reverse=True)
            sorted_states = [s[0] for s in state_importance]
            
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
        """
        new_placement = placement.copy()
        
        group_name = random.choice(list(self.groups.keys()))
        group_info = self.groups[group_name]
        pos_start, pos_end = group_info['positions']
        
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
                           use_greedy_init: bool = True,
                           verbose: bool = True) -> Tuple[List[int], float]:
        """
        焼きなまし法による最適化（現在のグループ設定で実行）
        """
        # 初期解の生成
        if use_greedy_init:
            current = self.generate_initial_placement_greedy()
            if verbose:
                print("初期解: 貪欲法で生成")
        else:
            current = self.generate_initial_placement_random()
            if verbose:
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
        
        if verbose:
            print(f"初期コスト: {current_cost:.4f}")
        
        accepted = 0
        total = 0
        
        iteration = 0
        while temperature > min_temp and iteration < max_iterations:
            neighbor = self.get_neighbor(current)
            neighbor_cost = self.calculate_cost(neighbor)
            
            delta = neighbor_cost - current_cost
            total += 1
            
            if delta < 0:
                current = neighbor
                current_cost = neighbor_cost
                accepted += 1
            else:
                probability = np.exp(-delta / temperature)
                if random.random() < probability:
                    current = neighbor
                    current_cost = neighbor_cost
                    accepted += 1
            
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost
            
            temperature *= cooling_rate
            
            if iteration % 100 == 0:
                self.history['costs'].append(current_cost)
                self.history['temperatures'].append(temperature)
                if total > 0:
                    self.history['acceptance_rates'].append(accepted / total)
                    accepted = 0
                    total = 0
            
            iteration += 1
        
        if verbose:
            print(f"最終コスト: {best_cost:.4f}")
            print(f"反復回数: {iteration}")
        
        return best, best_cost
    
    def optimize_with_group_permutations(self,
                                          initial_temp: float = 100.0,
                                          cooling_rate: float = 0.995,
                                          min_temp: float = 0.1,
                                          max_iterations: int = 20000,
                                          use_greedy_init: bool = True,
                                          verbose: bool = True) -> Tuple[List[int], float, List[str]]:
        """
        グループの並び順（順列）を総当たりで探索し、各順列で焼きなまし法を実行
        
        Parameters:
        -----------
        initial_temp, cooling_rate, min_temp, max_iterations, use_greedy_init:
            焼きなまし法のパラメータ
        verbose : bool
            進捗を表示するか
        
        Returns:
        --------
        Tuple[List[int], float, List[str]]:
            (最良配置, 最良コスト, 最良グループ順序)
        """
        n_groups = len(self.group_names)
        all_permutations = list(permutations(self.group_names))
        n_permutations = len(all_permutations)
        
        print("=" * 70)
        print(f"グループ順列の総当たり探索")
        print(f"グループ数: {n_groups}, 順列数: {n_permutations}")
        print("=" * 70)
        
        best_placement = None
        best_cost = float('inf')
        best_group_order = None
        
        self.group_permutation_results = []
        
        for i, group_order in enumerate(all_permutations):
            group_order = list(group_order)
            
            if verbose:
                order_str = ' → '.join(group_order)
                print(f"\n[{i+1}/{n_permutations}] グループ順序: {order_str}")
            
            # グループ設定を更新
            self.groups = self._assign_positions(group_order)
            self._update_state_to_group()
            
            # 焼きなまし法を実行
            placement, cost = self.simulated_annealing(
                initial_temp=initial_temp,
                cooling_rate=cooling_rate,
                min_temp=min_temp,
                max_iterations=max_iterations,
                use_greedy_init=use_greedy_init,
                verbose=False
            )
            
            if verbose:
                print(f"  コスト: {cost:.4f}")
            
            # 結果を記録
            self.group_permutation_results.append({
                'group_order': group_order,
                'placement': placement,
                'cost': cost
            })
            
            # 最良解を更新
            if cost < best_cost:
                best_cost = cost
                best_placement = placement
                best_group_order = group_order
                if verbose:
                    print(f"  ★ 新しい最良解!")
        
        # 最良のグループ順序で設定を更新
        self.groups = self._assign_positions(best_group_order)
        self._update_state_to_group()
        
        # 結果をコストでソート
        self.group_permutation_results.sort(key=lambda x: x['cost'])
        
        print("\n" + "=" * 70)
        print("【グループ順列探索結果サマリー】")
        print("=" * 70)
        print(f"\n最良グループ順序: {' → '.join(best_group_order)}")
        print(f"最良コスト: {best_cost:.4f}")
        
        print("\n【全順列の結果（コスト順）】")
        for i, result in enumerate(self.group_permutation_results):
            order_str = ' → '.join(result['group_order'])
            marker = "★" if result['cost'] == best_cost else " "
            print(f"  {marker} {i+1}. {order_str}: コスト = {result['cost']:.4f}")
        
        return best_placement, best_cost, best_group_order
    
    def print_solution(self, placement: List[int], cost: float):
        """解を表示"""
        print("\n" + "=" * 70)
        print("【最適化結果】")
        print("=" * 70)
        
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
        
        print("\n【全体配置】")
        print("位置: ", end="")
        for i in range(self.n_states):
            print(f"{i+1:>4}", end=" ")
        print("\n状態: ", end="")
        for i in range(self.n_states):
            print(f"{self.state_labels[placement[i]]:>4}", end=" ")
        
        print(f"\n\n総コスト: {cost:.4f}")
    
    def analyze_cost_contributions(self, placement: List[int]):
        """コストへの寄与度を分析（上位10遷移を表示）"""
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
        
        print(f"\n【コスト寄与度（上位10遷移）】")
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
        ax3.set_title('Cost History (Last Permutation)')
        ax3.grid(True, alpha=0.3)
        
        # 4. グループ順列ごとのコスト比較
        ax4 = axes[1, 1]
        if self.group_permutation_results:
            costs = [r['cost'] for r in self.group_permutation_results]
            labels = [' → '.join(r['group_order']) for r in self.group_permutation_results]
            
            # コストでソートされているのでそのまま表示
            y_pos = range(len(costs))
            bars = ax4.barh(y_pos, costs, color='steelblue')
            
            # 最小コストをハイライト
            min_cost = min(costs)
            for i, (bar, cost) in enumerate(zip(bars, costs)):
                if cost == min_cost:
                    bar.set_color('gold')
            
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(labels, fontsize=8)
            ax4.set_xlabel('Cost')
            ax4.set_title('Cost by Group Permutation')
            ax4.grid(True, alpha=0.3, axis='x')
        else:
            ax4.text(0.5, 0.5, 'No permutation data', ha='center', va='center')
            ax4.set_title('Cost by Group Permutation')
        
        plt.tight_layout()
        return fig


def load_transition_matrix(filepath: str) -> np.ndarray:
    """CSVファイルから遷移確率行列を読み込む"""
    try:
        matrix = np.loadtxt(filepath, delimiter=',')
        print(f"✅ 遷移確率行列を読み込みました: {filepath}")
        print(f"   サイズ: {matrix.shape[0]} × {matrix.shape[1]}")
        return matrix
    except FileNotFoundError:
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")


def create_sample_transition_matrix_15x15() -> np.ndarray:
    """15×15のサンプル遷移確率行列を生成"""
    np.random.seed(42)
    n = 15
    
    matrix = np.zeros((n, n))
    
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
    print("グループ制約付き配置最適化プログラム（グループ順列対応版）")
    print("=" * 70)
    
    # 遷移確率行列の読み込み
    input_file = 'transition_matrix.csv'
    try:
        transition_matrix = load_transition_matrix(input_file)
    except FileNotFoundError:
        print(f"{input_file} が見つかりません。サンプルデータを生成します。")
        transition_matrix = create_sample_transition_matrix_15x15()
    
    print(f"行列サイズ: {transition_matrix.shape}")
    
    # 状態ラベル
    state_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    
    # ラベル名からインデックスを取得するヘルパー関数
    def get_indices(labels):
        return [state_labels.index(label) for label in labels]
    
    # グループ定義（positionsは省略 - 自動計算される）
    groups = {
        'Group1': {
            'states': get_indices(['A', 'B', 'C', 'D', 'E', 'F']),
        },
        'Group2': {
            'states': get_indices(['G', 'H', 'I', 'J']),
        },
        'Group3': {
            'states': get_indices(['K', 'L', 'M']),
        },
        'Group4': {
            'states': get_indices(['N', 'O']),
        }
    }
    
    print("\n【グループ設定】")
    for group_name, group_info in groups.items():
        states_str = ', '.join([state_labels[s] for s in group_info['states']])
        print(f"  {group_name}: 状態[{states_str}] ({len(group_info['states'])}個)")
    
    # オプティマイザの作成
    optimizer = GroupConstrainedPlacementOptimizer(
        transition_matrix, 
        groups,
        state_labels
    )
    
    # グループ順列を含めた最適化を実行
    start_time = time.time()
    best_placement, best_cost, best_group_order = optimizer.optimize_with_group_permutations(
        initial_temp=100.0,
        cooling_rate=0.997,
        min_temp=0.1,
        max_iterations=25000
    )
    elapsed_time = time.time() - start_time
    
    print(f"\n総実行時間: {elapsed_time:.2f}秒")
    
    # 結果の詳細表示
    optimizer.print_solution(best_placement, best_cost)
    optimizer.analyze_cost_contributions(best_placement)
    
    # 結果の可視化
    fig = optimizer.visualize_results(best_placement)
    plt.savefig('optimization_results_grouped.png', dpi=150, bbox_inches='tight')
    print("\n結果を optimization_results_grouped.png に保存しました")


if __name__ == "__main__":
    main()
