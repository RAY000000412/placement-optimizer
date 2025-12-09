"""
配置最適化プログラム（グループ順列×グループ内配置 完全総当たり版）
optimize_placement_bruteforce_1209.py

機能:
- CSVファイルから遷移確率行列を読み込み
- グループの並び順（順列）も含めた完全総当たり探索
- 全解のコスト分布を可視化（局所解の分析）

作成日: 2024/12/09
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict
from itertools import permutations, product
import math


class BruteForceOptimizerWithGroupPermutation:
    """
    グループ順列×グループ内配置の完全総当たり最適化クラス
    
    グループの並び順とグループ内の配置を全て列挙して最適解を見つける
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
                ...
            }
            ※ 'positions' は省略可能（グループ順列探索時は自動計算）
        state_labels : List[str], optional
            状態のラベル
        """
        self.transition_matrix = transition_matrix
        self.n_states = transition_matrix.shape[0]
        self.groups_original = groups
        
        if state_labels is None:
            self.state_labels = [chr(65 + i) for i in range(self.n_states)]
        else:
            self.state_labels = state_labels
        
        # グループ名のリスト
        self.group_names = list(groups.keys())
        self.n_groups = len(self.group_names)
        
        # 各グループのサイズ
        self.group_sizes = {name: len(info['states']) for name, info in groups.items()}
        
        # 基本検証
        self._validate_groups_basic()
        
        # 組み合わせ数を計算
        self._calculate_total_combinations()
        
        # 全解の記録用
        self.all_solutions = []
        
        # グループ順列ごとの結果
        self.group_permutation_results = []
        
        # 現在のグループ設定（positions付き）
        self.groups = None
        self.state_to_group = {}
    
    def _validate_groups_basic(self):
        """グループ設定の基本検証（状態の重複・網羅性のみ）"""
        all_states = set()
        
        for group_name, group_info in self.groups_original.items():
            states = group_info['states']
            
            for state in states:
                if state in all_states:
                    raise ValueError(f"状態 {state} が複数のグループに属しています")
                all_states.add(state)
        
        if len(all_states) != self.n_states:
            missing = set(range(self.n_states)) - all_states
            raise ValueError(f"グループに属していない状態があります: {missing}")
        
        print("✅ グループ設定の検証完了")
    
    def _calculate_total_combinations(self):
        """総組み合わせ数を計算"""
        # グループ順列の数
        self.n_group_permutations = math.factorial(self.n_groups)
        
        # グループ内配置の組み合わせ数
        self.n_internal_combinations = 1
        for group_info in self.groups_original.values():
            n = len(group_info['states'])
            self.n_internal_combinations *= math.factorial(n)
        
        # 総組み合わせ数
        self.total_combinations = self.n_group_permutations * self.n_internal_combinations
        
        print(f"グループ順列: {self.n_group_permutations:,} 通り")
        print(f"グループ内配置: {self.n_internal_combinations:,} 通り")
        print(f"総組み合わせ数: {self.total_combinations:,} 通り")
    
    def _assign_positions(self, group_order: List[str]) -> Dict[str, Dict]:
        """グループ順序に基づいて位置を割り当てる"""
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
        """配置のコストを計算"""
        cost = 0.0
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i != j and self.transition_matrix[i, j] > 0:
                    pos_i = placement.index(i)
                    pos_j = placement.index(j)
                    distance = abs(pos_i - pos_j)
                    cost += distance * self.transition_matrix[i, j]
        return cost
    
    def brute_force(self, verbose: bool = True) -> Tuple[List[int], float, List[str]]:
        """
        グループ順列×グループ内配置の完全総当たり探索
        
        Returns:
        --------
        Tuple[List[int], float, List[str]] : (最適配置, 最小コスト, 最適グループ順序)
        """
        if verbose:
            print("\n" + "=" * 70)
            print("グループ順列×グループ内配置 完全総当たり探索を開始")
            print(f"グループ順列: {self.n_group_permutations:,} 通り")
            print(f"グループ内配置: {self.n_internal_combinations:,} 通り")
            print(f"総組み合わせ数: {self.total_combinations:,} 通り")
            print("=" * 70)
        
        best_placement = None
        best_cost = float('inf')
        best_group_order = None
        
        self.all_solutions = []
        self.group_permutation_results = []
        
        start_time = time.time()
        evaluated = 0
        
        # 全グループ順列を列挙
        all_group_orders = list(permutations(self.group_names))
        
        for perm_idx, group_order in enumerate(all_group_orders):
            group_order = list(group_order)
            
            if verbose:
                order_str = ' → '.join(group_order)
                print(f"\n[{perm_idx+1}/{self.n_group_permutations}] グループ順序: {order_str}")
            
            # このグループ順序での位置を計算
            groups_with_pos = self._assign_positions(group_order)
            
            # 各グループの全順列を事前計算
            group_perms = []
            group_positions = []
            
            for group_name in group_order:
                states = groups_with_pos[group_name]['states']
                pos_start, pos_end = groups_with_pos[group_name]['positions']
                group_perms.append(list(permutations(states)))
                group_positions.append((pos_start, pos_end))
            
            # このグループ順序での最良解
            best_placement_for_order = None
            best_cost_for_order = float('inf')
            
            # グループ内配置の全組み合わせを探索
            for combo in product(*group_perms):
                # 配置を構築
                placement = [None] * self.n_states
                for g_idx, perm in enumerate(combo):
                    pos_start, pos_end = group_positions[g_idx]
                    for i, state in enumerate(perm):
                        placement[pos_start + i] = state
                
                # コスト計算
                cost = self.calculate_cost(placement)
                evaluated += 1
                
                # 全解を記録
                self.all_solutions.append({
                    'group_order': group_order,
                    'placement': placement.copy(),
                    'cost': cost
                })
                
                # このグループ順序での最良解を更新
                if cost < best_cost_for_order:
                    best_cost_for_order = cost
                    best_placement_for_order = placement.copy()
                
                # 全体の最良解を更新
                if cost < best_cost:
                    best_cost = cost
                    best_placement = placement.copy()
                    best_group_order = group_order
                
                # 進捗表示
                if verbose and evaluated % 100000 == 0:
                    elapsed = time.time() - start_time
                    progress = evaluated / self.total_combinations * 100
                    print(f"    進捗: {progress:.1f}% ({evaluated:,}/{self.total_combinations:,}) "
                          f"現在の最良: {best_cost:.4f} 経過: {elapsed:.1f}秒")
            
            # このグループ順序の結果を記録
            self.group_permutation_results.append({
                'group_order': group_order,
                'best_placement': best_placement_for_order,
                'best_cost': best_cost_for_order
            })
            
            if verbose:
                print(f"    最良コスト: {best_cost_for_order:.4f}")
                if best_cost_for_order == best_cost:
                    print(f"    ★ 全体の最良解!")
        
        elapsed_time = time.time() - start_time
        
        # 最良のグループ順序で設定を更新
        self.groups = self._assign_positions(best_group_order)
        self._update_state_to_group()
        
        # グループ順列の結果をソート
        self.group_permutation_results.sort(key=lambda x: x['best_cost'])
        
        if verbose:
            print("\n" + "=" * 70)
            print("✅ 完全総当たり探索完了")
            print(f"評価した組み合わせ数: {evaluated:,}")
            print(f"実行時間: {elapsed_time:.2f}秒")
            print(f"最小コスト: {best_cost:.4f} （厳密な最適解）")
            print(f"最適グループ順序: {' → '.join(best_group_order)}")
            print("=" * 70)
            
            # グループ順列ごとの結果サマリー
            print("\n【グループ順列ごとの最良コスト（コスト順）】")
            for i, result in enumerate(self.group_permutation_results):
                order_str = ' → '.join(result['group_order'])
                marker = "★" if result['best_cost'] == best_cost else " "
                print(f"  {marker} {i+1}. {order_str}: {result['best_cost']:.4f}")
        
        return best_placement, best_cost, best_group_order
    
    def analyze_solutions(self) -> Dict:
        """全解の統計分析"""
        if not self.all_solutions:
            raise ValueError("先に brute_force() を実行してください")
        
        costs = [sol['cost'] for sol in self.all_solutions]
        costs_array = np.array(costs)
        
        min_cost = np.min(costs_array)
        max_cost = np.max(costs_array)
        mean_cost = np.mean(costs_array)
        std_cost = np.std(costs_array)
        
        # 最適解の数
        optimal_count = np.sum(np.abs(costs_array - min_cost) < 0.0001)
        
        # コストの分位点
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(costs_array, percentiles)
        
        analysis = {
            'total_solutions': len(costs),
            'min_cost': min_cost,
            'max_cost': max_cost,
            'mean_cost': mean_cost,
            'std_cost': std_cost,
            'optimal_count': optimal_count,
            'percentiles': dict(zip(percentiles, percentile_values)),
            'costs_array': costs_array
        }
        
        print("\n" + "=" * 70)
        print("【全解の統計分析】")
        print("=" * 70)
        print(f"総解数:     {analysis['total_solutions']:,}")
        print(f"最小コスト: {analysis['min_cost']:.4f}")
        print(f"最大コスト: {analysis['max_cost']:.4f}")
        print(f"平均コスト: {analysis['mean_cost']:.4f}")
        print(f"標準偏差:   {analysis['std_cost']:.4f}")
        print(f"最適解の数: {analysis['optimal_count']:,} "
              f"({analysis['optimal_count']/analysis['total_solutions']*100:.4f}%)")
        
        print("\n【コストの分位点】")
        for p, v in analysis['percentiles'].items():
            print(f"  {p}%点: {v:.4f}")
        
        return analysis
    
    def visualize_all_solutions(self, save_path: str = None) -> plt.Figure:
        """全解のコスト分布を可視化"""
        if not self.all_solutions:
            raise ValueError("先に brute_force() を実行してください")
        
        costs = np.array([sol['cost'] for sol in self.all_solutions])
        min_cost = np.min(costs)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. ソート済み全解のコスト
        ax1 = axes[0, 0]
        sorted_costs = np.sort(costs)
        ax1.scatter(range(len(sorted_costs)), sorted_costs, 
                   c=sorted_costs, cmap='RdYlGn_r', s=1, alpha=0.5)
        ax1.axhline(y=min_cost, color='red', linestyle='--', linewidth=2, label=f'Optimal: {min_cost:.4f}')
        ax1.set_xlabel('Solution Rank')
        ax1.set_ylabel('Cost')
        ax1.set_title(f'All {len(costs):,} Solutions (Sorted)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. コストのヒストグラム
        ax2 = axes[0, 1]
        ax2.hist(costs, bins=100, edgecolor='black', alpha=0.7)
        ax2.axvline(x=min_cost, color='red', linestyle='--', linewidth=2, label=f'Optimal: {min_cost:.4f}')
        ax2.set_xlabel('Cost')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Cost Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 累積分布
        ax3 = axes[1, 0]
        sorted_costs = np.sort(costs)
        cumulative = np.arange(1, len(sorted_costs) + 1) / len(sorted_costs)
        ax3.plot(sorted_costs, cumulative, 'b-', linewidth=1)
        ax3.axvline(x=min_cost, color='red', linestyle='--', linewidth=2, label=f'Optimal: {min_cost:.4f}')
        ax3.set_xlabel('Cost')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution Function')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. グループ順列ごとのコスト比較
        ax4 = axes[1, 1]
        if self.group_permutation_results:
            perm_costs = [r['best_cost'] for r in self.group_permutation_results]
            perm_labels = [' → '.join(r['group_order']) for r in self.group_permutation_results]
            
            y_pos = range(len(perm_costs))
            bars = ax4.barh(y_pos, perm_costs, color='steelblue')
            
            # 最小コストをハイライト
            min_perm_cost = min(perm_costs)
            for i, (bar, cost) in enumerate(zip(bars, perm_costs)):
                if abs(cost - min_perm_cost) < 0.0001:
                    bar.set_color('gold')
            
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(perm_labels, fontsize=8)
            ax4.set_xlabel('Best Cost')
            ax4.set_title('Best Cost by Group Permutation')
            ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n図を保存しました: {save_path}")
        
        return fig
    
    def visualize_local_optima(self, save_path: str = None) -> plt.Figure:
        """局所解の分析と可視化"""
        if not self.all_solutions:
            raise ValueError("先に brute_force() を実行してください")
        
        costs = np.array([sol['cost'] for sol in self.all_solutions])
        min_cost = np.min(costs)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. コスト値の度数分布
        ax1 = axes[0]
        rounded_costs = np.round(costs, 2)
        unique_costs, counts = np.unique(rounded_costs, return_counts=True)
        
        ax1.bar(unique_costs, counts, width=0.02, edgecolor='black', alpha=0.7)
        ax1.axvline(x=min_cost, color='red', linestyle='--', linewidth=2, label=f'Optimal: {min_cost:.4f}')
        ax1.set_xlabel('Cost (rounded to 0.01)')
        ax1.set_ylabel('Number of Solutions')
        ax1.set_title('Local Optima Clusters')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 上位の局所解クラスタ
        ax2 = axes[1]
        sorted_unique_costs = np.sort(unique_costs)[:20]
        counts_top = [counts[np.where(unique_costs == c)[0][0]] for c in sorted_unique_costs]
        
        bars = ax2.barh(range(len(sorted_unique_costs)), counts_top, 
                       color=['gold' if abs(c - min_cost) < 0.01 else 'steelblue' for c in sorted_unique_costs])
        ax2.set_yticks(range(len(sorted_unique_costs)))
        ax2.set_yticklabels([f'{c:.4f}' for c in sorted_unique_costs])
        ax2.set_xlabel('Number of Solutions')
        ax2.set_ylabel('Cost')
        ax2.set_title('Top 20 Local Optima (by cost)')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        for i, (bar, count) in enumerate(zip(bars, counts_top)):
            ax2.text(bar.get_width() + max(counts_top) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{count:,}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n図を保存しました: {save_path}")
        
        return fig
    
    def print_solution(self, placement: List[int], cost: float):
        """解の詳細表示"""
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
    
    def print_top_solutions(self, n: int = 10):
        """上位n個の解を表示"""
        if not self.all_solutions:
            raise ValueError("先に brute_force() を実行してください")
        
        sorted_solutions = sorted(self.all_solutions, key=lambda x: x['cost'])
        
        print(f"\n" + "=" * 70)
        print(f"【上位{n}個の解】")
        print("=" * 70)
        
        for rank, sol in enumerate(sorted_solutions[:n], 1):
            placement = sol['placement']
            cost = sol['cost']
            group_order = sol['group_order']
            labels = [self.state_labels[placement[i]] for i in range(self.n_states)]
            order_str = '→'.join(group_order)
            print(f"#{rank}: コスト={cost:.4f}  グループ順序: {order_str}")
            print(f"      配置: {' '.join(labels)}")
    
    def analyze_cost_contributions(self, placement: List[int]):
        """コストへの寄与度を分析"""
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
    print("グループ順列×グループ内配置 完全総当たり探索")
    print("=" * 70)
    
    # ============================================================
    # 設定（ここを変更してください）
    # ============================================================
    
    # 入力ファイル
    input_file = 'transition_matrix.csv'
    
    # 状態ラベル（A〜O の15個）
    state_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    
    # ラベル名からインデックスを取得するヘルパー関数
    def get_indices(labels):
        return [state_labels.index(label) for label in labels]
    
    # グループ定義（positionsは省略 - 自動計算される）
    groups = {
        'Group1': {
            'states': get_indices(['A', 'B', 'C', 'D', 'E', 'F']),  # 6個
        },
        'Group2': {
            'states': get_indices(['G', 'H', 'I', 'J']),            # 4個
        },
        'Group3': {
            'states': get_indices(['K', 'L', 'M']),                 # 3個
        },
        'Group4': {
            'states': get_indices(['N', 'O']),                      # 2個
        }
    }
    
    # ============================================================
    # 遷移確率行列の読み込み
    # ============================================================
    
    try:
        transition_matrix = load_transition_matrix(input_file)
    except FileNotFoundError:
        print(f"{input_file} が見つかりません。サンプルデータを生成します。")
        transition_matrix = create_sample_transition_matrix_15x15()
    
    # ============================================================
    # グループ設定の表示
    # ============================================================
    
    print("\n【グループ設定】")
    for group_name, group_info in groups.items():
        states_str = ', '.join([state_labels[s] for s in group_info['states']])
        n_states = len(group_info['states'])
        print(f"  {group_name} ({n_states}個): [{states_str}]")
    
    # 組み合わせ数の計算
    n_groups = len(groups)
    n_group_perms = math.factorial(n_groups)
    n_internal = 1
    for group_info in groups.values():
        n_internal *= math.factorial(len(group_info['states']))
    total = n_group_perms * n_internal
    
    print(f"\n予想組み合わせ数:")
    print(f"  グループ順列: {n_group_perms:,} 通り")
    print(f"  グループ内配置: {n_internal:,} 通り")
    print(f"  合計: {total:,} 通り")
    
    # ============================================================
    # 最適化の実行
    # ============================================================
    
    optimizer = BruteForceOptimizerWithGroupPermutation(
        transition_matrix,
        groups,
        state_labels
    )
    
    # 完全総当たり探索
    start_time = time.time()
    best_placement, best_cost, best_group_order = optimizer.brute_force(verbose=True)
    elapsed_time = time.time() - start_time
    
    print(f"\n総実行時間: {elapsed_time:.2f}秒")
    
    # 結果表示
    optimizer.print_solution(best_placement, best_cost)
    
    # コスト寄与度分析
    optimizer.analyze_cost_contributions(best_placement)
    
    # 上位10個の解を表示
    optimizer.print_top_solutions(n=10)
    
    # ============================================================
    # 全解の分析と可視化
    # ============================================================
    
    # 統計分析
    analysis = optimizer.analyze_solutions()
    
    # 全解のコスト分布を可視化
    fig1 = optimizer.visualize_all_solutions(save_path='all_solutions_distribution_1209.png')
    
    # 局所解の分析
    fig2 = optimizer.visualize_local_optima(save_path='local_optima_analysis_1209.png')
    
    print("\n" + "=" * 70)
    print("完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
