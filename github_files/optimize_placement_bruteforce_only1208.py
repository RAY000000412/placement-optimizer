"""
配置最適化プログラム（総当たり専用版）
optimize_placement_bruteforce_only.py

機能:
- CSVファイルから遷移確率行列を読み込み
- グループ制約付き総当たり探索
- 全解のコスト分布を可視化（局所解の分析）

作成日: 2024/12/04
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict
from itertools import permutations, product
import math


class BruteForceOptimizer:
    """
    グループ制約付き総当たり配置最適化クラス
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
            グループ定義
        state_labels : List[str], optional
            状態のラベル
        """
        self.transition_matrix = transition_matrix
        self.n_states = transition_matrix.shape[0]
        self.groups = groups
        
        if state_labels is None:
            self.state_labels = [chr(65 + i) for i in range(self.n_states)]
        else:
            self.state_labels = state_labels
        
        self._validate_groups()
        
        self.state_to_group = {}
        for group_name, group_info in self.groups.items():
            for state in group_info['states']:
                self.state_to_group[state] = group_name
        
        # 総組み合わせ数を計算
        self.total_combinations = 1
        for group_info in self.groups.values():
            n = len(group_info['states'])
            self.total_combinations *= math.factorial(n)
        
        print(f"総組み合わせ数: {self.total_combinations:,} 通り")
        
        # 全解の記録用
        self.all_solutions = []  # [(placement, cost), ...]
    
    def _validate_groups(self):
        """グループ設定の妥当性を検証"""
        all_states = set()
        all_positions = set()
        
        for group_name, group_info in self.groups.items():
            states = group_info['states']
            pos_start, pos_end = group_info['positions']
            
            n_states_in_group = len(states)
            n_positions = pos_end - pos_start + 1
            if n_states_in_group != n_positions:
                raise ValueError(
                    f"グループ '{group_name}': 状態数({n_states_in_group})と"
                    f"位置数({n_positions})が一致しません"
                )
            
            for state in states:
                if state in all_states:
                    raise ValueError(f"状態 {state} が複数のグループに属しています")
                all_states.add(state)
            
            for pos in range(pos_start, pos_end + 1):
                if pos in all_positions:
                    raise ValueError(f"位置 {pos} が複数のグループに割り当てられています")
                all_positions.add(pos)
        
        if len(all_states) != self.n_states:
            missing = set(range(self.n_states)) - all_states
            raise ValueError(f"グループに属していない状態があります: {missing}")
        
        print("✅ グループ設定の検証完了")
    
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
    
    def brute_force(self, verbose: bool = True) -> Tuple[List[int], float]:
        """
        総当たり探索（全解を記録）
        
        Returns:
        --------
        Tuple[List[int], float] : (最適配置, 最小コスト)
        """
        if verbose:
            print("\n" + "=" * 60)
            print("総当たり探索を開始")
            print(f"探索する組み合わせ数: {self.total_combinations:,} 通り")
            print("=" * 60)
        
        # 各グループの全順列を事前計算
        group_perms = []
        group_positions = []
        
        for group_name, group_info in self.groups.items():
            states = group_info['states']
            pos_start, pos_end = group_info['positions']
            group_perms.append(list(permutations(states)))
            group_positions.append((pos_start, pos_end))
        
        best_placement = None
        best_cost = float('inf')
        evaluated = 0
        
        # 全解を記録
        self.all_solutions = []
        
        start_time = time.time()
        
        # 全グループの順列の直積を生成
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
            self.all_solutions.append((placement.copy(), cost))
            
            if cost < best_cost:
                best_cost = cost
                best_placement = placement.copy()
            
            # 進捗表示
            if verbose and evaluated % 50000 == 0:
                elapsed = time.time() - start_time
                progress = evaluated / self.total_combinations * 100
                print(f"  進捗: {progress:.1f}% ({evaluated:,}/{self.total_combinations:,}) "
                      f"現在の最良: {best_cost:.4f} 経過: {elapsed:.1f}秒")
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n✅ 総当たり探索完了")
            print(f"評価した組み合わせ数: {evaluated:,}")
            print(f"実行時間: {elapsed_time:.2f}秒")
            print(f"最小コスト: {best_cost:.4f} （厳密な最適解）")
        
        return best_placement, best_cost
    
    def analyze_solutions(self) -> Dict:
        """
        全解の統計分析
        
        Returns:
        --------
        Dict : 分析結果
        """
        if not self.all_solutions:
            raise ValueError("先に brute_force() を実行してください")
        
        costs = [sol[1] for sol in self.all_solutions]
        costs_array = np.array(costs)
        
        min_cost = np.min(costs_array)
        max_cost = np.max(costs_array)
        mean_cost = np.mean(costs_array)
        std_cost = np.std(costs_array)
        
        # 最適解の数（同じ最小コストを持つ解）
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
        
        print("\n" + "=" * 60)
        print("【全解の統計分析】")
        print("=" * 60)
        print(f"総解数:     {analysis['total_solutions']:,}")
        print(f"最小コスト: {analysis['min_cost']:.4f}")
        print(f"最大コスト: {analysis['max_cost']:.4f}")
        print(f"平均コスト: {analysis['mean_cost']:.4f}")
        print(f"標準偏差:   {analysis['std_cost']:.4f}")
        print(f"最適解の数: {analysis['optimal_count']:,} "
              f"({analysis['optimal_count']/analysis['total_solutions']*100:.2f}%)")
        
        print("\n【コストの分位点】")
        for p, v in analysis['percentiles'].items():
            print(f"  {p}%点: {v:.4f}")
        
        return analysis
    
    def visualize_all_solutions(self, save_path: str = None) -> plt.Figure:
        """
        全解のコスト分布を2次元でプロット
        
        Parameters:
        -----------
        save_path : str, optional
            保存先パス
        
        Returns:
        --------
        plt.Figure : 図
        """
        if not self.all_solutions:
            raise ValueError("先に brute_force() を実行してください")
        
        costs = np.array([sol[1] for sol in self.all_solutions])
        min_cost = np.min(costs)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 全解のコスト散布図（ソート済み）
        ax1 = axes[0, 0]
        sorted_costs = np.sort(costs)
        ax1.scatter(range(len(sorted_costs)), sorted_costs, 
                   c=sorted_costs, cmap='viridis', s=1, alpha=0.5)
        ax1.axhline(y=min_cost, color='red', linestyle='--', linewidth=2, label=f'Optimal: {min_cost:.4f}')
        ax1.set_xlabel('Solution Rank')
        ax1.set_ylabel('Cost')
        ax1.set_title('All Solutions (Sorted by Cost)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. コストのヒストグラム
        ax2 = axes[0, 1]
        n_bins = min(100, len(costs) // 100)
        n, bins, patches = ax2.hist(costs, bins=n_bins, edgecolor='black', alpha=0.7)
        
        # 最適解の位置を強調
        ax2.axvline(x=min_cost, color='red', linestyle='--', linewidth=2, label=f'Optimal: {min_cost:.4f}')
        ax2.set_xlabel('Cost')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Cost Distribution (Histogram)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 累積分布
        ax3 = axes[1, 0]
        sorted_costs = np.sort(costs)
        cumulative = np.arange(1, len(sorted_costs) + 1) / len(sorted_costs)
        ax3.plot(sorted_costs, cumulative, 'b-', linewidth=1)
        ax3.axvline(x=min_cost, color='red', linestyle='--', linewidth=2, label=f'Optimal: {min_cost:.4f}')
        
        # 最適解から5%以内の解の割合を表示
        threshold = min_cost * 1.05
        within_5pct = np.sum(costs <= threshold) / len(costs) * 100
        ax3.axvline(x=threshold, color='orange', linestyle=':', linewidth=2, 
                   label=f'5% threshold: {threshold:.4f} ({within_5pct:.1f}%)')
        
        ax3.set_xlabel('Cost')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 最適解付近の詳細（上位10%）
        ax4 = axes[1, 1]
        top_10pct_threshold = np.percentile(costs, 10)
        top_solutions = [(i, c) for i, c in enumerate(costs) if c <= top_10pct_threshold]
        top_indices = [s[0] for s in top_solutions]
        top_costs = [s[1] for s in top_solutions]
        
        ax4.scatter(range(len(top_costs)), sorted(top_costs), 
                   c=sorted(top_costs), cmap='RdYlGn_r', s=10, alpha=0.7)
        ax4.axhline(y=min_cost, color='red', linestyle='--', linewidth=2, label=f'Optimal: {min_cost:.4f}')
        ax4.set_xlabel('Solution Rank (Top 10%)')
        ax4.set_ylabel('Cost')
        ax4.set_title(f'Top 10% Solutions ({len(top_costs):,} solutions)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n図を保存しました: {save_path}")
        
        return fig
    
    def visualize_local_optima(self, save_path: str = None) -> plt.Figure:
        """
        局所解の分析と可視化
        
        Parameters:
        -----------
        save_path : str, optional
            保存先パス
        
        Returns:
        --------
        plt.Figure : 図
        """
        if not self.all_solutions:
            raise ValueError("先に brute_force() を実行してください")
        
        costs = np.array([sol[1] for sol in self.all_solutions])
        min_cost = np.min(costs)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. コスト値の度数分布（局所解のクラスタを可視化）
        ax1 = axes[0]
        
        # コストを丸めてクラスタリング
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
        
        # 上位20個の異なるコスト値
        sorted_unique_costs = np.sort(unique_costs)[:20]
        counts_top = [counts[np.where(unique_costs == c)[0][0]] for c in sorted_unique_costs]
        
        bars = ax2.barh(range(len(sorted_unique_costs)), counts_top, 
                       color=['red' if c == min_cost else 'steelblue' for c in sorted_unique_costs])
        ax2.set_yticks(range(len(sorted_unique_costs)))
        ax2.set_yticklabels([f'{c:.4f}' for c in sorted_unique_costs])
        ax2.set_xlabel('Number of Solutions')
        ax2.set_ylabel('Cost')
        ax2.set_title('Top 20 Local Optima (by cost)')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        # 各バーにラベルを追加
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
        
        # コストでソート
        sorted_solutions = sorted(self.all_solutions, key=lambda x: x[1])
        
        print(f"\n" + "=" * 70)
        print(f"【上位{n}個の解】")
        print("=" * 70)
        
        for rank, (placement, cost) in enumerate(sorted_solutions[:n], 1):
            labels = [self.state_labels[placement[i]] for i in range(self.n_states)]
            print(f"#{rank}: コスト={cost:.4f}  配置: {' '.join(labels)}")


def load_transition_matrix(filepath: str) -> np.ndarray:
    """CSVファイルから遷移確率行列を読み込む"""
    try:
        matrix = np.loadtxt(filepath, delimiter=',')
        print(f"✅ 遷移確率行列を読み込みました: {filepath}")
        print(f"   サイズ: {matrix.shape[0]} × {matrix.shape[1]}")
        return matrix
    except FileNotFoundError:
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")


def main():
    """メイン実行関数"""
    print("=" * 70)
    print("グループ制約付き配置最適化（総当たり専用版）")
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
    
    # グループ定義（6-4-3-2 の構成）
    # ※ここを変更すればグループ構成を変えられます
    groups = {
        'Group1': {
            'states': get_indices(['G', 'H', 'I', 'J', 'E', 'F']),  # 6個
            'positions': (0, 5)   # 位置1〜6
        },
        'Group2': {
            'states': get_indices(['A', 'C', 'D', 'B']),            # 4個
            'positions': (6, 9)   # 位置7〜10
        },
        'Group3': {
            'states': get_indices(['K', 'L', 'M']),                 # 3個
            'positions': (10, 12) # 位置11〜13
        },
        'Group4': {
            'states': get_indices(['N', 'O']),                      # 2個
            'positions': (13, 14) # 位置14〜15
        }
    }
    
    # ============================================================
    # 遷移確率行列の読み込み
    # ============================================================
    
    transition_matrix = load_transition_matrix(input_file)
    
    # ============================================================
    # グループ設定の表示
    # ============================================================
    
    print("\n【グループ設定】")
    for group_name, group_info in groups.items():
        states_str = ', '.join([state_labels[s] for s in group_info['states']])
        pos = group_info['positions']
        n_states = len(group_info['states'])
        print(f"  {group_name} ({n_states}個): [{states_str}] → 位置{pos[0]+1}〜{pos[1]+1}")
    
    # 組み合わせ数の計算
    total = 1
    for group_info in groups.values():
        total *= math.factorial(len(group_info['states']))
    print(f"\n予想組み合わせ数: {total:,} 通り")
    
    # ============================================================
    # 最適化の実行
    # ============================================================
    
    optimizer = BruteForceOptimizer(
        transition_matrix,
        groups,
        state_labels
    )
    
    # 総当たり探索
    best_placement, best_cost = optimizer.brute_force(verbose=True)
    
    # 結果表示
    optimizer.print_solution(best_placement, best_cost)
    
    # 上位10個の解を表示
    optimizer.print_top_solutions(n=10)
    
    # ============================================================
    # 全解の分析と可視化
    # ============================================================
    
    # 統計分析
    analysis = optimizer.analyze_solutions()
    
    # 全解のコスト分布を可視化
    fig1 = optimizer.visualize_all_solutions(save_path='all_solutions_distribution.png')
    
    # 局所解の分析
    fig2 = optimizer.visualize_local_optima(save_path='local_optima_analysis.png')
    
    print("\n" + "=" * 70)
    print("完了")
    print("=" * 70)
    
    # plt.show()  # 画面に表示したい場合


if __name__ == "__main__":
    main()
