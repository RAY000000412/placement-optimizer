"""
配置最適化プログラム（完全総当たり版：グループ順列×グループ内配置）
optimize_placement_bruteforce_1209_2.py

機能:
- 遷移確率行列に対応
- グループ分けによる配置位置の制限
- グループの並び順（順列）も総当たりで探索
- グループ内の要素配置も総当たりで探索
- 目的関数: Σ(距離 × 遷移確率) を最小化（全体で計算）

作成日: 2024/12/09

計算量の目安（4グループ 6-4-3-2構成の場合）:
- グループ順列: 4! = 24通り
- グループ内配置: 6! × 4! × 3! × 2! = 207,360通り
- 合計: 24 × 207,360 = 4,976,640通り
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict
from itertools import permutations, product
from math import factorial


class BruteForceOptimizerWithGroupPermutation:
    """
    完全総当たり配置最適化クラス（グループ順列対応版）
    
    グループの並び順とグループ内の要素配置を全組み合わせで探索し、
    厳密な最適解を求める
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
        
        # グループ名のリスト
        self.group_names = list(groups.keys())
        
        # 基本検証
        self._validate_groups()
        
        # 計算量の見積もり
        self._estimate_complexity()
        
        # 結果保存用
        self.all_solutions = []
        self.group_permutation_summary = []
    
    def _validate_groups(self):
        """グループ設定の妥当性を検証"""
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
    
    def _estimate_complexity(self):
        """計算量を見積もる"""
        # グループ順列数
        n_group_perms = factorial(len(self.group_names))
        
        # グループ内配置数
        n_internal_perms = 1
        for group_name in self.group_names:
            n_states = len(self.groups_original[group_name]['states'])
            n_internal_perms *= factorial(n_states)
        
        # 合計
        total = n_group_perms * n_internal_perms
        
        print(f"\n【計算量の見積もり】")
        print(f"  グループ数: {len(self.group_names)}")
        print(f"  グループ順列数: {n_group_perms:,}")
        print(f"  グループ内配置数: {n_internal_perms:,}")
        print(f"  総組み合わせ数: {total:,}")
        
        # 時間見積もり（1秒あたり約20万組み合わせを処理できると仮定）
        estimated_seconds = total / 200000
        if estimated_seconds < 60:
            print(f"  推定実行時間: 約{estimated_seconds:.1f}秒")
        elif estimated_seconds < 3600:
            print(f"  推定実行時間: 約{estimated_seconds/60:.1f}分")
        else:
            print(f"  推定実行時間: 約{estimated_seconds/3600:.1f}時間")
        
        self.n_group_perms = n_group_perms
        self.n_internal_perms = n_internal_perms
        self.total_combinations = total
    
    def _assign_positions(self, group_order: List[str]) -> Dict[str, Tuple[int, int]]:
        """グループ順序に基づいて位置範囲を割り当てる"""
        positions = {}
        current_pos = 0
        
        for group_name in group_order:
            n_states = len(self.groups_original[group_name]['states'])
            positions[group_name] = (current_pos, current_pos + n_states - 1)
            current_pos += n_states
        
        return positions
    
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
    
    def _generate_internal_permutations(self, group_order: List[str]):
        """
        指定されたグループ順序に対して、グループ内配置の全組み合わせを生成
        
        Yields:
        -------
        List[int] : 配置リスト
        """
        # 各グループの状態の全順列を生成
        group_perms = []
        for group_name in group_order:
            states = self.groups_original[group_name]['states']
            group_perms.append(list(permutations(states)))
        
        # 全グループの順列の直積（組み合わせ）
        for combo in product(*group_perms):
            # 配置リストを構築
            placement = []
            for group_states in combo:
                placement.extend(group_states)
            yield placement
    
    def optimize(self, verbose: bool = True, save_all: bool = False) -> Tuple[List[int], float, List[str]]:
        """
        完全総当たり探索を実行
        
        Parameters:
        -----------
        verbose : bool
            進捗を表示するか
        save_all : bool
            全解を保存するか（メモリ注意）
        
        Returns:
        --------
        Tuple[List[int], float, List[str]]:
            (最良配置, 最良コスト, 最良グループ順序)
        """
        print("\n" + "=" * 70)
        print("完全総当たり探索を開始")
        print("=" * 70)
        
        start_time = time.time()
        
        best_placement = None
        best_cost = float('inf')
        best_group_order = None
        
        self.all_solutions = []
        self.group_permutation_summary = []
        
        all_group_orders = list(permutations(self.group_names))
        
        total_evaluated = 0
        
        for g_idx, group_order in enumerate(all_group_orders):
            group_order = list(group_order)
            
            if verbose:
                order_str = ' → '.join(group_order)
                print(f"\n[{g_idx+1}/{self.n_group_perms}] グループ順序: {order_str}")
            
            # このグループ順序での最良解
            best_cost_for_order = float('inf')
            best_placement_for_order = None
            count_for_order = 0
            
            # グループ内配置の全組み合わせを探索
            for placement in self._generate_internal_permutations(group_order):
                cost = self.calculate_cost(placement)
                count_for_order += 1
                total_evaluated += 1
                
                if save_all:
                    self.all_solutions.append({
                        'group_order': group_order,
                        'placement': placement.copy(),
                        'cost': cost
                    })
                
                if cost < best_cost_for_order:
                    best_cost_for_order = cost
                    best_placement_for_order = placement.copy()
                
                if cost < best_cost:
                    best_cost = cost
                    best_placement = placement.copy()
                    best_group_order = group_order
            
            # このグループ順序の結果を記録
            self.group_permutation_summary.append({
                'group_order': group_order,
                'best_cost': best_cost_for_order,
                'best_placement': best_placement_for_order,
                'n_evaluated': count_for_order
            })
            
            if verbose:
                print(f"  最良コスト: {best_cost_for_order:.4f}")
                if best_cost_for_order == best_cost:
                    print(f"  ★ 全体最良解!")
        
        elapsed_time = time.time() - start_time
        
        # 結果をコストでソート
        self.group_permutation_summary.sort(key=lambda x: x['best_cost'])
        
        # サマリー表示
        print("\n" + "=" * 70)
        print("【完全総当たり探索結果】")
        print("=" * 70)
        print(f"\n総評価数: {total_evaluated:,}")
        print(f"実行時間: {elapsed_time:.2f}秒")
        print(f"処理速度: {total_evaluated/elapsed_time:,.0f} 配置/秒")
        
        print(f"\n最良グループ順序: {' → '.join(best_group_order)}")
        print(f"最良コスト: {best_cost:.4f}")
        
        print("\n【グループ順列別の最良コスト（上位10件）】")
        for i, result in enumerate(self.group_permutation_summary[:10]):
            order_str = ' → '.join(result['group_order'])
            marker = "★" if result['best_cost'] == best_cost else " "
            print(f"  {marker} {i+1}. {order_str}: コスト = {result['best_cost']:.4f}")
        
        # 現在のグループ設定を最良順序で更新
        self.current_group_order = best_group_order
        self.current_positions = self._assign_positions(best_group_order)
        
        return best_placement, best_cost, best_group_order
    
    def print_solution(self, placement: List[int], cost: float):
        """解を表示"""
        print("\n" + "=" * 70)
        print("【最適化結果】")
        print("=" * 70)
        
        if hasattr(self, 'current_group_order') and hasattr(self, 'current_positions'):
            print("\n【グループ別配置】")
            for group_name in self.current_group_order:
                pos_start, pos_end = self.current_positions[group_name]
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
    
    def visualize_results(self, placement: List[int]) -> plt.Figure:
        """結果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 配置の可視化
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.group_names)))
        
        if hasattr(self, 'current_group_order') and hasattr(self, 'current_positions'):
            group_colors = {name: colors[i] for i, name in enumerate(self.current_group_order)}
            
            # 状態→グループのマッピング
            state_to_group = {}
            for group_name in self.group_names:
                for state in self.groups_original[group_name]['states']:
                    state_to_group[state] = group_name
            
            for pos in range(self.n_states):
                state = placement[pos]
                group = state_to_group[state]
                color = group_colors[group]
                ax1.barh(0, 1, left=pos, color=color, edgecolor='black', linewidth=1)
                ax1.text(pos + 0.5, 0, self.state_labels[state], 
                        ha='center', va='center', fontsize=10, fontweight='bold')
            
            # グループ境界線
            for group_name in self.current_group_order:
                pos_end = self.current_positions[group_name][1]
                if pos_end < self.n_states - 1:
                    ax1.axvline(x=pos_end + 1, color='red', linewidth=2, linestyle='--')
            
            # 凡例
            legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=group_colors[g], 
                              edgecolor='black', label=g) for g in self.current_group_order]
            ax1.legend(handles=legend_elements, loc='upper right')
        
        ax1.set_xlim(0, self.n_states)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('Position')
        ax1.set_title('Optimized Placement (Grouped)')
        ax1.set_yticks([])
        
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
        
        # 3. グループ順列ごとのコスト比較
        ax3 = axes[1, 0]
        if self.group_permutation_summary:
            costs = [r['best_cost'] for r in self.group_permutation_summary]
            labels_bar = [' → '.join(r['group_order']) for r in self.group_permutation_summary]
            
            y_pos = range(len(costs))
            bars = ax3.barh(y_pos, costs, color='steelblue')
            
            min_cost = min(costs)
            for i, (bar, cost) in enumerate(zip(bars, costs)):
                if cost == min_cost:
                    bar.set_color('gold')
            
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(labels_bar, fontsize=8)
            ax3.set_xlabel('Cost')
            ax3.set_title('Best Cost by Group Permutation')
            ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. コスト分布（全解を保存している場合）
        ax4 = axes[1, 1]
        if self.all_solutions:
            all_costs = [s['cost'] for s in self.all_solutions]
            ax4.hist(all_costs, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax4.axvline(x=min(all_costs), color='red', linestyle='--', linewidth=2, label=f'Best: {min(all_costs):.4f}')
            ax4.set_xlabel('Cost')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Cost Distribution (All Solutions)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'All solutions not saved\n(set save_all=True)', 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('Cost Distribution')
        
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


def create_sample_transition_matrix(n: int = 15, seed: int = 42) -> np.ndarray:
    """サンプル遷移確率行列を生成"""
    np.random.seed(seed)
    
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
    print("完全総当たり配置最適化プログラム（グループ順列対応版）")
    print("=" * 70)
    
    # ============================================================
    # 設定
    # ============================================================
    
    input_file = 'transition_matrix.csv'
    
    # 状態ラベル
    state_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    
    def get_indices(labels):
        return [state_labels.index(label) for label in labels]
    
    # グループ定義（6-4-3-2構成）
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
    
    # ============================================================
    # 遷移確率行列の読み込み
    # ============================================================
    
    try:
        transition_matrix = load_transition_matrix(input_file)
    except FileNotFoundError:
        print(f"{input_file} が見つかりません。サンプルデータを生成します。")
        transition_matrix = create_sample_transition_matrix(15)
    
    # ============================================================
    # グループ設定の表示
    # ============================================================
    
    print("\n【グループ設定】")
    for group_name, group_info in groups.items():
        states_str = ', '.join([state_labels[s] for s in group_info['states']])
        print(f"  {group_name}: 状態[{states_str}] ({len(group_info['states'])}個)")
    
    # ============================================================
    # オプティマイザの作成と実行
    # ============================================================
    
    optimizer = BruteForceOptimizerWithGroupPermutation(
        transition_matrix, 
        groups,
        state_labels
    )
    
    # 完全総当たり探索を実行
    # save_all=True にすると全解を保存（メモリ注意）
    best_placement, best_cost, best_group_order = optimizer.optimize(
        verbose=True,
        save_all=False  # 全解保存は無効（メモリ節約）
    )
    
    # 結果の詳細表示
    optimizer.print_solution(best_placement, best_cost)
    optimizer.analyze_cost_contributions(best_placement)
    
    # 結果の可視化
    fig = optimizer.visualize_results(best_placement)
    plt.savefig('optimization_results_bruteforce_1209_2.png', dpi=150, bbox_inches='tight')
    print("\n結果を optimization_results_bruteforce_1209_2.png に保存しました")


if __name__ == "__main__":
    main()
