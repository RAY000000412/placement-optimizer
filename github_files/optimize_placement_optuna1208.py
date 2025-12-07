"""
配置最適化プログラム（グループ制約付き・Optuna対応版）
optimize_placement_optuna.py

機能:
- 15×15の遷移確率行列に対応
- グループ分けによる配置位置の制限
- 目的関数: Σ(距離 × 遷移確率) を最小化（全体で計算）
- Optunaによるパラメータ自動調整

作成日: 2024/12/04
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Optional
import random
import warnings

# Optunaのインポート（インストールされていない場合のエラーハンドリング）
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optunaがインストールされていません。pip install optuna でインストールしてください。")


class GroupConstrainedPlacementOptimizer:
    """
    グループ制約付き配置最適化クラス（Optuna対応版）
    
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
        
        # Optunaの最適化結果を保存
        self.optuna_study = None
        self.best_params = None
    
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
        """貪欲法による初期配置生成（グループ制約付き）"""
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
        """ランダムな初期配置生成（グループ制約付き）"""
        placement = [None] * self.n_states
        
        for group_name, group_info in self.groups.items():
            states = list(group_info['states'])
            pos_start, pos_end = group_info['positions']
            
            random.shuffle(states)
            for i, pos in enumerate(range(pos_start, pos_end + 1)):
                placement[pos] = states[i]
        
        return placement
    
    def get_neighbor(self, placement: List[int]) -> List[int]:
        """近傍解の生成（同一グループ内での交換のみ）"""
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
        verbose : bool
            進捗を表示するか
        
        Returns:
        --------
        Tuple[List[int], float] : (最良配置, 最良コスト)
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
    
    def tune_with_optuna(self,
                         n_trials: int = 50,
                         n_runs_per_trial: int = 3,
                         timeout: float = None,
                         show_progress: bool = True) -> Dict:
        """
        Optunaによるパラメータ自動調整
        
        Parameters:
        -----------
        n_trials : int
            試行回数（デフォルト: 50）
        n_runs_per_trial : int
            各試行での実行回数（平均を取る、デフォルト: 3）
        timeout : float
            タイムアウト秒数（Noneで無制限）
        show_progress : bool
            進捗バーを表示するか
        
        Returns:
        --------
        Dict : 最適パラメータ
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optunaがインストールされていません。pip install optuna でインストールしてください。")
        
        print("=" * 60)
        print("Optunaによるパラメータ最適化を開始")
        print(f"試行回数: {n_trials}, 各試行の実行回数: {n_runs_per_trial}")
        print("=" * 60)
        
        def objective(trial):
            # パラメータの探索範囲を定義
            initial_temp = trial.suggest_float('initial_temp', 10.0, 500.0, log=True)
            cooling_rate = trial.suggest_float('cooling_rate', 0.990, 0.9995)
            max_iterations = trial.suggest_int('max_iterations', 5000, 50000, step=5000)
            
            # 複数回実行して平均コストを計算（安定性のため）
            costs = []
            for _ in range(n_runs_per_trial):
                _, cost = self.simulated_annealing(
                    initial_temp=initial_temp,
                    cooling_rate=cooling_rate,
                    max_iterations=max_iterations,
                    verbose=False
                )
                costs.append(cost)
            
            avg_cost = np.mean(costs)
            
            # 中間結果を報告（枝刈り用）
            trial.report(avg_cost, 0)
            
            return avg_cost
        
        # Optunaのログレベルを設定
        if not show_progress:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # サンプラーの設定（TPE: Tree-structured Parzen Estimator）
        sampler = TPESampler(seed=42)
        
        # スタディの作成と最適化実行
        self.optuna_study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name='SA_parameter_tuning'
        )
        
        self.optuna_study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress
        )
        
        # 最適パラメータを保存
        self.best_params = self.optuna_study.best_params
        
        print("\n" + "=" * 60)
        print("【最適化完了】")
        print("=" * 60)
        print(f"最良コスト: {self.optuna_study.best_value:.4f}")
        print(f"最適パラメータ:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        return self.best_params
    
    def run_with_best_params(self, 
                             n_runs: int = 5,
                             verbose: bool = True) -> Tuple[List[int], float]:
        """
        Optunaで見つけた最適パラメータで実行
        
        Parameters:
        -----------
        n_runs : int
            実行回数（最良結果を返す）
        verbose : bool
            進捗を表示するか
        
        Returns:
        --------
        Tuple[List[int], float] : (最良配置, 最良コスト)
        """
        if self.best_params is None:
            raise ValueError("先に tune_with_optuna() を実行してください")
        
        if verbose:
            print(f"\n最適パラメータで {n_runs} 回実行...")
        
        best_placement = None
        best_cost = float('inf')
        
        for i in range(n_runs):
            placement, cost = self.simulated_annealing(
                initial_temp=self.best_params['initial_temp'],
                cooling_rate=self.best_params['cooling_rate'],
                max_iterations=self.best_params['max_iterations'],
                verbose=False
            )
            
            if cost < best_cost:
                best_cost = cost
                best_placement = placement
            
            if verbose:
                print(f"  実行 {i+1}/{n_runs}: コスト = {cost:.4f}")
        
        if verbose:
            print(f"最良コスト: {best_cost:.4f}")
        
        return best_placement, best_cost
    
    def optimize_with_optuna(self,
                             n_trials: int = 50,
                             n_runs_per_trial: int = 3,
                             n_final_runs: int = 5,
                             timeout: float = None,
                             show_progress: bool = True) -> Tuple[List[int], float]:
        """
        Optunaでパラメータ調整後、最適パラメータで実行（一括実行）
        
        ※ simulated_annealing() と同じ形式で結果を返す
        
        Parameters:
        -----------
        n_trials : int
            Optunaの試行回数
        n_runs_per_trial : int
            各試行での実行回数
        n_final_runs : int
            最終実行の回数
        timeout : float
            タイムアウト秒数
        show_progress : bool
            進捗を表示するか
        
        Returns:
        --------
        Tuple[List[int], float] : (最良配置, 最良コスト)
        """
        # パラメータ最適化
        self.tune_with_optuna(
            n_trials=n_trials,
            n_runs_per_trial=n_runs_per_trial,
            timeout=timeout,
            show_progress=show_progress
        )
        
        # 最適パラメータで実行
        best_placement, best_cost = self.run_with_best_params(
            n_runs=n_final_runs,
            verbose=show_progress
        )
        
        return best_placement, best_cost
    
    def visualize_optuna_results(self) -> plt.Figure:
        """
        Optunaの最適化結果を可視化
        """
        if self.optuna_study is None:
            raise ValueError("先に tune_with_optuna() を実行してください")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 最適化履歴
        ax1 = axes[0, 0]
        trials = self.optuna_study.trials
        values = [t.value for t in trials if t.value is not None]
        ax1.plot(values, 'b-', alpha=0.7, label='各試行のコスト')
        ax1.plot(np.minimum.accumulate(values), 'r-', linewidth=2, label='最良コスト')
        ax1.set_xlabel('試行回数')
        ax1.set_ylabel('コスト')
        ax1.set_title('最適化履歴')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. パラメータの重要度（簡易版）
        ax2 = axes[0, 1]
        param_names = list(self.best_params.keys())
        
        # 各パラメータと目的関数の相関を計算
        correlations = []
        for param in param_names:
            param_values = [t.params[param] for t in trials if t.value is not None]
            cost_values = [t.value for t in trials if t.value is not None]
            corr = np.abs(np.corrcoef(param_values, cost_values)[0, 1])
            correlations.append(corr if not np.isnan(corr) else 0)
        
        bars = ax2.barh(param_names, correlations, color='steelblue')
        ax2.set_xlabel('コストとの相関（絶対値）')
        ax2.set_title('パラメータの影響度')
        ax2.set_xlim(0, 1)
        
        # 3. initial_temp vs コスト
        ax3 = axes[1, 0]
        initial_temps = [t.params['initial_temp'] for t in trials if t.value is not None]
        costs = [t.value for t in trials if t.value is not None]
        ax3.scatter(initial_temps, costs, alpha=0.6, c='blue')
        ax3.axvline(x=self.best_params['initial_temp'], color='red', linestyle='--', label='最適値')
        ax3.set_xlabel('initial_temp')
        ax3.set_ylabel('コスト')
        ax3.set_title('初期温度 vs コスト')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. cooling_rate vs コスト
        ax4 = axes[1, 1]
        cooling_rates = [t.params['cooling_rate'] for t in trials if t.value is not None]
        ax4.scatter(cooling_rates, costs, alpha=0.6, c='green')
        ax4.axvline(x=self.best_params['cooling_rate'], color='red', linestyle='--', label='最適値')
        ax4.set_xlabel('cooling_rate')
        ax4.set_ylabel('コスト')
        ax4.set_title('冷却率 vs コスト')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
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
    print("グループ制約付き配置最適化プログラム（Optuna対応版）")
    print("=" * 70)
    
    # 遷移確率行列の生成
    transition_matrix = create_sample_transition_matrix_15x15()
    print(f"行列サイズ: {transition_matrix.shape}")
    
    # グループ定義
    groups = {
        'Group1': {
            'states': [0, 1, 2, 3, 4],
            'positions': (0, 4)
        },
        'Group2': {
            'states': [5, 6, 7, 8, 9],
            'positions': (5, 9)
        },
        'Group3': {
            'states': [10, 11, 12, 13, 14],
            'positions': (10, 14)
        }
    }
    
    state_labels = [chr(65 + i) for i in range(15)]
    
    print("\n【グループ設定】")
    for group_name, group_info in groups.items():
        states_str = ', '.join([state_labels[s] for s in group_info['states']])
        pos_start, pos_end = group_info['positions']
        print(f"  {group_name}: 状態[{states_str}] → 位置[{pos_start+1}〜{pos_end+1}]")
    
    # オプティマイザの作成
    optimizer = GroupConstrainedPlacementOptimizer(
        transition_matrix, 
        groups,
        state_labels
    )
    
    # ============================================================
    # 方法1: 従来の手動パラメータ指定
    # ============================================================
    print("\n" + "=" * 70)
    print("【方法1】手動パラメータで実行")
    print("=" * 70)
    
    start_time = time.time()
    best_placement_manual, best_cost_manual = optimizer.simulated_annealing(
        initial_temp=100.0,
        cooling_rate=0.997,
        min_temp=0.1,
        max_iterations=25000
    )
    manual_time = time.time() - start_time
    
    print(f"実行時間: {manual_time:.2f}秒")
    
    # ============================================================
    # 方法2: Optunaによる自動パラメータ調整
    # ============================================================
    if OPTUNA_AVAILABLE:
        print("\n" + "=" * 70)
        print("【方法2】Optunaによる自動パラメータ調整")
        print("=" * 70)
        
        start_time = time.time()
        best_placement_optuna, best_cost_optuna = optimizer.optimize_with_optuna(
            n_trials=30,           # 試行回数（デモ用に少なめ）
            n_runs_per_trial=2,    # 各試行での実行回数
            n_final_runs=5         # 最終実行の回数
        )
        optuna_time = time.time() - start_time
        
        print(f"\n総実行時間: {optuna_time:.2f}秒")
        
        # 結果の比較
        print("\n" + "=" * 70)
        print("【結果比較】")
        print("=" * 70)
        print(f"手動パラメータ: コスト = {best_cost_manual:.4f}, 時間 = {manual_time:.2f}秒")
        print(f"Optuna最適化:  コスト = {best_cost_optuna:.4f}, 時間 = {optuna_time:.2f}秒")
        
        improvement = (best_cost_manual - best_cost_optuna) / best_cost_manual * 100
        if improvement > 0:
            print(f"→ Optunaにより {improvement:.2f}% 改善")
        else:
            print(f"→ 手動パラメータの方が {-improvement:.2f}% 良い結果")
        
        # 結果の表示
        optimizer.print_solution(best_placement_optuna, best_cost_optuna)
        
        # 可視化
        fig1 = optimizer.visualize_results(best_placement_optuna)
        plt.savefig('/home/claude/optimization_results_optuna.png', dpi=150, bbox_inches='tight')
        
        fig2 = optimizer.visualize_optuna_results()
        plt.savefig('/home/claude/optuna_analysis.png', dpi=150, bbox_inches='tight')
        
        print("\n結果を保存しました:")
        print("  - optimization_results_optuna.png")
        print("  - optuna_analysis.png")
    else:
        print("\nOptunaがインストールされていないため、方法2はスキップします")
        optimizer.print_solution(best_placement_manual, best_cost_manual)


if __name__ == "__main__":
    main()
