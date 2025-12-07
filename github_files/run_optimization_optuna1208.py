"""
optimize_placement_optuna.py の使い方サンプル

必要なライブラリ:
    pip install numpy matplotlib optuna

実行方法:
    python run_optimization_optuna.py
"""

import numpy as np
from optimize_placement_optuna import GroupConstrainedPlacementOptimizer
import matplotlib.pyplot as plt

# ============================================================
# ステップ1: 遷移確率行列を用意する
# ============================================================

# サンプルとして15×15のランダムな遷移確率行列を作成
#np.random.seed(123)
#n = 15
transition_matrix = np.loadtxt('transition_matrix.csv', delimiter=',')
#for i in range(n):
#    targets = np.random.choice([j for j in range(n) if j != i], size=3, replace=False)
#    probs = np.random.dirichlet(np.ones(3))
#    for t, p in zip(targets, probs):
#        transition_matrix[i, t] = p

print("遷移確率行列のサイズ:", transition_matrix.shape)

# ============================================================
# ステップ2: グループを定義する
# ============================================================

state_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']

def get_indices(labels):
    return [state_labels.index(label) for label in labels]

groups = {
    'Group1': {
        'states': get_indices(['A', 'B', 'C', 'D', 'E']),
        'positions': (0, 4)
    },
    'Group2': {
        'states': get_indices(['F', 'G', 'H', 'I', 'J']),
        'positions': (5, 9)
    },
    'Group3': {
        'states': get_indices(['K', 'L', 'M', 'N', 'O']),
        'positions': (10, 14)
    }
}

# ============================================================
# ステップ3: オプティマイザを作成
# ============================================================

optimizer = GroupConstrainedPlacementOptimizer(
    transition_matrix=transition_matrix,
    groups=groups,
    state_labels=state_labels
)

# ============================================================
# 方法A: 従来の手動パラメータ指定（高速）
# ============================================================

print("\n" + "=" * 60)
print("【方法A】手動パラメータで実行")
print("=" * 60)

best_placement, best_cost = optimizer.simulated_annealing(
    initial_temp=100.0,
    cooling_rate=0.997,
    max_iterations=25000
)

optimizer.print_solution(best_placement, best_cost)

# ============================================================
# 方法B: Optunaでパラメータ自動調整（推奨）
# ============================================================

print("\n" + "=" * 60)
print("【方法B】Optunaでパラメータ自動調整")
print("=" * 60)

# 一括実行（パラメータ探索 → 最適パラメータで実行）
best_placement_optuna, best_cost_optuna = optimizer.optimize_with_optuna(
    n_trials=30,           # 試行回数（多いほど良いパラメータが見つかりやすい）
    n_runs_per_trial=2,    # 各試行での実行回数（平均を取る）
    n_final_runs=5         # 最終実行の回数（最良を採用）
)

optimizer.print_solution(best_placement_optuna, best_cost_optuna)

# ============================================================
# 方法C: パラメータ探索と実行を分ける
# ============================================================

# # パラメータ探索のみ
# best_params = optimizer.tune_with_optuna(n_trials=50)
# print("最適パラメータ:", best_params)
#
# # 最適パラメータで実行
# best_placement, best_cost = optimizer.run_with_best_params(n_runs=10)

# ============================================================
# ステップ4: 結果を可視化
# ============================================================

# 配置結果の可視化
fig1 = optimizer.visualize_results(best_placement_optuna)
plt.savefig('result_placement.png', dpi=150, bbox_inches='tight')
print("\n配置結果を result_placement.png に保存しました")

# Optuna分析結果の可視化
fig2 = optimizer.visualize_optuna_results()
plt.savefig('result_optuna_analysis.png', dpi=150, bbox_inches='tight')
print("Optuna分析結果を result_optuna_analysis.png に保存しました")

# plt.show()  # 画面に表示したい場合
