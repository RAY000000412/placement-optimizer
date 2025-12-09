"""
optimize_placement_bruteforce_1209.py の使い方サンプル
このファイルを optimize_placement_bruteforce_1209.py と同じフォルダに置いて実行してください。

実行方法:
    python run_optimization_1209.py

特徴:
- グループの並び順（順列）も含めた完全総当たり探索
- 厳密な最適解が保証される
- 計算時間はかかるが、中規模問題なら実用的
"""

import numpy as np
from optimize_placement_bruteforce_1209 import BruteForceOptimizerWithGroupPermutation
import math

# ============================================================
# ステップ1: 遷移確率行列を用意する
# ============================================================

# 方法A: CSVから読み込む場合
# transition_matrix = np.loadtxt('transition_matrix.csv', delimiter=',')

# 方法B: npyファイルから読み込む場合
# transition_matrix = np.load('your_transition_matrix.npy')

# 方法C: 手動で作成する場合（15×15の例）
np.random.seed(123)
n = 15
transition_matrix = np.zeros((n, n))
for i in range(n):
    targets = np.random.choice([j for j in range(n) if j != i], size=3, replace=False)
    probs = np.random.dirichlet(np.ones(3))
    for t, p in zip(targets, probs):
        transition_matrix[i, t] = p

print("遷移確率行列のサイズ:", transition_matrix.shape)

# ============================================================
# ステップ2: グループを定義する
# ============================================================

# 状態ラベル（A〜O）
state_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']

# ラベル名からインデックスを取得するヘルパー関数
def get_indices(labels):
    return [state_labels.index(label) for label in labels]

# ---------- グループ定義 ----------
# ※ 'positions' は不要（グループ順列探索時に自動計算）
groups = {
    'Group1': {
        'states': get_indices(['A', 'B', 'C', 'D', 'E']),  # 5個
    },
    'Group2': {
        'states': get_indices(['F', 'G', 'H', 'I', 'J']),  # 5個
    },
    'Group3': {
        'states': get_indices(['K', 'L', 'M', 'N', 'O']),  # 5個
    }
}

# ============================================================
# 計算量の確認
# ============================================================

print("\n" + "=" * 60)
print("計算量の確認")
print("=" * 60)

n_groups = len(groups)
n_group_perms = math.factorial(n_groups)
n_internal = 1
for name, info in groups.items():
    n_states = len(info['states'])
    n_internal *= math.factorial(n_states)
    print(f"  {name}: {n_states}個 → {math.factorial(n_states):,}通り")

total = n_group_perms * n_internal
print(f"\nグループ順列: {n_group_perms:,} 通り")
print(f"グループ内配置: {n_internal:,} 通り")
print(f"合計: {total:,} 通り")

# 実行時間の目安
estimated_time = total / 50000  # 約5万件/秒として概算
print(f"\n推定実行時間: 約 {estimated_time:.1f} 秒")

# ============================================================
# ステップ3: 最適化を実行する
# ============================================================

print("\n" + "=" * 60)
print("最適化を開始します")
print("=" * 60)

# オプティマイザを作成
optimizer = BruteForceOptimizerWithGroupPermutation(
    transition_matrix=transition_matrix,
    groups=groups,
    state_labels=state_labels
)

# 完全総当たり探索を実行
best_placement, best_cost, best_group_order = optimizer.brute_force(verbose=True)

# ============================================================
# ステップ4: 結果を表示する
# ============================================================

optimizer.print_solution(best_placement, best_cost)
optimizer.analyze_cost_contributions(best_placement)
optimizer.print_top_solutions(n=10)

# ============================================================
# ステップ5: 可視化して保存する（オプション）
# ============================================================

import matplotlib.pyplot as plt

# 全解の分布
fig1 = optimizer.visualize_all_solutions(save_path='my_result_distribution_1209.png')

# 局所解の分析
fig2 = optimizer.visualize_local_optima(save_path='my_result_local_optima_1209.png')

print("\n結果を保存しました:")
print("  - my_result_distribution_1209.png")
print("  - my_result_local_optima_1209.png")

# plt.show()  # 画面に表示したい場合


# ============================================================
# 補足: 計算量の目安
# ============================================================
"""
【グループ順列×グループ内配置の計算量】

例1: 3グループ（5-5-5構成）
  グループ順列: 3! = 6
  グループ内: 5! × 5! × 5! = 120 × 120 × 120 = 1,728,000
  合計: 6 × 1,728,000 = 約1000万 → 約3分

例2: 4グループ（6-4-3-2構成）
  グループ順列: 4! = 24
  グループ内: 6! × 4! × 3! × 2! = 720 × 24 × 6 × 2 = 207,360
  合計: 24 × 207,360 = 約500万 → 約2分

例3: 5グループ（5-4-3-2-1構成）
  グループ順列: 5! = 120
  グループ内: 5! × 4! × 3! × 2! × 1! = 120 × 24 × 6 × 2 × 1 = 34,560
  合計: 120 × 34,560 = 約400万 → 約1-2分

目安:
  〜100万: 数十秒
  〜1000万: 数分
  〜1億: 30分〜1時間
  10億〜: 非現実的（焼きなまし法を検討）
"""
