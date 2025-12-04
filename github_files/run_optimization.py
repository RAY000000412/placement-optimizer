"""
optimize_placement_1204.py の使い方サンプル
このファイルを optimize_placement_1204.py と同じフォルダに置いて実行してください。

実行方法:
    python run_optimization.py
"""

import numpy as np
from optimize_placement_1204 import GroupConstrainedPlacementOptimizer

# ============================================================
# ステップ1: 遷移確率行列を用意する
# ============================================================

# 方法A: 自分のファイルから読み込む場合
# transition_matrix = np.load('your_transition_matrix.npy')

# 方法B: CSVから読み込む場合
# transition_matrix = np.loadtxt('your_matrix.csv', delimiter=',')

# 方法C: 手動で作成する場合（15×15の例）
# ここではサンプルとしてランダムなデータを作成
np.random.seed(123)
n = 15  # 状態数
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
    """ラベル名のリストからインデックスのリストを返す"""
    return [state_labels.index(label) for label in labels]

# ---------- グループ定義の例1: 標準的な分け方 ----------
groups_standard = {
    'Group1': {
        'states': get_indices(['A', 'B', 'C', 'D', 'E']),  # → [0, 1, 2, 3, 4]
        'positions': (0, 4)   # 位置1〜5に配置（0始まりなので0〜4）
    },
    'Group2': {
        'states': get_indices(['F', 'G', 'H', 'I', 'J']),  # → [5, 6, 7, 8, 9]
        'positions': (5, 9)   # 位置6〜10に配置
    },
    'Group3': {
        'states': get_indices(['K', 'L', 'M', 'N', 'O']),  # → [10, 11, 12, 13, 14]
        'positions': (10, 14) # 位置11〜15に配置
    }
}

# ---------- グループ定義の例2: 任意の組み合わせ ----------
groups_custom = {
    'Group1': {
        'states': get_indices(['F', 'A', 'O', 'K', 'J']),  # 任意の5つを選択
        'positions': (0, 4)
    },
    'Group2': {
        'states': get_indices(['B', 'M', 'C', 'N', 'D']),
        'positions': (5, 9)
    },
    'Group3': {
        'states': get_indices(['E', 'G', 'H', 'I', 'L']),
        'positions': (10, 14)
    }
}

# ============================================================
# ステップ3: 最適化を実行する
# ============================================================

# 使用するグループ設定を選択（例1を使用）
groups = groups_standard

print("\n" + "=" * 60)
print("グループ設定:")
for name, info in groups.items():
    states_str = ', '.join([state_labels[i] for i in info['states']])
    pos = info['positions']
    print(f"  {name}: [{states_str}] → 位置{pos[0]+1}〜{pos[1]+1}")
print("=" * 60)

# オプティマイザを作成
optimizer = GroupConstrainedPlacementOptimizer(
    transition_matrix=transition_matrix,
    groups=groups,
    state_labels=state_labels
)

# 最適化を実行
best_placement, best_cost = optimizer.simulated_annealing(
    initial_temp=100.0,      # 初期温度
    cooling_rate=0.997,      # 冷却率（大きいほどゆっくり冷却）
    min_temp=0.1,            # 最小温度
    max_iterations=25000     # 最大反復回数
)

# ============================================================
# ステップ4: 結果を表示する
# ============================================================

optimizer.print_solution(best_placement, best_cost)

# ============================================================
# ステップ5: 可視化して保存する（オプション）
# ============================================================

import matplotlib.pyplot as plt

fig = optimizer.visualize_results(best_placement)
plt.savefig('my_optimization_result.png', dpi=150, bbox_inches='tight')
print("\n結果を my_optimization_result.png に保存しました")

# plt.show()  # 画面に表示したい場合はコメントを外す
