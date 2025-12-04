# Grouped Placement Optimizer

グループ制約付き配置最適化プログラム

## 概要

遷移確率行列に基づいて、要素の配置を最適化します。  
各要素をグループに分け、グループごとに配置可能な位置範囲を制限しながら、全体の目的関数（Σ距離×遷移確率）を最小化します。

### 特徴

- 任意のn×n遷移確率行列に対応（デフォルトは15×15）
- グループ制約による配置位置の制限
- 貪欲法 + 焼きなまし法によるハイブリッド最適化
- 結果の可視化機能

## 必要環境

- Python 3.8+
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```

## ファイル構成

```
.
├── optimize_placement_1204.py   # メインプログラム
├── run_optimization.py          # 使用例サンプル
└── README.md
```

## 使い方

### 基本的な使い方

```bash
python run_optimization.py
```

### 自分のデータで使う場合

`run_optimization.py` を編集します。

#### 1. 遷移確率行列の読み込み

```python
# NumPyファイルから
transition_matrix = np.load('your_matrix.npy')

# CSVから
transition_matrix = np.loadtxt('your_matrix.csv', delimiter=',')
```

#### 2. グループの定義

```python
state_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']

def get_indices(labels):
    return [state_labels.index(label) for label in labels]

groups = {
    'Group1': {
        'states': get_indices(['A', 'B', 'C', 'D', 'E']),
        'positions': (0, 4)   # 位置1〜5
    },
    'Group2': {
        'states': get_indices(['F', 'G', 'H', 'I', 'J']),
        'positions': (5, 9)   # 位置6〜10
    },
    'Group3': {
        'states': get_indices(['K', 'L', 'M', 'N', 'O']),
        'positions': (10, 14) # 位置11〜15
    }
}
```

任意の組み合わせも可能です：

```python
groups = {
    'Group1': {
        'states': get_indices(['F', 'A', 'O', 'K', 'J']),  # 任意の5つ
        'positions': (0, 4)
    },
    # ...
}
```

#### 3. 最適化の実行

```python
from optimize_placement_1204 import GroupConstrainedPlacementOptimizer

optimizer = GroupConstrainedPlacementOptimizer(
    transition_matrix=transition_matrix,
    groups=groups,
    state_labels=state_labels
)

best_placement, best_cost = optimizer.simulated_annealing(
    initial_temp=100.0,
    cooling_rate=0.997,
    min_temp=0.1,
    max_iterations=25000
)

optimizer.print_solution(best_placement, best_cost)
```

## パラメータ説明

| パラメータ | 説明 | デフォルト値 |
|-----------|------|-------------|
| `initial_temp` | 焼きなまし法の初期温度 | 100.0 |
| `cooling_rate` | 冷却率（大きいほどゆっくり） | 0.997 |
| `min_temp` | 最小温度 | 0.1 |
| `max_iterations` | 最大反復回数 | 25000 |
| `use_greedy_init` | 貪欲法で初期解を生成するか | True |

## 出力例

```
【全体配置】
位置:    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15 
状態:    B    E    D    A    C    I    G    J    H    F    N    O    L    M    K 

総コスト: 70.8426
```

## 可視化

```python
fig = optimizer.visualize_results(best_placement)
plt.savefig('result.png', dpi=150, bbox_inches='tight')
```

出力される画像：
- 配置結果（グループ別に色分け）
- 遷移確率行列（配置順に並べ替え）
- コストの推移
- 温度と受理率の推移

## ライセンス

MIT License
