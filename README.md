# SPSS-style CHAID Visualizer

PythonでSPSS風のCHAID分析結果を可視化するツールです。

ノード内にクラス分布の棒グラフを表示し、Graphvizを利用して見やすいCHAIDツリーを出力できます。

![CHAID結果](images/chaid_result.PNG)

---

## 特徴

* SPSS風の見やすいCHAIDツリー表示
* ノード内棒グラフ表示
* 連続値の自動ビニング
* カテゴリ変数対応
* 日本語データ対応
* GraphvizによるPNG出力
* 欠損値対応
* ケースウェイト対応
* Jupyter Notebookでそのまま実行可能

---

## 利用シーン

* アンケート分析
* 顧客セグメント分析
* 継続率分析
* 満足度分析
* マーケティング分析
* ユーザー離脱分析
* ゲーム運営データ分析

---

## 必要ライブラリ

```bash
pip install -r requirements.txt
```

requirements.txt

```text
numpy
pandas
scipy
graphviz
pillow
jupyter
notebook
```

---

## 注意

Pythonライブラリのgraphvizに加えて、Graphviz本体のインストールが必要です。

### Windows

https://graphviz.org/download/

インストール後、GraphvizのPATH設定を行ってください。

---

## 実行例

```python
from chaid_spss_fast_bars_image_v6 import (
    CHAIDTree,
    infer_variable_types
)

import pandas as pd

# データ読み込み
df = pd.read_csv("sample_data.csv")

# 説明変数
features = [
    "gender",
    "age",
    "plan",
    "play_days"
]

# 変数型自動判定
variable_types = infer_variable_types(
    df,
    target_col="churn",
    independent_cols=features
)

# モデル作成
tree = CHAIDTree(
    max_depth=3,
    min_parent_size=100,
    min_child_size=50,
    alpha_merge=0.05,
    alpha_split=0.05
)

# 学習
tree.fit(
    df=df,
    target_col="churn",
    independent_cols=features,
    variable_types=variable_types
)

# 可視化
dot = tree.to_graphviz(
    bar_mode="image",
    bar_orientation="vertical"
)

dot.render(
    "chaid_tree",
    format="png",
    cleanup=True
)
```

---

## 出力例

![CHAID結果](images/chaid_result.PNG)

---

## ファイル構成

```text
CHAID-Visualizer
├── README.md
├── chaid_visualizer.ipynb
├── chaid_spss_fast_bars_image_v6.py
├── requirements.txt
└── images
    └── chaid_result.png
```










