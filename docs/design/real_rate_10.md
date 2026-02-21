# Real Rate Submodel - Attempt 10 Design Document
# PCA Decomposition of 5-Maturity TIPS Yield Curve

## 0. Architect Fact-Check

データソース確認（attempt 7〜9で検証済み）：

| Series | FRED ID | 頻度 | 確認済み期間 |
|--------|---------|------|-------------|
| 5Y TIPS | DFII5  | 日次 | 2015-01-02〜現在 |
| 7Y TIPS | DFII7  | 日次 | 2015-01-02〜現在 |
| 10Y TIPS| DFII10 | 日次 | 2015-01-02〜現在 |
| 20Y TIPS| DFII20 | 日次 | 2015-01-02〜現在 |
| 30Y TIPS| DFII30 | 日次 | 2015-01-02〜現在 |

**Attempts 7〜9の根本的問題の整理:**

| Attempt | 特徴量 | Gate3 PASS | 主問題 |
|---------|-------|-----------|-------|
| 7 (4feat) | rr_level_change_z, rr_slope_chg_z, rr_curvature_chg_z, rr_slope_level_z | Sharpe +0.329 | rr_slope_level_z 自己相関0.937 → MAE悪化 |
| 8 (3feat) | attempt7 から rr_slope_level_z を除去 | MAE -0.0203 | Sharpe -0.260 に反転 |
| 9 (5feat) | attempt7 + slope×curvature interaction | MAE -0.0141 | 全指標でattempt 8より劣化 |

**Attempt 10の戦略的判断:**

- 手作りの yield curve特徴量（attempts 7-9）はSharpe/MAEトレードオフを内包
- PCAによるデータ駆動的な分解で、このトレードオフを打破する
- DFII5/7/10/20/30（5本全て）を使用: attempts 7-9は3本(5/10/30)のみ使用
- 新規情報: DFII7（7年）とDFII20（20年）は未使用
- 主成分は数学的に直交 → VIF ≡ 1.0

---

## 1. 設計概要

| 属性 | 値 |
|------|----|
| モデル種別 | 決定的変換（PCA + rolling z-score） |
| データソース | FRED: DFII5, DFII7, DFII10, DFII20, DFII30 |
| 出力列数 | 3列 |
| 学習要否 | 不要（決定的変換） |
| PCAフィット期間 | 2015-01-01〜2021-12-31（全データの〜70%） |
| Kaggle GPU | false |
| Enable Internet | true（FRED API） |
| 推定実行時間 | < 2分 |
| attempts 7-9との差異 | 5本全マチュリティ使用 + PCAによるデータ駆動的直交分解 |

---

## 2. PCA設計

### 2.1 入力データ

DFII5, DFII7, DFII10, DFII20, DFII30の**日次変化量**（.diff()）：

```
dDFII5 = DFII5.diff()   # 5Y TIPS daily change
dDFII7 = DFII7.diff()   # 7Y TIPS daily change
dDFII10 = DFII10.diff() # 10Y TIPS daily change
dDFII20 = DFII20.diff() # 20Y TIPS daily change
dDFII30 = DFII30.diff() # 30Y TIPS daily change
```

### 2.2 PCAフィット

- **フィット期間**: 2015-01-01〜2021-12-31（訓練データのみ。テストへの情報漏洩なし）
- **成分数**: 3（典型的にTIPS曲線の95%以上の分散を説明）
- **正規化**: StandardScaler（各列の平均・標準偏差で正規化してからPCA）

### 2.3 PC成分の経済的解釈

| 主成分 | 典型的形状 | TIPS曲線での意味 | 金への影響 |
|--------|-----------|-----------------|-----------|
| PC1 | 全マチュリティが同方向 | 実金利のパラレルシフト | 負相関（実金利上昇 → 金下落） |
| PC2 | 短期 vs 長期 | 実金利カーブのスロープ変化 | 中程度 |
| PC3 | 中期 vs 両端 | 実金利カーブのバタフライ | 弱い |

### 2.4 出力特徴量

各PCスコアにrolling z-scoreを適用：

```python
def rolling_zscore(series, window=60):
    mu = series.rolling(window, min_periods=window//2).mean()
    sigma = series.rolling(window, min_periods=window//2).std()
    sigma = sigma.replace(0, np.nan)
    return ((series - mu) / sigma).clip(-4, 4)

features['rr_pc1_z'] = rolling_zscore(pc_scores[:, 0])  # 実金利レベル変化
features['rr_pc2_z'] = rolling_zscore(pc_scores[:, 1])  # スロープ変化
features['rr_pc3_z'] = rolling_zscore(pc_scores[:, 2])  # カーブ変化
```

---

## 3. 特徴量特性（期待値）

| 列 | 自己相関期待値 | VIF | 説明 |
|----|--------------|-----|------|
| rr_pc1_z | ~0.03 | 1.0 (定義上) | 日次変化の線形結合 → 低自己相関 |
| rr_pc2_z | ~0.05 | 1.0 (定義上) | 同上 |
| rr_pc3_z | ~0.10 | 1.0 (定義上) | バタフライは若干平均回帰傾向あり |

**VIFが正確に1.0になる理由**: PCA成分は定義上直交（covariance = 0）なため、多重共線性は数学的に0。

---

## 4. データパイプライン

```python
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === 1. FRED API認証 (Kaggle Secrets) ===
from kaggle_secrets import UserSecretsClient
try:
    FRED_API_KEY = UserSecretsClient().get_secret("FRED_API_KEY")
except Exception:
    FRED_API_KEY = os.environ.get('FRED_API_KEY')
from fredapi import Fred
fred = Fred(api_key=FRED_API_KEY)

# === 2. 5本のDFIIシリーズ取得 ===
series_ids = ['DFII5', 'DFII7', 'DFII10', 'DFII20', 'DFII30']
raw = {}
for sid in series_ids:
    raw[sid] = fred.get_series(sid, observation_start='2015-01-01')
df = pd.DataFrame(raw).dropna(how='all')

# === 3. 日次変化量の計算 ===
diffs = df[series_ids].diff().dropna()

# === 4. PCAフィット（訓練期間のみ: 2015-01-01〜2021-12-31）===
train_mask = diffs.index <= '2021-12-31'
train_diffs = diffs[train_mask]

scaler = StandardScaler()
scaler.fit(train_diffs)
scaled_all = scaler.transform(diffs)

pca = PCA(n_components=3)
pca.fit(scaler.transform(train_diffs))

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative: {pca.explained_variance_ratio_.cumsum()}")

# === 5. 全データにPCA変換を適用 ===
pc_scores = pca.transform(scaled_all)
pc_df = pd.DataFrame(pc_scores, index=diffs.index, columns=['pc1', 'pc2', 'pc3'])

# === 6. rolling z-score適用 ===
def rolling_zscore(s, window=60):
    mu = s.rolling(window, min_periods=window//2).mean()
    sigma = s.rolling(window, min_periods=window//2).std()
    sigma = sigma.replace(0, np.nan)
    return ((s - mu) / sigma).clip(-4, 4)

features = pd.DataFrame(index=pc_df.index)
features['rr_pc1_z'] = rolling_zscore(pc_df['pc1'])
features['rr_pc2_z'] = rolling_zscore(pc_df['pc2'])
features['rr_pc3_z'] = rolling_zscore(pc_df['pc3'])

# === 7. target.csvのインデックスに合わせてreindex ===
target = pd.read_csv('/kaggle/input/gold-prediction-submodels/target.csv', index_col=0)
target.index = pd.to_datetime(target.index)
features = features.reindex(target.index)

# === 8. 保存 ===
features.to_csv('/kaggle/working/submodel_output.csv')
print(f"Output shape: {features.shape}")
print(features.describe())
```

---

## 5. ハイパーパラメータ

学習不要（決定的変換）。PCAフィット期間のみ設計上の決定：
- `train_end = '2021-12-31'`（全データの約70%）
- `n_components = 3`（TIPS曲線の構造的次元数）
- `rolling_window = 60`（attempts 7-9と一致）

---

## 6. Gate期待値

### Gate 1
- オーバーフィット比率: 1.0（モデルなし、オーバーフィット不可）
- 全列 非NaN・非定数: YES（全5シリーズは2015年以降フル日次カバレッジ）
- 期待結果: **PASS**

### Gate 2
- MI増加: 3つの直交成分。各PC成分はattempt 8（3特徴量、MI +8-9%）と同等の情報量を期待
  - DFII7とDFII20の追加情報により、attempt 8より若干高いMIが期待される
  - 目標: > 5%（閾値）、期待: 8-12%
- VIF: 定義上1.0（最良可能値）
- 安定性: 日次変化の線形結合 → 高安定性（stability < 0.10期待）
- 期待結果: **PASS**

### Gate 3
- MA期待: attempt 8（-0.0203）と同等。PCAはattempt 8の3特徴量と概念的に類似（高自己相関特徴量なし）
- Sharpe期待: PCAがPC1にレベル変動を適切に捕捉できれば、attempt 7（+0.329）に近づく可能性あり
  - ただし最低限+0.05を満たせば合格
- DA期待: 中立〜若干改善
- 期待結果: **PASS（MAEが最も確実）**

---

## 7. Kernel設定

```json
{
  "id": "bigbigzabuton/gold-real-rate-10",
  "title": "Gold Real Rate Model - Attempt 10",
  "code_file": "train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["bigbigzabuton/gold-prediction-submodels"],
  "competition_sources": [],
  "kernel_sources": []
}
```

---

## 8. リスク評価

1. **PCA成分がattempt 7-9と高相関の場合**: PC1 ≈ rr_level_change_zになる可能性あり。ただし5マチュリティ使用の追加情報でMIは改善するはず。
2. **Gate 2 MI不足**: PC成分はattempt 8の手作り特徴量と概念的に等価。MI < 5%ならattempt 8より劣ることを意味し、受け入れ可能な失敗（attempt 7を最終採用）。
3. **Sharpe再び負値**: 高自己相関特徴量を除くとShapeが-0.26（attempt 8）になった事実がある。PCAも同様の結果になる可能性あり。その場合MAEのみでGate 3判定。

**これはattempt 10（ユーザー指定の上限はattempt 11）。Gate 3失敗の場合はattempt 7（Sharpe +0.329）を最終出力として確定。**
