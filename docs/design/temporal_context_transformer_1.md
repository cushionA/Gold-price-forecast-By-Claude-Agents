# サブモデル設計書: Temporal Context Transformer (Attempt 1)

**作成日**: 2026-02-16
**設計者**: architect (Opus)
**要件出典**: shared/current_task.json (entrance作成)
**リサーチ出典**: docs/research/temporal_context_transformer_1.md (researcher作成)

---

## 0. ファクトチェック結果

### データ検証

| 項目 | 結果 | 詳細 |
|------|------|------|
| base_features.csv | OK | 2523行, 20カラム, 2015-01-30 ~ 2025-02-12 |
| submodel_outputs/*.csv | OK | 9ファイル全存在。全サブモデル出力確認済み |
| 全データ結合後の行数 | CORRECTION | researcher: "2131サンプル" → 実測: **2461行** (全データ結合後) |
| base features列名 | CORRECTION | researcher: "real_rate_change" 等 → 実態: base_features.csvは**レベル値** (real_rate_real_rate, dxy_dxy等)。diff()変換はnotebook内で実施が必要 |
| 14特徴量の構成 | OK | base 5 (diff後) + submodel 9 = 14。researcherの選択は妥当 |

### 手法検証

| 項目 | 結果 | 詳細 |
|------|------|------|
| Masked Reconstruction | OK | 妥当。BERTのMLMに類似した確立手法。金相場予測禁止の原則に準拠 |
| Vanilla Transformer Autoencoder | OK (with corrections) | 妥当だが**パラメータ数に重大な問題**あり |
| ウィンドウ 5-20日 | OK | HMM(20d/60d)との差別化も適切。Optuna探索で妥当 |
| Dropout 0.3-0.5 | OK with adjustment | 妥当だが、極小モデルでは0.3上限推奨（0.5は学習困難にするリスク） |
| 出力1-2カラム | OK | options_market 1カラム成功の教訓に合致 |
| PatchTST不採用 | OK | Channel-independentが目的と矛盾。理由は妥当 |
| Learned PE推奨 | OK | 金融データの非周期性に適合。妥当 |

### 数値検証 -- 重大な誤り

| 項目 | 結果 | 詳細 |
|------|------|------|
| Attention params公式 | **ERROR** | researcher: "4 * d_model^2 * n_heads" → **正解: 4 * d_model^2 + 4 * d_model (n_headsに依存しない)**。PyTorch nn.MultiheadAttention で実測確認済み |
| d=64, h=4の場合 | **ERROR** | researcher: "65,536パラメータ" → **正解: 16,640パラメータ**。ただしd=64, h=8で "131,072(2倍)" も誤り -- 同じ16,640 |
| サンプル数 | **CORRECTION** | researcher: "2131サンプル" → **実測: 2461行** (全データ結合後) |
| d=32-64, L=1-2のモデル全体パラメータ | **CRITICAL CONCERN** | researcherは1-2層/32-64次元を推奨しているが、対称Autoencoderでd=32/L=1でも27,056パラメータ。~1,700 training samples (70%) / 27,056 params = 0.063。**極度の過パラメータ化**。d=64/L=2では203,216パラメータ -- 完全に不可能 |
| PatchTST MSE改善 "21.0%" | UNVERIFIED | arXiv論文の数値だがデータセット・条件依存。金融データでの適用性は不明 |
| Residual Dropout "4 BLEU改善" | CAUTION | NLP(BLEU)の結果であり金融時系列への直接適用は保証されない |

### アーキテクチャ再設計の必要性

researcherの推奨する d_model=32-64, L=1-2 の**対称Transformer Autoencoder**は、~1,700 training samplesに対して明らかに過パラメータである。以下の修正を設計に反映する:

1. **d_model を 16-32 に縮小** (researcherの32-64から下方修正)
2. **非対称アーキテクチャ**: Encoder(Transformer) + Lightweight Decoder(Linear) -- Decoderにはself-attentionを使わない
3. **FFN expansion ratio を 2-4** (標準の4xではなく小さめも許容)
4. **目標パラメータ数: 3,000-10,000** (samples/params ratio >= 0.17)

---

## 1. 概要

### 目的

既存23特徴量のうち文脈依存性の高い14特徴量の過去Nウィンドウを処理し、XGBoostメタモデルが単一時点からは捕捉できない**クロス時間・クロス特徴量の相互作用パターン**を抽出する。

具体的に捕捉する情報:
- 複数特徴量の同期的/非同期的変動パターン（例: VIXスパイク + 金テクニカルregime遷移の同時発生）
- HMMの2-3離散状態では表現できない連続的レジーム遷移の速度・方向
- 5-20日間の時間的文脈（HMMの固定ウィンドウ20d/60dとの差別化）

### 手法とその理由

**Asymmetric Transformer Autoencoder with Masked Reconstruction**

| 設計要素 | 選択 | 理由 |
|---------|------|------|
| Encoder | Transformer (self-attention) | 可変長の時間依存性、クロス特徴量相互作用を捕捉 |
| Decoder | 単層Linear | パラメータ節約。Encoderのbottleneck表現力に集中 |
| 学習目標 | Masked Reconstruction | 過学習抑制、安定した勾配、金相場予測禁止に準拠 |
| 位置符号化 | Learned PE | 金融データの非周期性に適応 |
| 出力 | 1カラム (sigmoid, 0-1正規化) | options_market成功パターンに合致、VIF制約回避 |

### 期待効果

- meta_model DA +0.5% 以上（Gate 3閾値）
- 既存HMMサブモデル群との差別化（連続値、可変ウィンドウ、クロス特徴量）
- MI +5% 以上（Gate 2閾値）

### 過去の教訓の反映

| 教訓 | 反映 |
|------|------|
| real_rate att.3: MI+23.8%だがGate 3 FAIL (月次→日次ミスマッチ) | 全入力が日次データ -- この問題は解消 |
| real_rate att.5: 7カラム出力でGate 3全失敗 | 出力を1カラムに限定 |
| options_market att.2: 1カラムでGate 3 PASS (MAE -0.156) | 1カラムsigmoid出力を採用 |
| HMM成功8サブモデル: 全てregime_probを含む | 出力をsigmoid(0-1)正規化 -- regime-like |
| real_rate att.3 Transformer: overfit 1.28で解決 | 小モデル + 強正則化で再現 |

---

## 2. データ仕様

### メインデータ

全て既存パイプラインから取得。**新規API呼び出し不要**。

#### 入力特徴量 (14次元)

**Base features (5) -- base_features.csv からdiff変換:**

| # | カラム名 (変換後) | 元カラム | 変換 |
|---|------------------|---------|------|
| 1 | real_rate_change | real_rate_real_rate | diff() |
| 2 | dxy_change | dxy_dxy | diff() |
| 3 | vix | vix_vix | そのまま (レベル値、定常) |
| 4 | yield_spread_change | yield_curve_yield_spread | diff() |
| 5 | inflation_exp_change | inflation_expectation_inflation_expectation | diff() |

**Submodel outputs (9) -- data/submodel_outputs/*.csv から:**

| # | カラム名 | ファイル | 値域 | 用途 |
|---|---------|---------|------|------|
| 6 | vix_regime_probability | vix.csv | [0,1] | VIXレジーム状態 |
| 7 | vix_mean_reversion_z | vix.csv | (-inf,+inf) | VIX平均回帰z-score |
| 8 | tech_trend_regime_prob | technical.csv | [0,1] | テクニカルトレンドレジーム |
| 9 | tech_mean_reversion_z | technical.csv | (-inf,+inf) | テクニカル平均回帰z-score |
| 10 | tech_volatility_regime | technical.csv | [0,1] | テクニカルボラティリティレジーム |
| 11 | xasset_regime_prob | cross_asset.csv | [0,1] | クロスアセットレジーム |
| 12 | xasset_divergence | cross_asset.csv | (-inf,+inf) | クロスアセット乖離z-score |
| 13 | etf_regime_prob | etf_flow.csv | [0,1] | ETFフローレジーム |
| 14 | options_risk_regime_prob | options_market.csv | [0,1] | オプション市場リスクレジーム |

### 前処理手順

1. **データ結合**: base_features.csv + 9つのsubmodel output CSVを日付で内部結合
2. **diff変換**: base features のうち4つ (real_rate, dxy, yield_spread, inflation_exp) に diff() 適用
3. **NaN処理**:
   - regime_prob系: 0.5 (最大不確実性)
   - z-score系: 0.0 (平均位置)
   - diff()による先頭NaN: 行削除
4. **正規化**: 各特徴量を train set の mean/std で StandardScaler (z-score正規化)
5. **ウィンドウ化**: 過去W日のスライディングウィンドウ → shape: (N-W+1, W, 14)

### 期待サンプル数

| Split | 全行数 | Window=10適用後 | Window=20適用後 |
|-------|--------|----------------|----------------|
| Train (70%) | 1,722 | 1,713 | 1,703 |
| Val (15%) | 369 | 360 | 350 |
| Test (15%) | 370 | 361 | 351 |
| **合計** | **2,461** | **2,434** | **2,404** |

### データ取得一覧 (builder_data向け)

| データ | 取得元 | パス |
|--------|--------|------|
| Base features | data/processed/base_features.csv | 既存 |
| VIX submodel | data/submodel_outputs/vix.csv | 既存 |
| Technical submodel | data/submodel_outputs/technical.csv | 既存 (tz-aware注意) |
| Cross-asset submodel | data/submodel_outputs/cross_asset.csv | 既存 |
| ETF flow submodel | data/submodel_outputs/etf_flow.csv | 既存 |
| Options market submodel | data/submodel_outputs/options_market.csv | 既存 (tz-aware注意) |
| Target | data/processed/target.csv | 既存 (評価のみに使用) |

**Kaggle環境**: submodel outputs は Kaggle Dataset `gold-prediction-complete` から読み込み。meta_model_6 notebook と同じ読み込みパターンを使用。

---

## 3. モデルアーキテクチャ (PyTorch)

### 設計方針

1. **非対称Autoencoder**: Encoder(Transformer) + Decoder(Linear) -- Decoderにはattentionを使わない
2. **極小モデル**: d_model=16-32, L=1-2, H=2-4 → パラメータ数 3,000-10,000
3. **Bottleneck**: Encoder出力の時間軸mean pool → 線形射影 → 1次元sigmoid
4. **出力形式**: 日次1カラム、0-1正規化、regime-likeスコア

### 入力

- Shape: `(batch_size, window_size, 14)`
- 各次元の意味: [real_rate_change, dxy_change, vix, yield_spread_change, inflation_exp_change, vix_regime_prob, vix_mean_reversion_z, tech_trend_regime_prob, tech_mean_reversion_z, tech_volatility_regime, xasset_regime_prob, xasset_divergence, etf_regime_prob, options_risk_regime_prob]

### 出力

- Shape: `(batch_size, 1)`
- 意味: temporal_context_score -- 0-1の連続値。市場状態の時系列文脈スコア
- 解釈: 高値 = 複数特徴量が方向的に整合している状態、低値 = 特徴量間の不一致/遷移中

### PyTorchクラス設計

```python
class TemporalContextTransformer(nn.Module):
    """
    Asymmetric Transformer Autoencoder for temporal context extraction.

    Architecture:
      Input (batch, seq, 14)
        -> Input Projection (14 -> d_model)
        -> Learned Positional Encoding
        -> TransformerEncoder (L layers, H heads)
        -> Mean Pool over time
        -> Bottleneck Linear (d_model -> 1)
        -> Sigmoid -> context_score (0-1)

      Reconstruction branch (training only):
        -> Bottleneck (1) -> Expand (d_model)
        -> Repeat to seq_len
        -> Output Projection (d_model -> 14)
    """

    def __init__(self, input_dim=14, d_model=24, n_heads=2, n_layers=1,
                 ffn_ratio=2, dropout=0.2, max_seq_len=20):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learned positional encoding
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-Norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # Lightweight Decoder (reconstruction branch)
        self.decoder_expand = nn.Linear(1, d_model)
        self.decoder_output = nn.Linear(d_model, input_dim)

        # Dropout for input
        self.input_dropout = nn.Dropout(dropout)

    def encode(self, x):
        """
        x: (batch, seq_len, input_dim)
        Returns: context_score (batch, 1), encoded (batch, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        h = self.input_proj(x)  # (batch, seq, d_model)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device)
        h = h + self.pos_encoding(positions).unsqueeze(0)

        # Apply input dropout
        h = self.input_dropout(h)

        # Transformer encoder
        encoded = self.encoder(h)  # (batch, seq, d_model)

        # Mean pool over time
        pooled = encoded.mean(dim=1)  # (batch, d_model)

        # Bottleneck -> context score
        context_score = self.bottleneck(pooled)  # (batch, 1)

        return context_score, pooled

    def decode(self, context_score, seq_len):
        """
        Reconstruct from bottleneck for masked reconstruction loss.
        context_score: (batch, 1)
        Returns: (batch, seq_len, input_dim)
        """
        # Expand bottleneck
        expanded = self.decoder_expand(context_score)  # (batch, d_model)

        # Repeat to sequence length
        expanded = expanded.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq, d_model)

        # Output projection
        reconstructed = self.decoder_output(expanded)  # (batch, seq, input_dim)

        return reconstructed

    def forward(self, x, mask_ratio=0.2):
        """
        Forward pass with masked reconstruction.
        x: (batch, seq_len, input_dim)
        Returns: context_score, reconstructed, mask
        """
        batch_size, seq_len, input_dim = x.shape

        # Create random mask (mask time steps, not features)
        mask = torch.rand(batch_size, seq_len, device=x.device) < mask_ratio
        # Ensure at least 1 step is masked and 1 is unmasked
        mask[:, 0] = False  # Keep first step
        if seq_len > 2:
            mask[:, -1] = True   # Always mask last step

        # Apply mask (zero out masked positions)
        x_masked = x.clone()
        x_masked[mask] = 0.0

        # Encode
        context_score, pooled = self.encode(x_masked)

        # Decode (reconstruct)
        reconstructed = self.decode(context_score, seq_len)

        return context_score, reconstructed, mask

    def extract(self, x):
        """
        Extract context score for inference (no masking).
        x: (batch, seq_len, input_dim)
        Returns: context_score (batch, 1)
        """
        context_score, _ = self.encode(x)
        return context_score
```

### パラメータ数の見積もり

| 構成 (d, L, H, FFN) | パラメータ数 | samples/params | 評価 |
|---------------------|------------|----------------|------|
| d=16, L=1, H=2, FFN=2x | 3,104 | 0.55 | 過小? |
| **d=24, L=1, H=2, FFN=2x** | **6,184** | **0.28** | **推奨** |
| d=24, L=1, H=4, FFN=2x | 6,184 | 0.28 | 代替 |
| d=32, L=1, H=2, FFN=2x | 10,288 | 0.17 | 上限 |
| d=24, L=2, H=2, FFN=2x | 11,056 | 0.15 | リスクあり |

**推奨デフォルト**: d_model=24, n_layers=1, n_heads=2, ffn_ratio=2 (~6,200パラメータ)

---

## 4. ハイパーパラメータ

### 固定値

| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| input_dim | 14 | base 5 + submodel 9 |
| output_dim | 1 | options_market成功パターン、VIF制約 |
| output_activation | sigmoid | 0-1正規化、regime-like出力 |
| max_seq_len | 20 | ウィンドウ上限 |
| norm_first | True | Pre-Norm配置、小規模データで学習安定 |
| activation | GELU | Transformer標準、ReLUより滑らか |
| optimizer | AdamW | Weight decay対応 |
| batch_size | 64 | 小規模データでの安定性 |
| mask_strategy | random time-step | 特徴量軸ではなく時間軸でマスク |
| early_stopping_monitor | val_loss | 再構成損失を監視 |
| train/val/test | 70/15/15 | 時系列順、シャッフルなし |
| seed | 42 | 再現性 |

### Optuna探索空間

| パラメータ | 範囲 | 分布 | 根拠 |
|-----------|------|------|------|
| window_size | {5, 10, 15, 20} | categorical | 週次/2週/3週/月次パターン。HMM(20d/60d)との差別化 |
| d_model | {16, 24, 32} | categorical | パラメータ数制御。16=最小、24=推奨、32=上限 |
| n_heads | {2, 4} | categorical | d_modelの約数制約。16の場合は2のみ |
| n_layers | {1, 2} | int | 1=基本、2=表現力追加（過学習リスク増） |
| ffn_ratio | {2, 3} | categorical | FFN拡張率。2=最小、3=標準寄り |
| dropout | [0.1, 0.3] | float (uniform) | 過学習防止。0.3超は学習困難化リスク |
| mask_ratio | [0.15, 0.30] | float (uniform) | マスク比率。低すぎると学習が容易すぎ、高すぎると情報不足 |
| learning_rate | [1e-4, 3e-3] | float (log) | AdamW標準範囲 |
| weight_decay | [0.01, 0.1] | float (log) | L2正則化。小規模データでは強め |
| max_epochs | 200 | 固定 | Early stoppingで制御 |
| patience | {7, 10, 15} | categorical | Early stopping patience |

### d_model と n_heads の制約

```python
# Optunaのconditional sampling
d_model = trial.suggest_categorical('d_model', [16, 24, 32])
if d_model == 16:
    n_heads = trial.suggest_categorical('n_heads_16', [2])  # 16/2=8 (head_dim)
elif d_model == 24:
    n_heads = trial.suggest_categorical('n_heads_24', [2, 4])  # 24/2=12 or 24/4=6
else:  # d_model == 32
    n_heads = trial.suggest_categorical('n_heads_32', [2, 4])  # 32/2=16 or 32/4=8
```

### 探索設定

- **n_trials**: 30
- **timeout**: 1800秒 (30分)
- **sampler**: TPESampler(seed=42)
- **pruner**: MedianPruner(n_startup_trials=5, n_warmup_steps=10)
  - 根拠: 30 trialsのうち最初の5は全て完走。6試行目以降はvalidation lossの中央値でプルーニング

### Optuna目的関数

```python
def objective(trial, train_windows, val_windows, train_labels, val_labels):
    """
    Objective: Minimize masked reconstruction loss on validation set.

    Secondary: Monitor overfitting ratio (train_loss / val_loss).
    """
    # Sample hyperparameters
    config = sample_config(trial)

    # Build model
    model = TemporalContextTransformer(**config)

    # Train with early stopping
    train_loss, val_loss, overfit_ratio = train_model(
        model, train_windows, val_windows, config
    )

    # Pruning based on validation loss
    trial.report(val_loss, step=epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()

    # Primary objective: minimize val_loss
    # Penalize overfitting
    penalty = max(0, overfit_ratio - 1.5) * 0.1

    return val_loss + penalty
```

---

## 5. 学習設定

### 損失関数

```python
def masked_reconstruction_loss(original, reconstructed, mask):
    """
    MSE loss computed ONLY on masked time steps.

    original: (batch, seq_len, 14)
    reconstructed: (batch, seq_len, 14)
    mask: (batch, seq_len) -- True where masked
    """
    # Expand mask to feature dimension
    mask_expanded = mask.unsqueeze(-1).expand_as(original)  # (batch, seq, 14)

    # Compute MSE only on masked positions
    diff = (original - reconstructed) ** 2
    masked_diff = diff[mask_expanded]

    if masked_diff.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)

    return masked_diff.mean()
```

### オプティマイザ

- **AdamW** (weight_decay は Optuna で探索)
- betas: (0.9, 0.999)
- eps: 1e-8

### 学習率スケジューラ

- **CosineAnnealingWarmRestarts** (T_0=20, T_mult=2)
- 理由: Warmup不要の小規模モデル。コサインアニーリングで安定した収束

### 早期停止

- Monitor: val_loss (masked reconstruction loss)
- Patience: Optuna探索 {7, 10, 15}
- Min delta: 1e-5
- Restore best weights: True

### 学習フロー

```
For each Optuna trial:
  1. Build model with sampled hyperparameters
  2. Create windowed datasets (train/val)
  3. StandardScaler fit on train, transform val
  4. Train loop:
     a. Forward: mask 15-30% of time steps
     b. Reconstruct masked positions
     c. Loss: MSE on masked positions only
     d. Backprop + AdamW step
     e. Validate every epoch (same masking)
     f. Early stopping on val_loss
  5. Report final val_loss to Optuna

After Optuna:
  1. Retrain best config on train+val
  2. Extract context scores for ALL dates (train+val+test)
  3. Save to submodel_output.csv
```

---

## 6. Kaggle実行設定

| 設定 | 値 | 根拠 |
|------|-----|------|
| enable_gpu | **true** | PyTorch Transformer。CPU可能だが GPU で安全マージン確保 |
| 推定実行時間 | **15-25分** | 30 trials x 200 epochs (early stop ~50-100) x ~1,700 samples。モデル極小のため高速 |
| 推定メモリ使用量 | **< 2 GB** | バッチ64 x window20 x 14features x float32 = 微小。モデルパラメータ ~10K |
| timeout | **1800秒** | 30分。Optuna timeout と同値 |
| 必要な追加pipパッケージ | **なし** | torch, optuna, pandas, numpy, scikit-learn は全てKaggle標準環境に含まれる |

### Kaggle Dataset依存

- `gold-prediction-complete`: submodel output CSV群 (meta_model notebook と同じ)
- FRED API: **不要** (base_features は base_features.csv から取得可能、または meta_model 同様にAPI取得)

### FRED API Key

- Kaggle Secrets から `FRED_API_KEY` を使用 (meta_model notebook と同じパターン)
- base features 取得に必要 (real_rate, vix, yield_spread, inflation_exp)

---

## 7. 実装指示

### builder_data向け

**新規データ取得は不要。** 全てのデータは既存パイプラインで利用可能。

builder_data の作業:
1. `data/processed/base_features.csv` のスキーマ確認
2. `data/submodel_outputs/` の9ファイルのスキーマ確認
3. 日付結合・NaN処理・diff変換のロジックを検証用スクリプトで確認
4. 結合後のデータを `data/processed/temporal_context_input.csv` として保存 (14特徴量 + target)

**注意**: technical.csv と options_market.csv はタイムゾーン付き日付 (`-04:00`, `-05:00`) を含む。meta_model_6 notebook と同じ `utc=True` + `strftime('%Y-%m-%d')` パターンで正規化すること。

### builder_model向け

#### Notebook構成 (self-contained)

```
Cell 1: Import + setup
Cell 2: Data fetching (API-based, meta_model_6と同パターン)
         - FRED API for base features
         - Kaggle Dataset for submodel outputs
Cell 3: Data preprocessing
         - diff変換 (4 base features)
         - NaN imputation
         - Date alignment (inner join)
         - StandardScaler fit on train
Cell 4: Windowing function
         - create_windows(data, window_size) -> (N-W+1, W, 14)
Cell 5: Model definition (TemporalContextTransformer class)
Cell 6: Training function
         - masked_reconstruction_loss
         - train loop with early stopping
Cell 7: Optuna HPO (30 trials, 1800s timeout)
Cell 8: Final training with best params
         - Retrain on train+val
         - Extract context scores for full dataset
Cell 9: Evaluation metrics
         - Overfit ratio check
         - Output statistics
Cell 10: Save results
         - submodel_output.csv
         - training_result.json
         - model.pt
```

#### PyTorchクラス設計の注意事項

1. **nn.TransformerEncoderLayer の batch_first=True** を必ず指定
2. **norm_first=True** (Pre-Norm) を使用 -- 小規模データでの安定性
3. **Learned PE は nn.Embedding** で実装 -- nn.Parameter でも可だが Embedding がシンプル
4. **mask は時間軸のみ** -- 特徴量軸は全て可視のまま
5. **decode時は context_score (1次元) から展開** -- bottleneck情報の圧縮度を検証する意味がある
6. **extract() メソッド** -- 推論時用。mask なしで encode のみ実行
7. **gradient clipping**: max_norm=1.0 を適用 (Transformer の勾配爆発防止)

#### train.py 固有の注意事項

1. **StandardScaler は train set のみで fit** -- val/test には transform のみ
2. **ウィンドウは各 split 内で生成** -- split 境界をまたがないこと
3. **Kaggle output directory**: `.` (カレントディレクトリ) に保存
4. **submodel_output.csv のフォーマット**:
   ```csv
   date,temporal_context_score
   2015-02-11,0.4523
   2015-02-12,0.5102
   ...
   ```
5. **training_result.json に含める情報**:
   - feature, attempt, timestamp
   - best_params (Optuna結果)
   - val_loss, overfit_ratio
   - output_shape, output_columns
   - output_statistics (mean, std, min, max)
   - model_param_count

---

## 8. リスクと代替案

### リスク1: 過学習 (確率 35-45%)

**症状**: train_loss << val_loss (overfit ratio > 1.5)

**対策** (既に設計に組み込み済み):
- 極小モデル (3,000-10,000 params)
- Dropout 0.1-0.3
- Weight decay 0.01-0.1
- Masked reconstruction (全時間ステップを使わない)
- Early stopping (patience 7-15)
- Gradient clipping (max_norm=1.0)

**代替案** (Attempt 2):
- Data augmentation: ガウスノイズ注入 (std=0.01-0.05)
- Larger mask ratio (0.3-0.5)
- Full reconstruction (masking なし、よりシンプル)

### リスク2: HMMとの高相関 -- Gate 2 VIF > 10 (確率 30-40%)

**症状**: temporal_context_score が regime_prob と高い相関を持つ

**対策**:
- Masking により単純スムージングではない学習タスクを強制
- Sigmoid 出力の値域は [0,1] だが分布が regime_prob と異なる可能性が高い
  (Transformer は連続値遷移、HMM は急峻な状態切替)
- MI が VIF とは別指標なので、VIF > 10 でも MI > 5% なら Gate 2 は条件付き PASS の可能性あり
  (etf_flow: VIF=12.47 でも Gate 3 PASS の前例)

**代替案** (Attempt 2):
- 出力を tanh([-1, 1]) に変更して regime_prob (0-1) との直交性を高める
- Correlation regularization loss: bottleneck と regime_prob の相関をペナルティ化

### リスク3: Gate 3 冗長性 (確率 25-35%)

**症状**: XGBoost に追加しても DA/Sharpe/MAE が改善しない

**対策**:
- 1 カラム出力で XGBoost へのノイズ注入を最小化
- sigmoid(0-1) 正規化で既存 regime_prob と同スケール
- Transformer が捕捉する情報は「時間軸上のクロス特徴量パターン」であり、HMM の「瞬間的レジーム」とは性質が異なる

**代替案** (Attempt 2):
- 出力を 2 カラムに拡張: [context_score, context_volatility]
- TCN Autoencoder に切り替え (より少ないパラメータ)
- iTransformer アーキテクチャ (特徴量次元と時間次元を入れ替え)

### リスク4: Kaggle 30分超過 (確率 5-10%)

**症状**: Optuna が timeout に達し十分な trial が完了しない

**対策**:
- 極小モデルのため 1 trial あたり 30-60秒で完了予定
- 30 trials x 60秒 = 30分で十分収まる
- Early stopping で大半の trial は 50-100 epochs で終了

**代替案**:
- n_trials を 20 に削減
- timeout を 2400秒 (40分) に延長 (Kaggle の 9時間制限には十分余裕)

---

## 付録: researcher レポートの主要な誤りまとめ

| 項目 | researcher記載 | 正確な値 | 影響度 |
|------|--------------|---------|--------|
| MHA params公式 | 4 * d^2 * n_heads | 4 * d^2 + 4 * d (n_heads非依存) | 高: パラメータ推定に重大な誤り |
| d=64/h=4のMHAパラメータ | 65,536 | 16,640 | 高: 約4倍の過大評価 |
| d=64/h=8のMHAパラメータ | 131,072 | 16,640 | 高: 約8倍の過大評価 |
| サンプル数 | 2,131 | 2,461 | 中: 14%の過小評価 |
| 対称AE推奨 (d=32-64) | 27K-203K params | 適切: 3K-10K params | 高: 過パラメータ化の原因 |
| Residual Dropout効果 | "4 BLEU改善" | NLP固有指標、金融で未検証 | 低: 参考情報 |

---

**設計書終了**
