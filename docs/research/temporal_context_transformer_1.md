# リサーチレポート: Temporal Context Transformer (Attempt 1)

**作成日**: 2026-02-16
**調査者**: researcher (Sonnet 4.5)
**要件出典**: shared/current_task.json (entrance作成)

---

## エグゼクティブサマリー

本レポートは、Temporal Context Transformerサブモデル開発のための8つの重要な調査質問に回答する。調査結果に基づき、小規模金融データ（~2000サンプル）での安定した学習を実現する**軽量Transformer Autoencoder**の設計を推奨する。

**主要な推奨事項**:
1. **アーキテクチャ**: Vanilla Transformer Autoencoder（PatchTSTより単純）、1-2層、32-64次元
2. **学習目標**: Masked Reconstruction（安定性と過学習防止）
3. **ウィンドウサイズ**: 10-20日（Optuna探索: 5, 10, 15, 20日）
4. **位置エンコーディング**: Learned Positional Encoding（金融データの不規則性に対応）
5. **正則化**: Dropout 0.3-0.5、Weight Decay 0.05-0.1、Early Stopping
6. **Attention Heads**: 2-4個（パラメータ効率重視）
7. **出力次元**: 1-2カラム（VIF制約、冗長性回避）

---

## 調査事項

以下の8つの調査質問に回答する:

1. 時系列Transformer Autoencoderのアーキテクチャ選択肢
2. 学習安定化手法（小規模データでの過学習防止）
3. ウィンドウサイズの理論的根拠
4. PatchTST等の最新アーキテクチャの適用可能性
5. 学習目標の比較（masked prediction vs reconstruction vs contrastive）
6. 位置エンコーディングの選択
7. Multi-head Attention数の最適化
8. 金融時系列でのTransformer成功事例

---

## 回答1: 時系列Transformer Autoencoderのアーキテクチャ選択肢

### 調査結果

#### 1.1 Vanilla Transformer Autoencoder

**構成**: Encoder（self-attention + FFN層）→ Bottleneck → Decoder（reconstruction）

**利点**:
- シンプルで実装が容易
- 小規模データでも安定（パラメータ数を制御可能）
- PyTorchで標準的な`nn.Transformer`を利用可能
- 再構成損失によりボトルネック層が自然に低次元文脈表現を生成

**欠点**:
- Point-wise attentionは時系列の局所的な意味情報を捉えにくい
- 長いシーケンスでメモリ使用量が二乗的に増加

**小規模データでの適性**: ★★★★★（最適）

#### 1.2 PatchTST (2023 ICLR)

**構成**: 時系列を固定長パッチに分割 → 各パッチをトークン化 → Channel-independent Transformer

**利点**:
- パッチ化により局所的意味情報を保持
- メモリ使用量が大幅削減（同じlookback windowでattention mapsが二乗的に減少）
- SOTA: Informerより**MSE 21.0%削減、MAE 16.7%削減**
- Self-supervised learningとの相性が良い

**欠点**:
- 追加の複雑性（パッチ長、ストライドのハイパーパラメータ）
- Channel-independentアプローチは特徴量間の相互作用を無視
- 小規模データでのベンチマークが不足

**小規模データでの適性**: ★★★☆☆（要検証）

#### 1.3 Informer (2021)

**構成**: ProbSparse attention（注意機構のスパース近似）

**利点**:
- 長期予測に特化
- 計算効率が高い

**欠点**:
- PatchTSTに性能で劣る（Weather/Traffic/Electricityデータセットで顕著）
- Point-wise tokenizationの限界
- 金融データの不規則性には不向き

**小規模データでの適性**: ★★☆☆☆（推奨しない）

#### 1.4 Supervised Autoencoder MLP (2024)

**構成**: Autoencoder + Supervised MLP（encoded featuresと元の入力を連結して予測）

**利点**:
- S&P 500、EUR/USD、BTC/USDで投資戦略の改善を実証
- Noise augmentation + bottleneck sizeの調整で過学習抑制
- 金融データのノイズに強い

**欠点**:
- Supervisedアプローチは「サブモデルは金相場を予測しない」原則に反する
- 教師ラベル（金リターン）を使用するため本プロジェクトでは採用不可

**小規模データでの適性**: N/A（原則に反するため不採用）

### 推奨アーキテクチャ

**第1優先**: **Vanilla Transformer Autoencoder**
- 理由: シンプル、安定、小規模データで実績あり、PyTorch標準実装
- 設計指針: 1-2層、32-64次元、2-4 attention heads

**第2優先**: **PatchTST（簡略版）**
- 理由: パッチ化による局所的意味保持、メモリ効率
- 条件: attempt 1でVanilla版が失敗した場合のフォールバック
- 注意: 特徴量間相互作用の喪失を評価する必要あり

### データソース

- [Supervised Autoencoder MLP for Financial Time Series Forecasting (arXiv 2024)](https://arxiv.org/abs/2404.01866)
- [PatchTST: A Time Series is Worth 64 Words (ICLR 2023)](https://arxiv.org/abs/2211.14730)
- [Deep Learning for Financial Forecasting: A Review (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S1059056025008822)

---

## 回答2: 学習安定化手法（小規模データでの過学習防止）

### 調査結果

#### 2.1 Dropout戦略

**基本原理**: 訓練時にニューロンをランダムに無効化（0に設定）し、特定のニューロン間の依存関係を防ぐ

**小規模データでの推奨設定**:
- **Dropout rate: 0.3-0.5**（大規模データでは0.1-0.2）
- 適用箇所: Attention層 + FFN層 + 残差接続後

**2024年の新手法: Residual Dropout**
- 残差接続にDropoutを追加
- 小規模データセットで**4 BLEU points以上の改善**を実証
- 過学習を効果的に抑制

**注意事項**:
- Dropout > 0.5は学習を阻害するため避ける
- BatchNormとの併用は分布を乱すため非推奨（TransformerではLayerNormを使用）

#### 2.2 Layer Normalization

**推奨理由**:
- TransformerではBatchNormよりLayerNormが標準
- バッチサイズに依存しないため小規模データで安定
- Dropoutとの相性が良い

**配置**:
- Pre-Norm（LayerNorm → Attention/FFN）が学習安定性で優位
- Post-Norm（Attention/FFN → LayerNorm）は大規模データ向け

#### 2.3 Weight Decay（L2正則化）

**推奨設定**:
- **Weight Decay: 0.05-0.1**（小規模データ）
- AdamWオプティマイザとの併用が標準

#### 2.4 Early Stopping

**設定**:
- Validation lossを監視（連続5-10 epochsで改善なしで停止）
- 過学習の兆候を早期に検出

#### 2.5 その他の手法

**Label Smoothing**:
- 過信した予測を防止
- Classification taskでは有効だがRegressionでは効果限定的

**Data Augmentation**:
- Noise injection（ガウスノイズ、0.01-0.05倍の標準偏差）
- Time warping（時間軸の伸縮）
- 金融データでは慎重に適用（過度の変形は現実性を損なう）

**Transfer Learning**:
- 大規模時系列コーパス（Weather/Electricityデータ）で事前学習
- 金融データでFine-tuning
- **注意**: 本プロジェクトでは時間的制約からattempt 1では非推奨（attempt 2以降の改善策として保留）

### 推奨設定（優先度順）

1. **Dropout 0.3-0.5**（Attention + FFN + Residual）
2. **Weight Decay 0.05-0.1**（AdamW）
3. **Early Stopping**（patience=5-10 epochs）
4. **LayerNorm（Pre-Norm配置）**
5. **Residual Dropout**（2024年手法、試験的導入）

### データソース

- [Advanced Regularization Protocols for Transformers (Medium 2024)](https://medium.com/@hassanbinabid/the-art-and-science-of-hyperparameter-optimization-in-llm-fine-tuning-f95bc6e9a80b)
- [Residual Dropout: A Simple Approach (ACL 2024)](https://aclanthology.org/2024.sigul-1.35.pdf)
- [Transformer Regularization Techniques (APXML 2024)](https://apxml.com/courses/foundations-transformers-architecture/chapter-7-implementation-details-optimization/transformer-regularization)

---

## 回答3: ウィンドウサイズの理論的根拠

### 調査結果

#### 3.1 標準Transformerの制約

**問題点**:
- 長いlookback windowで性能劣化とメモリ爆発
- 冗長な情報が学習を阻害
- 時間的ノイズへの過学習傾向
- 二次的計算複雑性

**金融データでの特殊性**:
- 周期性・季節性の欠如により明確なパターンが観察困難
- ボラティリティが高く、過去の関連性が急速に減衰
- 直近データの重要性が高い（Trend layerで検証済み）

#### 3.2 iTransformer（ICLR 2024 Spotlight）の発見

**画期的発見**:
- "Inverted" Transformer（特徴量次元とシーケンス次元を入れ替え）
- **lookback windowの長さに比例して性能が改善**
- 標準Transformerの制約を克服

**本プロジェクトへの適用性**:
- 入力14特徴量 × 10-20日ウィンドウ = 140-280次元
- iTransformerでは特徴量を "tokens"、時間を "features" として扱う
- **要検証**: 14特徴量は少なすぎる可能性（iTransformerは通常数百特徴量を想定）

#### 3.3 動的ウィンドウサイジング（2024年トレンド）

**アプローチ**:
- ボラティリティに基づいてウィンドウ長をリアルタイム調整
- 高ボラティリティ期間 → 短いウィンドウ（直近5-10日に集中）
- 低ボラティリティ期間 → 長いウィンドウ（20-30日の文脈）

**本プロジェクトへの適用性**:
- attempt 1では複雑すぎる（実装コスト高）
- attempt 2以降の改善策として保留
- 現時点では固定ウィンドウ + Optuna探索を推奨

#### 3.4 金融日次データにおける経験則

**既存研究からの示唆**:
- **5日**: 1週間（週次パターン、短期モメンタム）
- **10日**: 2週間（短期トレンド反転の検出）
- **20日**: 1ヶ月（技術分析の標準期間、ボリンジャーバンド等）
- **60日**: 3ヶ月（中期トレンド）

**本プロジェクトの制約**:
- 既存HMMサブモデルは20日・60日の固定ウィンドウを使用
- Transformer文脈サブモデルは**これらより短い5-20日が適切**
- 理由: 長期文脈はHMM regime_probが既に捕捉済み、冗長性回避

#### 3.5 Hybrid Transformer-Mamba（2024年）

**特徴**:
- Information Bottleneck Filter（IBF）で冗長な部分列を除去
- Mambaの線形複雑性 + Transformerの短期モデリング
- 長いシーケンスの制約を克服

**本プロジェクトへの適用性**:
- 複雑すぎる（Mambaの追加実装が必要）
- attempt 1では非推奨

### 推奨ウィンドウサイズ

**Optuna探索範囲**: [5, 10, 15, 20]日

**理論的根拠**:
1. **5日**: 週次パターン、最小限の文脈（過学習リスク最小）
2. **10日**: 2週間、短期トレンド反転
3. **15日**: 3週間、バランス型（推奨初期値）
4. **20日**: 1ヶ月、HMMとの重複リスクあり（上限）

**20日超を推奨しない理由**:
- HMM regime_probが20日・60日固定ウィンドウを使用
- Gate 2でVIF > 10のリスク
- メモリと計算コストの増加
- 金融データの過学習リスク

### データソース

- [iTransformer: Inverted Transformers Are Effective (ICLR 2024)](https://arxiv.org/html/2310.06625v4)
- [Dynamic Window Sizing for Time-Series Forecasting (Preprints.org 2025)](https://www.preprints.org/manuscript/202501.1424)
- [Modality-aware Transformer for Financial Forecasting (arXiv 2023)](https://arxiv.org/html/2310.01232v2)

---

## 回答4: PatchTST等の最新アーキテクチャの適用可能性

### 調査結果

#### 4.1 PatchTST（ICLR 2023）の詳細評価

**アーキテクチャ**:
- 時系列を固定長パッチ（P=16, Stride=8）に分割
- 各パッチを1トークンとして扱う
- Channel-independent: 各特徴量を独立に処理

**性能**:
- Informerと比較: MSE **21.0%削減**、MAE **16.7%削減**
- Traffic dataset（336-step horizon）: MSE **0.408 vs Informer 0.776**
- Weather/Electricity/Traffic全てでSOTA

**Self-supervised learningとの統合**:
- Masked pre-training → Fine-tuning
- 大規模データセットでの事前学習が可能
- Transfer learningで少数サンプルでも高精度

**本プロジェクトへの適用可能性の評価**:

**利点**:
- メモリ効率（二乗的削減）
- 局所的意味情報の保持（パッチ内の時間的関係）
- Self-supervised learningとの相性

**欠点**:
- **Channel-independentアプローチは致命的**
  - 本プロジェクトの目的は「複数特徴量の時系列相互作用を捉える」こと
  - PatchTSTは各特徴量を完全に独立に処理 → 目的と矛盾
- 追加のハイパーパラメータ（パッチ長、ストライド）
- 小規模データ（2131サンプル）での実績が不足
- 実装の複雑性

**結論**: **Attempt 1では採用しない。Attempt 2以降の改善策として保留。**

**代替案**: **CT-PatchTST（2025）**を検討
- Channel-Time Patch Transformer
- Channel attention → Time attention
- 特徴量間の相互作用を明示的にモデリング
- ただし実装複雑性が高いため、attempt 1では保留

#### 4.2 その他の最新アーキテクチャ

**Temporal Convolutional Network（TCN）**:
- 畳み込みベースの時系列モデル
- パラメータ効率が高い
- 小規模データで安定

**比較（Transformer vs TCN vs LSTM Autoencoder）**:
- **Transformer**: 長期依存性、self-attention、柔軟性
- **TCN**: パラメータ効率、受容野の制御、安定性
- **LSTM Autoencoder**: 実績あり、シンプル、逐次処理

**小規模金融データでの推奨順位**:
1. **Vanilla Transformer Autoencoder**（最もバランスが良い）
2. **TCN Autoencoder**（フォールバック）
3. **LSTM Autoencoder**（baseline比較用）

#### 4.3 Ensemble of Temporal Transformers（2024）

**アプローチ**:
- 複数のTemporal Transformerをスライディングウィンドウで学習
- アンサンブルで予測
- 金融時系列の条件付き異分散性に対応

**本プロジェクトへの適用性**:
- 単一サブモデル出力が目標（1-3カラム）
- Ensembleは複雑すぎる
- attempt 1では非推奨

### 推奨アーキテクチャ（再確認）

**Attempt 1**: **Vanilla Transformer Autoencoder**
- 理由: シンプル、特徴量間相互作用を保持、実装が容易

**Attempt 2（改善策）**:
- **CT-PatchTST**（channel-time統合）
- **iTransformer**（inverted architecture）
- **Hybrid Transformer-TCN**

### データソース

- [PatchTST: A Breakthrough in Time Series Forecasting (DataScienceWithMarco 2024)](https://www.datasciencewithmarco.com/blog/patchtst-a-breakthrough-in-time-series-forecasting)
- [CT-PatchTST: Channel-Time Patch Transformer (arXiv 2025)](https://arxiv.org/html/2501.08620v3)
- [Ensemble of Temporal Transformers for Financial Time Series (Springer 2024)](https://link.springer.com/article/10.1007/s10844-024-00851-2)

---

## 回答5: 学習目標の比較（masked prediction vs reconstruction vs contrastive）

### 調査結果

#### 5.1 TempSSL（2024）: Temporal Masked Modeling + Temporal Contrastive Learning

**アプローチ**:
1. **Temporal Masked Modeling (TMM)**:
   - 過去（context）から未来（target）を再構成
   - Cross-attentionで時間的依存性を捕捉
2. **Temporal Contrastive Learning (TCL)**:
   - Context-targetを正例ペアとして扱う
   - 分布シフトを緩和、判別表現を強化

**性能**:
- 7つの主流データセットで**1.92-78.12%の改善**
- Self-supervised + End-to-end学習を上回る

**本プロジェクトへの適用性**:
- **TMM**: 「次時点の特徴量ベクトル予測」に対応 → 採用可能
- **TCL**: 追加の複雑性（positive/negative sampling）
- attempt 1ではTMM単独を推奨、TCLはattempt 2以降

#### 5.2 TSDE（KDD 2024）: Diffusion-based Reconstruction

**アプローチ**:
- Imputation-Interpolation-Forecasting (IIF) maskでデータ分割
- Diffusion processでmasked部分を予測
- Dual-orthogonal Transformer encoder

**性能**:
- Imputation, Interpolation, Forecasting, Anomaly Detection全てで優位
- 従来のReconstruction/Adversarial/Contrastive手法を上回る

**本プロジェクトへの適用性**:
- **Diffusion processは複雑**（実装コスト高、Kaggle 30分制約に抵触のリスク）
- attempt 1では非推奨
- 参考: IIF maskingの概念は有用（mask strategy設計に活用）

#### 5.3 DECL（IJCAI 2024）: Denoising-Aware Contrastive Learning

**アプローチ**:
- Denoiserで正例サンプル生成
- Noiseを加えて負例サンプル生成
- 金融データのノイズに強い

**本プロジェクトへの適用性**:
- 金融データは本質的にノイズが多い → 有用
- ただし、contrastive lossの実装複雑性
- attempt 1では保留、attempt 2以降で検討

#### 5.4 SSL4TS Taxonomy（IEEE TPAMI 2024）

**Self-supervised learning for時系列の分類**:
1. **Generative-based**:
   - Autoregressive forecasting
   - Autoencoder reconstruction
   - Diffusion generation
2. **Contrastive-based**:
   - Augmentation-based
   - Temporal-based
3. **Adversarial-based**:
   - GAN-based

**推奨**: **Autoencoder reconstruction**
- 理由: 安定、シンプル、ボトルネック層が自然に低次元表現を生成

### 学習目標の比較表

| 学習目標 | 安定性 | 実装複雑性 | 金融データ適性 | 本プロジェクト適性 |
|---------|-------|-----------|-------------|-----------------|
| **Masked Reconstruction** | ★★★★★ | ★★★★☆ | ★★★★☆ | **★★★★★（最推奨）** |
| Full Reconstruction | ★★★★★ | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| Masked Prediction (next-step) | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| Contrastive Learning | ★★★☆☆ | ★★☆☆☆ | ★★★★★ | ★★★☆☆（attempt 2以降） |
| Diffusion-based | ★★★★☆ | ★☆☆☆☆ | ★★★★★ | ★★☆☆☆（複雑すぎ） |

### 推奨学習目標（Attempt 1）

**第1優先**: **Masked Reconstruction**
- ランダムに時間ステップの一部をマスク（15-30%）
- Encoderで文脈を圧縮 → Bottleneck → Decoderでマスク部分を再構成
- 損失: MSE(original, reconstructed) on masked positions

**利点**:
- 安定した学習（再構成損失は勾配が安定）
- ボトルネック層が自然に文脈特徴量を生成
- 「金相場を予測しない」原則に準拠
- BERTのmasked language modelingと同様の手法（実績あり）

**実装**:
```python
# Pseudo-code
def masked_reconstruction_loss(x, mask_ratio=0.2):
    # x: [batch, seq_len, features]
    mask = random_mask(seq_len, mask_ratio)  # 20% of time steps
    x_masked = x * (1 - mask)

    encoded = encoder(x_masked)  # → bottleneck
    decoded = decoder(encoded)

    loss = mse_loss(decoded[mask], x[mask])  # only on masked positions
    return loss, encoded  # encoded = context features for meta-model
```

**第2優先**: **Full Reconstruction**（フォールバック）
- 全時間ステップを再構成
- Maskingなし、よりシンプル
- 過学習リスクがやや高い

### データソース

- [TempSSL: Rethinking Self-Supervised Learning for Time Series (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124012863)
- [TSDE: Diffusion Process for Time Series Representation (KDD 2024)](https://dl.acm.org/doi/10.1145/3637528.3671673)
- [SSL4TS: Self-Supervised Learning for Time Series Analysis (IEEE TPAMI 2024)](https://arxiv.org/pdf/2306.10125)

---

## 回答6: 位置エンコーディングの選択

### 調査結果

#### 6.1 主要な位置エンコーディング手法

**1. Absolute Positional Encoding (APE)**
- 固定sinusoidal（Vaswani et al., 2017）
- 学習可能なembedding

**2. Rotary Positional Encoding (RoPE)**
- Query/Key vectorsを位置依存行列で回転
- 相対位置のみに依存（内積が相対変位のみの関数）
- 乗法的バイアス（multiplicative bias）

**3. ALiBi (Attention with Linear Biases)**
- Embeddingを追加せず、attention scoreにバイアス
- 距離に基づくペナルティ（近い → 低ペナルティ、遠い → 高ペナルティ）
- Recency bias（直近を重視）

#### 6.2 RoPE vs ALiBi

**RoPE**:
- 乗法的結合 → スペクトル収縮 → 最適化安定性向上
- Content-relative couplingで優れた汎化性能
- Early-layer "single-head deposit"（浅い層の1ヘッドに位置情報集中）
- DeepSeek-V3等の最新LLMで採用

**ALiBi**:
- 学習速度が**11%高速**、メモリ使用量**11%削減**
- Sequence長の外挿性能が優れる（訓練1024 → 推論2048で同等perplexity）
- RoPEと同等の性能
- 分散的な頭部特化（diffuse head specialization）

**比較表**:

| 特徴 | RoPE | ALiBi | APE |
|-----|------|-------|-----|
| 実装の複雑性 | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| 外挿性能 | ★★★★☆ | ★★★★★ | ★★☆☆☆ |
| 学習速度 | ★★★☆☆ | ★★★★★ | ★★★★☆ |
| 最適化安定性 | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| 時系列データ適性 | ★★★★☆ | ★★★★★ | ★★★☆☆ |

#### 6.3 時系列データにおける位置エンコーディングの特殊性

**arXiv:2502.12370 (2025): Positional Encoding in Transformer-Based Time Series Models**

**調査対象手法**:
- Sinusoidal PE
- Learned PE
- tAPE (time-aware PE)
- RPE (Relative PE)
- eRPE (enhanced RPE)

**重要な知見**:
- 時系列分類・予測タスクでは**Learned PE**が多くのケースで優位
- 固定sinusoidalは周期性を仮定するが、金融データは非周期的
- Relative PEは時間的距離の相対的重要性を捉える

#### 6.4 金融時系列における推奨

**Foumani et al. (2024): Improving Position Encoding for Multivariate Time Series**
- 多変量時系列分類タスクで位置エンコーディングの改善を実証
- 金融データの不規則性に対応

**推奨理由**:
- 金融データは**周期性・季節性が欠如**
- 固定sinusoidalは不適切（周期を仮定）
- **Learned PE**が最も柔軟で金融データに適応可能

### 推奨位置エンコーディング

**第1優先**: **Learned Positional Encoding**
- 理由: 金融データの不規則性に適応、実装が簡単、時系列分類で実績
- 実装: `nn.Embedding(max_len, d_model)`

**第2優先**: **ALiBi**
- 理由: Recency bias（直近重視）が金融データに適合、学習速度向上、外挿性能
- 実装複雑性: やや高い（attention scoreにバイアス追加）

**第3優先**: **Fixed Sinusoidal**（baseline比較用）
- 理由: PyTorch標準実装、計算コストゼロ
- 注意: 金融データの非周期性により性能は限定的

**RoPEを推奨しない理由**:
- 実装複雑性が高い
- 金融データでのベンチマークが不足
- ALiBiと同等性能だが学習速度で劣る

### データソース

- [Positional Embeddings in Transformers: RoPE & ALiBi (Towards Data Science 2024)](https://towardsdatascience.com/positional-embeddings-in-transformers-a-math-guide-to-rope-alibi/)
- [Positional Encoding in Transformer-Based Time Series Models: A Survey (arXiv 2025)](https://www.arxiv.org/pdf/2502.12370)
- [Improving Position Encoding for Multivariate Time Series (Springer 2024)](https://link.springer.com/article/10.1007/s10618-023-00948-2)

---

## 回答7: Multi-head Attention数の最適化

### 調査結果

#### 7.1 金融時系列における実証研究

**株価予測タスク（2024）**:
- **4 attention heads**が効果的
- Learning rate 0.001と併用で収束安定性向上

**暗号通貨予測（Temporal Fusion Transformer + ADE）**:
- Adaptive Differential Evolution (ADE)でハイパーパラメータ最適化
- Attention head数は**データセット依存**（自動調整が最適）

**Modality-aware Transformer (MAT, 2023)**:
- Intra-modal, Inter-modal, Target-modal MHA
- **4-8 heads**が一般的
- Feature-level attentionでモダリティごとに最適化

**LiteFormer（2023, 軽量Transformer）**:
- 低レイテンシ推論（38ms）
- **Head数を削減**してパラメータ効率向上
- RMSE/MAE **3.45-9.09%改善**（vanilla Transformerより）

#### 7.2 理論的考察

**Multi-head Attentionの役割**:
- 各ヘッドは異なるパターンに注目
- 複数ヘッドで多様な時間的依存性を捕捉
- ただし、ヘッド数が多すぎるとパラメータ爆発 → 過学習

**パラメータ数への影響**:
- Attention: `4 * d_model^2 * n_heads`（Q, K, V, O projections）
- d_model=64, n_heads=4の場合: 65,536パラメータ
- d_model=64, n_heads=8の場合: 131,072パラメータ（2倍）

**小規模データ（2131サンプル）での制約**:
- パラメータ数を抑える必要あり
- **2-4 heads**が妥当

#### 7.3 既存研究のパターン

**一般的な設定**:
- **小規模モデル**: 2-4 heads
- **中規模モデル**: 4-8 heads
- **大規模モデル**: 8-16 heads

**金融時系列の実績**:
- 大半の論文で**4 heads**が採用されている
- **8 heads**は大規模データセット向け

### 推奨Attention Head数

**Optuna探索範囲**: [2, 4, 8]

**理論的根拠**:
1. **2 heads**: 最小構成、パラメータ効率最優先
2. **4 heads**: バランス型、多くの研究で採用（推奨初期値）
3. **8 heads**: 表現力重視、過学習リスクあり（上限）

**推奨初期値**: **4 heads**
- 理由: 金融時系列の実証研究で最も頻繁に採用、パラメータ効率と表現力のバランス

**d_model との関係**:
- d_model = 64の場合 → 4 heads（各head dim=16）
- d_model = 32の場合 → 2 heads（各head dim=16）
- **Head dimension = 16-32が推奨**（小さすぎると表現力不足、大きすぎると過学習）

### データソース

- [Novel Transformer-Based Dual Attention for Financial Time Series (Springer 2025)](https://link.springer.com/article/10.1007/s44443-025-00045-y)
- [Enhanced Transformer for Online Stock Price Prediction (PLOS One 2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0316955)
- [LiteFormer: Lightweight Transformer for Financial Forecasting (SSRN 2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729648)

---

## 回答8: 金融時系列でのTransformer成功事例（2020-2024）

### 調査結果

#### 8.1 株式市場予測

**Transformer + Time2Vec（NASDAQ, S&P 500, Exxon Mobil）**:
- Time2Vecで時間特徴量を強化
- 極端な市場条件でも精度向上、誤差削減
- LSTM/RNN比で顕著な改善

**Hybrid LSTM + Transformer Sentiment（Apple Inc.）**:
- LSTM: 時系列モデリング（3年間の日次データ）
- Transformer (FinBERT): ニュースセンチメント抽出
- **MSE大幅削減、方向精度向上**
- 特に決算発表・製品発表前後で効果的

**Autoformer（Apple, Amazon, 2018-2024データ）**:
- **Transformerが最も高精度**（他モデルと比較）
- 長期予測で優位

**STL-Transformer Hybrid（米国株式市場, 2020-2024）**:
- Seasonal-Trend decomposition using Loess (STL) + Transformer
- Microsoft, Googleで頑健性を実証

#### 8.2 外国為替（Forex）市場

**Transformer + Time Embeddings（FX Spot Rates）**:
- **多変量・クロスセクショナル入力**が必須
- LSTM比で優位
- ただし単変量では優位性が限定的

**高頻度Forex取引（6通貨ペア, 60-720分足）**:
- Transformer > ResNet-LSTM（特に長時間足480-720分）
- **取引コストが短時間足（60分）で大きく影響**
- 現実的なバックテストの重要性を強調

**EUR/USD & GBP/USD（分足取引, 2023年）**:
- Transformer Encoder + EMA統合
- Cross-entropy loss < 0.2（高精度）

**主要通貨ペア（EUR/USD, GBP/USD, USD/CAD, USD/CHF）**:
- Hybrid model > GRU, LSTM, SMA
- 10分足でMSE/RMSE/MAE改善

#### 8.3 商品・マルチアセット予測

**STL-Transformer（USD/JPY, EUR/USD, 2020-2024）**:
- 外国為替と株式の両方で適用
- 連邦準備制度理事会（FRB）会合期間をカバー
- Microsoft, Googleでの予備テスト実施

**今後の課題**:
- **商品（Commodity）、債券（Fixed Income）での体系的検証が不足**
- 研究は株式・Forexに集中

#### 8.4 重要な共通知見

**1. Transformerの優位性**:
- LSTM/RNN/ARIMAを一貫して上回る（2020-2024全体で）
- 特に**長期依存性**が重要なタスクで顕著

**2. マルチモーダル統合の重要性**:
- ニュースセンチメント（FinBERT）との統合でMSE大幅削減
- 価格データ単独よりマルチモーダルが優位

**3. 取引コストの現実性**:
- 短時間足（60分以下）では取引コストが利益を相殺
- **現実的なバックテスト**が不可欠

**4. 多変量入力の重要性**:
- Forexでは多変量・クロスセクショナル入力が必須
- 単変量ではTransformerの優位性が限定的

**5. 複雑性とトレードオフ**:
- ハイブリッド手法（LSTM+Transformer, STL+Transformer）が高性能
- ただし実装複雑性が高い

#### 8.5 本プロジェクトへの示唆

**成功パターン**:
- **多変量入力**: 本プロジェクトは14特徴量（5 base + 9 submodel outputs）→ ✅
- **Self-attention**: クロス特徴量の時間的相互作用を捕捉 → ✅
- **長期依存性**: 5-20日ウィンドウで過去文脈をモデリング → ✅

**注意点**:
- **過学習リスク**: 金融データは本質的にノイズが多い（COVID-19, 地政学的リスク等）
- **取引コスト**: meta_modelでの最終評価ではTx cost 5bpsを考慮済み
- **多変量の重要性**: 単一特徴量ではTransformerの優位性が限定的 → 14特徴量は妥当

**差別化要因**:
- 本プロジェクトは**予測タスクではなく文脈抽出タスク**
- 金リターンを直接予測しない（自己教師あり学習）
- これにより、既存の金融Transformer研究（全て教師あり予測）と異なるアプローチ

### データソース

- [Transformer Encoder and Multi-features Time2Vec (EUSIPCO 2025)](https://eusipco2025.org/wp-content/uploads/pdfs/0001682.pdf)
- [LSTM + Transformer Sentiment for Apple Inc. (ANSER Press 2025)](https://www.anserpress.org/journal/jea/4/3/109/pdf)
- [Fx-spot Predictions with SOTA Transformer (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S0957417424004032)
- [STL-Transformer Hybrid Model (ACM 2024)](https://dl.acm.org/doi/pdf/10.1145/3760622.3760649)
- [Deep Learning for Financial Forecasting: A Review (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S1059056025008822)

---

## 推奨手法（優先度順）

### 1. **Vanilla Transformer Autoencoder with Masked Reconstruction**（最優先）

**概要**:
- 1-2層のEncoder-Bottleneck-Decoderアーキテクチャ
- 時間ステップの20%をランダムにマスク
- Masked positionsのみでMSE損失計算
- Bottleneck層（1-2次元）を文脈特徴量としてmeta_modelに提供

**期待効果**:
- 安定した学習（再構成損失は勾配が安定）
- ボトルネック層が自然に低次元文脈表現を生成
- 既存HMMサブモデルとの差別化（固定ウィンドウ・固定状態数の制約を超える）

**実装難易度**: ★★★☆☆（中程度）
- PyTorch標準`nn.Transformer`を活用
- Masking strategyの実装が必要
- Optuna統合が必要

**必要データ**:
- base_features.csv（5カラム: real_rate_change, dxy_change, vix, yield_spread_change, inflation_exp_change）
- submodel_outputs/*.csv（9カラム選択: vix_regime_probability, vix_mean_reversion_z, tech_trend_regime_prob, tech_mean_reversion_z, tech_volatility_regime, xasset_regime_prob, xasset_divergence, etf_regime_prob, options_risk_regime_prob）
- 合計14特徴量 × 過去N日ウィンドウ（N=5-20, Optuna探索）

### 2. **Lightweight Transformer with Residual Dropout**（第2優先）

**概要**:
- 残差接続にDropoutを追加（2024年新手法）
- Full reconstruction（maskingなし、よりシンプル）
- 小規模データでの過学習抑制に特化

**期待効果**:
- Residual Dropoutで4 BLEU points以上の改善（文献実績）
- 実装がMasked Reconstructionよりシンプル

**実装難易度**: ★★☆☆☆（低）

**必要データ**: 上記と同じ

### 3. **Temporal Convolutional Network (TCN) Autoencoder**（フォールバック）

**概要**:
- 畳み込みベースの時系列Autoencoder
- TransformerがGate 1/2で失敗した場合の代替案
- パラメータ効率が高い

**期待効果**:
- Transformerより安定（小規模データでの実績）
- 受容野の制御が容易

**実装難易度**: ★★★★☆（やや高）

**必要データ**: 上記と同じ

---

## 利用可能なデータソース

### 既存データ（追加取得不要）

| データ | ソース | 場所 | 期間 | 取得コード |
|--------|--------|------|------|-----------|
| **Base features** | FRED, Yahoo Finance | data/processed/base_features.csv | 2014-2024 | builder_dataが実装済み |
| **Submodel outputs** | 既存サブモデル | data/submodel_outputs/*.csv | 2014-2024 | builder_dataが実装済み |
| **Gold price (target)** | Yahoo Finance | data/processed/target.csv | 2014-2024 | builder_dataが実装済み |

### 入力特徴量の選択基準

**高文脈依存性（必須）**:
1. `vix_regime_probability` (vix HMM)
2. `vix_mean_reversion_z` (vix rolling z-score)
3. `tech_trend_regime_prob` (technical HMM)
4. `tech_mean_reversion_z` (technical z-score)
5. `tech_volatility_regime` (technical HMM)
6. `xasset_regime_prob` (cross_asset HMM)
7. `xasset_divergence` (cross_asset divergence)
8. `etf_regime_prob` (etf_flow HMM)
9. `options_risk_regime_prob` (options_market HMM)

**Base features（補完的）**:
1. `real_rate_change` (DFII10日次変化)
2. `dxy_change` (DXY日次変化)
3. `vix` (VIXレベル)
4. `yield_spread_change` (10Y-2Y spread変化)
5. `inflation_exp_change` (T10YIE変化)

**除外する特徴量**（低文脈依存性、冗長性）:
- `yield_curve`, `inflation_expectation`, `cny_demand`のsubmodel outputs（マクロ長期傾向が支配的）
- `persistence`, `capital_intensity`等の補助的特徴量（VIFリスク）

---

## 注意事項

### 過学習リスクの管理

**Primary Risk（確率30-40%）**:
- 2131サンプルでTransformerが過学習し、有意義な文脈特徴量を生成できない

**Mitigation**:
1. **極小モデル**: 1-2層、32-64次元、2-4ヘッド
2. **強いDropout**: 0.3-0.5（Attention + FFN + Residual）
3. **Weight Decay**: 0.05-0.1
4. **Early Stopping**: patience=5-10 epochs
5. **Masked Reconstruction**: 過学習抑制効果あり（全時間ステップを使わない）

**real_rate attempt 2-3からの教訓活用**:
- Multi-country Transformerで過学習を解決（overfit ratio 1.28）
- 30 Optuna trialsで十分な探索が可能

### VIF制約とGate 2

**Secondary Risk（確率25-35%）**:
- Transformer出力がHMM regime_probと高相関 → Gate 2失敗（MI < 5%, VIF > 10）

**Mitigation**:
1. **出力次元を1-2に制限**（real_rate 7カラム失敗の教訓）
2. **Correlation Regularization損失**（実験的、attempt 2以降）:
   ```python
   # Bottleneckとregime_probの相関ペナルティ
   corr_penalty = torch.corrcoef(bottleneck, regime_prob).abs().mean()
   total_loss = reconstruction_loss + 0.1 * corr_penalty
   ```
3. **Masked Prediction目標**で複雑な学習タスク（単なるスムージング防止）

### Gate 3とXGBoost統合

**Tertiary Risk（確率20-30%）**:
- Gate 3で既存submodel outputsとの冗長性により追加価値を提供できない

**Mitigation**:
1. **出力を1カラムに絞る**（options_market 1カラム成功の教訓）
2. **0-1正規化**（HMM regime_probと同様のスケール）
3. **DA/Sharpe改善に集中**（MAEは改善困難と予測）

### Kaggle環境制約

**Time Constraint**:
- 30分以内に学習完了（Optuna 20-30 trials）
- 軽量モデル（1-2層、32-64次元）で遵守可能

**GPU Constraint**:
- Kaggle T4 GPU（16GB VRAM）
- Batch size 32-64で問題なし

### データ周波数の整合性

**real_rateの根本的失敗原因は解消**:
- real_rateは月次 → 日次補間がGate 3失敗の根本原因
- temporal_contextは**全入力が日次**（base + submodel outputs）
- Forward-fillされたstep functionの問題なし

---

## 結論

Temporal Context Transformerサブモデルは、**Vanilla Transformer Autoencoder with Masked Reconstruction**を第1優先として開発すべきである。小規模金融データ（2131サンプル）での過学習リスクを管理するため、極小モデル（1-2層、32-64次元、2-4ヘッド）と強い正則化（Dropout 0.3-0.5、Weight Decay 0.05-0.1）を採用する。

ウィンドウサイズは5-20日（Optuna探索）、位置エンコーディングはLearned PE、学習目標はMasked Reconstruction、出力次元は1-2カラムに制限する。これにより、XGBoostが単一時点からは捕捉できない「クロス時間・クロス特徴量の相互作用パターン」を抽出し、meta_modelの方向精度（DA）を+0.5%以上改善することを目指す。

PatchTST、iTransformer、Contrastive Learning等の最新手法は実装複雑性が高く、attempt 1では採用しない。これらはattempt 2以降の改善策として保留する。

---

## Sources

### Transformer Autoencoder & Financial Data
- [Supervised Autoencoder MLP for Financial Time Series Forecasting](https://arxiv.org/abs/2404.01866)
- [Deep Learning in Quantitative Finance: Transformer Networks](https://blogs.mathworks.com/finance/2024/02/02/deep-learning-in-quantitative-finance-transformer-networks-for-time-series-prediction/)
- [Deep Learning Models for Price Forecasting Review](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1519)
- [LSTM Autoencoder-Based Network of Financial Indices](https://www.nature.com/articles/s41599-025-04412-y)

### PatchTST & Latest Architectures
- [PatchTST: A Time Series is Worth 64 Words (ICLR 2023)](https://arxiv.org/abs/2211.14730)
- [PatchTST: A Breakthrough in Time Series Forecasting](https://www.datasciencewithmarco.com/blog/patchtst-a-breakthrough-in-time-series-forecasting)
- [CT-PatchTST: Channel-Time Patch Transformer](https://arxiv.org/html/2501.08620v3)

### Window Size & Lookback
- [iTransformer: Inverted Transformers (ICLR 2024)](https://arxiv.org/html/2310.06625v4)
- [Dynamic Optimisation of Window Sizes](https://www.preprints.org/manuscript/202501.1424)
- [Modality-aware Transformer for Financial Forecasting](https://arxiv.org/html/2310.01232v2)

### Positional Encoding
- [Positional Embeddings in Transformers: RoPE & ALiBi](https://towardsdatascience.com/positional-embeddings-in-transformers-a-math-guide-to-rope-alibi/)
- [Positional Encoding in Transformer Time Series: Survey](https://www.arxiv.org/pdf/2502.12370)
- [Improving Position Encoding for Multivariate Time Series](https://link.springer.com/article/10.1007/s10618-023-00948-2)

### Regularization & Dropout
- [Advanced Regularization Protocols for Transformers](https://medium.com/@hassanbinabid/the-art-and-science-of-hyperparameter-optimization-in-llm-fine-tuning-f95bc6e9a80b)
- [Residual Dropout: A Simple Approach (ACL 2024)](https://aclanthology.org/2024.sigul-1.35.pdf)
- [Transformer Regularization Techniques](https://apxml.com/courses/foundations-transformers-architecture/chapter-7-implementation-details-optimization/transformer-regularization)

### Multi-Head Attention
- [Novel Transformer-Based Dual Attention for Financial Prediction](https://link.springer.com/article/10.1007/s44443-025-00045-y)
- [Enhanced Transformer for Online Stock Price Prediction](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0316955)
- [LiteFormer: Lightweight Transformer](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729648)

### Self-Supervised Learning Objectives
- [TempSSL: Rethinking Self-Supervised Learning](https://www.sciencedirect.com/science/article/abs/pii/S0950705124012863)
- [TSDE: Diffusion Process for Time Series (KDD 2024)](https://dl.acm.org/doi/10.1145/3637528.3671673)
- [SSL4TS: Self-Supervised Learning Survey (IEEE TPAMI)](https://arxiv.org/pdf/2306.10125)

### Financial Forecasting Success Cases
- [Transformer Encoder and Time2Vec (EUSIPCO 2025)](https://eusipco2025.org/wp-content/uploads/pdfs/0001682.pdf)
- [LSTM + Transformer Sentiment for Apple Inc.](https://www.anserpress.org/journal/jea/4/3/109/pdf)
- [Fx-spot Predictions with SOTA Transformer](https://www.sciencedirect.com/science/article/pii/S0957417424004032)
- [STL-Transformer Hybrid Model](https://dl.acm.org/doi/pdf/10.1145/3760622.3760649)
- [Deep Learning for Financial Forecasting: A Review](https://www.sciencedirect.com/science/article/pii/S1059056025008822)

---

**レポート終了**
