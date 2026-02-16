# Kaggle Dataset 更新手順

## 問題
`gold-prediction-submodels` Datasetに不要なファイル（ログ、サブディレクトリ）が含まれており、必要なCSVファイルの一部が欠けています。

### 欠けているファイル
- vix.csv
- technical.csv
- yield_curve.csv
- options_market.csv
- temporal_context.csv

## 解決方法：Web UIから手動更新

### ステップ1: Datasetページを開く
https://www.kaggle.com/datasets/bigbigzabuton/gold-prediction-submodels

### ステップ2: 新しいバージョンを作成
1. ページ右上の "New Version" ボタンをクリック
2. Version Type: "Complete" を選択（すべて置き換え）

### ステップ3: ファイルをアップロード
以下のディレクトリから10個のCSVファイルをすべてアップロード：

**ディレクトリ**: `C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents\data\dataset_upload_clean\`

**ファイル一覧** (合計1.9MB):
1. vix.csv (197 KB)
2. technical.csv (244 KB)
3. cross_asset.csv (180 KB)
4. etf_flow.csv (191 KB)
5. options_market.csv (118 KB)
6. yield_curve.csv (197 KB)
7. inflation_expectation.csv (202 KB)
8. cny_demand.csv (193 KB)
9. real_rate.csv (271 KB)
10. **temporal_context.csv (54 KB)** ← 新規追加

### ステップ4: 設定
- Visibility: Private（変更なし）
- Version notes: "Clean version: 10 CSV files only, removed logs and subdirs, added temporal_context.csv"

### ステップ5: 保存
"Create Version" をクリック

## 完了後
1. Dataset更新完了後、このファイルに「✅ 完了」と追記してください
2. meta_model attempt 7 のKernelを再提出します

---

## 進捗状況
- [ ] Dataset更新開始
- [ ] 10個のCSVファイルアップロード完了
- [ ] Version作成完了
- [ ] Kernelへの反映確認完了

更新完了したら「Datasetを更新したよ」と教えてください。
