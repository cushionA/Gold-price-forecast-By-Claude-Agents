# Claude Codeで「設計→学習→評価」を30回以上自動で回した話 ― 自動化パイプラインの運用記録

# はじめに

機械学習モデルの開発には、設計・データ取得・前処理・学習・評価・改善という長いサイクルがあります。
これを毎回手動で回すのは正直かなり面倒です。
本当にやること多すぎるので……。

なので、今回はこれを **Claude Code（AIエージェント）に丸ごと自動化させよう** という試みでした。
単にコードを書かせるだけではなく、**7つの専門エージェントを組み合わせたパイプライン**を構築し、設計から学習・評価までの全工程を自律的に回す仕組みを作りました。

さらに、学習処理はKaggleのクラウドGPUで実行し、**APIを通じて自動的に監視・結果取得する設計**にしています。
これにより、学習中はPCを閉じても問題ありません。

結果として、金（Gold）の翌日リターンを予測する回帰モデルが完成しました。

- **方向精度 60.04%**（ベースライン43.54%から+16.50pp）
- **高確信度方向精度 64.13%**（ベースライン42.74%から+21.39pp）
- **Sharpe Ratio 2.46**（ベースライン-1.70から+4.16）

この記事では、予測モデルの中身よりも**パイプラインをどう設計し、どう運用したか**に焦点を当てて解説します。
（正直近年は金が上がりすぎて、持ってるだけでいい感じしますし…）

---

# 全体構成：何を自動化したのか

## 目標
まず金価格に影響する9つの特徴量（実質金利・ドル指数・VIXなど）に対して、それぞれ「サブモデル」を構築します。
サブモデルは金価格を予測するのではなく、各特徴量の裏にある**レジーム（局面）や持続性といった潜在パターンを抽出**する役割です。
これらの**サブモデル出力をメタモデルに統合して、最終的な予測**を行います。

```
生の特徴量 → サブモデル（状態・文脈を抽出） → メタモデル → 予測
```

このような設計にした意図は、より多くの文脈を圧縮して金予想モデルに持たせたかったからです。
ともかく、モデル作成パイプラインを1サブモデルあたり1〜6回、計30回以上の設計→学習→評価サイクルで回す必要があります。

## 7つの専門エージェント

Claude Codeの `Task` 機能を使って、それぞれ異なる役割を持つエージェントを定義しました。

```
Orchestrator（Sonnet） ← 全体の進行管理・git操作
  │
  ├─ entrance（Opus）      ← 要件定義：何を作るか・調査事項の洗い出し
  ├─ researcher（Sonnet）  ← 手法調査：論文・先行事例のリサーチ
  ├─ architect（Opus）     ← 設計：ファクトチェック → 設計書 → HP探索空間
  ├─ builder_data（Sonnet）← データ取得・前処理
  ├─ datachecker（Haiku）  ← データ品質の定型7ステップチェック
  ├─ builder_model（Sonnet）← PyTorch学習スクリプト生成（Kaggle用）
  └─ evaluator（Opus）     ← Gate 1/2/3 評価 → ループ制御 → 改善計画
```

各エージェントは `.claude/agents/` ディレクトリに定義されたプロンプトに従って動作します。
Orchestratorが `shared/state.json` を読み、現在の進捗に応じて適切なエージェントを呼び出す仕組みです。

|<font color=""> **パイプライン全体像** </font>|
|:-:|
|![パイプライン全体像](証跡画像/pipeline_overview.png)|

---

# 1サイクルの流れ：何が自動で起きるのか

一つのサブモデルの構築は、以下の流れで自動的に進みます。

```
entrance: 「VIXサブモデルを作ります。調査事項は...」
    ↓
researcher: 「VIXのレジーム検出にはHMMが有効です。根拠は...」
    ↓
architect: 「リサーチ結果をファクトチェック → 設計書を作成」
    ↓
builder_data: 「FRED APIからVIXデータを取得 → 前処理」
    ↓
datachecker: 「7ステップチェック → 欠損・異常値・未来リーク確認」
    ↓
builder_model: 「PyTorch学習スクリプトをJupyter Notebookとして生成」
    ↓
[Kaggle提出 → クラウドで学習] ← PCを閉じてOK
    ↓
evaluator: 「Gate 1/2/3で品質評価 → PASS or 改善計画」
```

ポイントは**entranceからevaluatorまでのAgentがPCDAを回し続けること**です。

- evaluator が FAIL → attempt +1 として改善ループに入る
- entrance が再度方針を決めて、researcher が調査してarchitect が再設計

この設計により、設計と実行、結果評価がループするようにしています。

---

# Kaggleクラウド学習：PCを閉じても学習が進む仕組み

## なぜClaude CodeとKaggleを分離したのか

Claude Codeはローカルで動作するツールです。
一方、ニューラルネットの学習は数分〜30分かかることがあり、その間Claude Codeが動きっぱなしというのもトークンが無駄な気がします。
なので**Kaggleにノートブックを提出し、スクリプトでKaggleAPIを使用するループでその完了を確認し、Claude SDKで自動的にタスクの続きをトリガー**します。
この結果確認スクリプトにより、エラー発生時も自動的に修正が走るようになっています。

## 実行の流れ

```
1. Claude Code: 設計 → データ取得 → チェック → 学習スクリプト生成
2. Claude Code: Kaggle APIでNotebookを提出
3. Kaggle: クラウドで学習実行（数分〜30分）
4. Claude Code: 監視スクリプト起動
5. 監視スクリプト(Python): Kaggle APIで成功orエラーを検知。→ Claude Codeに制御を戻す
6. Claude Code: 続きから再開
   → git pull → Kaggle結果取得 → 評価 → 次のイテレーション
```

## kaggle_ops.py：Kaggle操作の一元管理

Kaggleとのやり取りは `scripts/kaggle_ops.py` という単一モジュールに集約しました。

```python
from scripts.kaggle_ops import submit_and_monitor

# 提出 → state.json更新 → git commit → バックグラウンド監視開始
result = submit_and_monitor(
    folder="notebooks/vix_1/",
    feature="vix",
    attempt=1,
)
```

`submit_and_monitor` は以下を1コマンドで実行します。

1. **Notebook提出**: Kaggle API v2でNotebookをプッシュ
2. **状態更新**: `state.json` を `"waiting_training"` に変更
3. **gitコミット&プッシュ**: 進捗をリポジトリに記録
4. **バックグラウンド監視開始**: 1分間隔でKaggleのステータスをポーリング

学習が完了すると、monitorが自動的に結果をダウンロードし、`state.json` を更新してgitコミットします。
次にPCを起動したとき、Orchestratorは `state.json` を読んで「結果取得済み→evaluatorを呼ぶ」と判断できます。

---

# 品質ゲート：3段階のふるい

evaluatorは以下の3段階でサブモデルの品質を評価します。全Gateを突破して初めて「このサブモデルは使える」と判定されます。

**Gate 1：スタンドアロン品質** — そもそもまともに動いているか
- 過学習比率 < 1.5
- 出力にNaN・定数がないか
- リーク指標がないか

**Gate 2：情報利得** — メタモデルにとって新しい情報を提供しているか
- 相互情報量の増加 > 5%
- VIF（多重共線性）< 10
- ローリング相関の安定性 < 0.15

**Gate 3：アブレーション** — 実際に予測を改善するか（以下のいずれか一つ）
- 方向精度 +0.5%
- Sharpe Ratio +0.05（取引コスト控除後）
- MAE -0.01%

Gate 3がこのシステムの肝です。
情報理論的には良さそうに見えても（Gate 2 PASS）、実際のアブレーションで効果が出ないケース（Gate 3 FAIL）は何度もありました。
「理屈上は有用なはず」と「実際に予測が良くなる」の間にはギャップがあり、Gate 3はそのギャップを検出する最後の関門です。

---

# パイプラインを安定させるための4つの仕組み

30回以上のイテレーションをエージェントに任せるうえで、同じ失敗を繰り返さない仕組みが不可欠でした。
実際に運用してみると、**Kaggle上でのエラーが最大のボトルネック**です。
ローカルでは動くのにKaggle環境ではクラッシュする、というパターンが繰り返し発生したため、4つの防御層を設けています。

## (1) Notebook検証スクリプト（提出前チェック）

Kaggle提出は1回あたり数分〜30分のフィードバックループなので、構文エラーや互換性問題でその時間を無駄にしたくありません。
`scripts/validate_notebook.py` という事前検証スクリプトを作り、builder_modelが生成したNotebookを**Kaggle提出前に必ずチェック**する運用にしました。

```python
# builder_modelエージェントの必須ステップ
# 生成 → 検証 → エラーなら修正 → 再検証 → PASSしたら提出
python scripts/validate_notebook.py notebooks/vix_1/
```

検証項目は10個あり、**全て実際のKaggleエラーから逆算して作られたもの**です。

| # | チェック内容 | 背景 |
|---|------------|------|
| 1 | Python構文チェック（ast.parse） | 構文エラーはKaggleでも即死 |
| 2 | メソッド名タイポ（`.UPPER()` 等） | LLMが大文字メソッドを生成しがち |
| 3 | SHAP + XGBoost 2.x 互換性 | Kaggle環境のバージョン不整合 |
| 4 | データセットパス参照の整合性 | API v2のパス変更で頻発 |
| 5 | kernel-metadata.json検証 | 必須フィールド・設定値チェック |
| 6 | 未定義変数の基本検出 | セル間依存の漏れを検出 |
| 7 | 非推奨pandas API | pandas 2.xで即クラッシュ |
| 8 | Optunaの動的パラメータ空間 | 探索空間の不整合を警告 |
| 9 | yfinanceの空データチェック | 取得失敗の未処理を検出 |
| 10 | GPU設定とCUDAコードの整合 | GPU有効化しているのにCPUコードだけ |

最初は構文チェックだけでしたが、pandasの非推奨APIでクラッシュしたら項目7を追加、データセットパスで落ちたら項目4を追加……という具合に、エラーが起きるたびにチェック項目を増やしていきました。

## (2) ナレッジベース（知識の再利用）

`docs/knowledge/` ディレクトリに、Q&A形式のナレッジベースを構築しました。

```
docs/knowledge/
├── methodologies/
│   ├── regime_detection.md         ← レジーム検出手法の比較
│   └── persistence_measurement.md  ← 持続性指標の計算方法
├── data_sources/
│   └── fred_multi_country.md       ← FREDで取得可能なシリーズ一覧
└── financial_concepts/
    └── window_lengths.md           ← ローリングウィンドウの適切な長さ
```

9つのサブモデルには**共通する知識**が多くあります。
「レジーム検出にはどの手法がいいか？」という問いは、VIX・DXY・テクニカル・クロスアセットなど多くのサブモデルで必要です。
ナレッジベースがなければ、researcherが毎回同じ調査をすることになります。
実際、導入前は似たような内容の調査レポートが `docs/research/` に何本も生成されていました。

researcherはまず `docs/knowledge/` をチェックし、既存の回答があればそれを引用。なければ新たに調査してナレッジベースに追記する、という流れです。

## (3) MEMORY.md（セッションをまたぐ記憶）

Claude Codeには `.claude/` 配下にプロジェクト固有のメモリファイルを持つ仕組みがあります。
ここにKaggle APIの罠やプロジェクト固有のルールを蓄積しました。

<!-- open属性なし -->
<details><summary><strong>MEMORY.mdに蓄積された知識の例（展開）</strong></summary>

```
## Kaggle API (Python v2.0.0)
- Windows cp932 problem: builtins.openをパッチしてUTF-8強制
- Auth mapping: .envのKAGGLE_API_TOKENをKAGGLE_KEYにリマップ必要
- Dataset mount path (API v2): /kaggle/input/datasets/{owner}/{slug}/
  （旧パス /kaggle/input/{slug}/ は動かない）
- 409 Conflict: 実行中のkernelがあるとpush失敗。Web UIから停止が必要
- Notebook内ではKaggle CLI認証が効かない。Secretsが必要

## Kaggle Workflow Rules (MUST FOLLOW)
- NEVER call notebook_push() directly. Always use submit() or submit_and_monitor()
- NEVER poll Kaggle status manually. Always use monitor()
```

</details>

これらは一度踏んだ地雷の記録です。
エージェントは新しいセッションを開始するたびにこのファイルを読むため、**同じ罠に二度はまらない**仕組みになっています。

特にKaggle APIの仕様はドキュメントが薄く、試行錯誤で得た知識が多いため、この記憶の蓄積は非常に重要でした。

## (4) エラー自動分類と差し戻し先の決定

Kaggle上でエラーが発生した場合、`kaggle_ops.py` のmonitor機能がエラーログを解析し、**種別を自動分類して適切な差し戻し先を決定**します。

```
Kaggleエラー発生
    ↓
monitor: エラーログを取得・解析
    ↓
分類: oom / pandas_compat / dataset_missing / network_timeout / unknown
    ↓
state.json更新: resume_from = "builder_model" + error_type + error_context
    ↓
git commit & push
    ↓
[次回再開時] orchestrator: state.jsonのerror_contextを読み、
    builder_modelに「OOMなのでモデルサイズを縮小して」等の指示を渡す
```

| エラー種別 | 自動対応 |
|-----------|---------|
| `oom` | builder_modelに差し戻し（モデルサイズ縮小） |
| `pandas_compat` | builder_modelに差し戻し（非推奨API修正） |
| `dataset_missing` | builder_modelに差し戻し（データセット参照修正） |
| `network_timeout` | builder_modelに差し戻し（同一コードでリトライ） |

これにより、Kaggleエラーが発生しても人間の介入なしにリカバリサイクルが回ります。
ただし `unknown` に分類されたエラーは人間の判断が必要になることもありました。

---

# 状態管理：PCを閉じて再開しても続きから

## state.jsonによる進捗管理

パイプライン全体の進捗は `shared/state.json` で管理されています。

```json
{
  "status": "waiting_training",
  "current_feature": "vix",
  "current_attempt": 1,
  "resume_from": "evaluator",
  "kaggle_kernel": "bigbigzabuton/gold-vix-1"
}
```

PCを再起動してClaude Codeに「前回の続きから」と伝えるだけで、このファイルを読んで適切な地点から再開します。

```
[ユーザー] 「前回の続きから再開して」
    ↓
[Orchestrator] git pull → state.json読み込み
    → status="waiting_training" → Kaggle結果取得
    → 結果あり → evaluator呼び出し → Gate評価
    → PASS → 次のサブモデルへ
```

## gitによる全行程の記録

全エージェントの作業後にgitコミットを行う設計にしたことで、開発の全行程がgit logに残っています。

```
485179c final: project completed - attempt 7 confirmed as final meta-model
eee508b eval: meta_model attempts 14 & 15 - no improvement over attempt 7
f9e0f19 kaggle: meta_model attempt 15 - results fetched
74f5782 kaggle: meta_model attempt 15 - submitted
...
297f205 eval: regime_classification attempt 1 - COMPLETED
f6a3b3b kaggle: regime_classification attempt 1 - results fetched
894ec7d kaggle: regime_classification attempt 1 - submitted
d1955bf model: regime_classification attempt 1 - notebook generated
41fc479 datacheck: regime_classification attempt 1 - PASS
28ec577 data: regime_classification attempt 1
32515ca design: regime_classification attempt 1
b7b23f4 research: regime_classification attempt 1
1ffffc5 entrance: regime_classification attempt 1
```

一つのサブモデルに対して `entrance → research → design → data → datacheck → model → kaggle submit → kaggle fetch → eval` という一連のコミットが並んでいるのが見て取れます。
これは**デバッグやロールバックの際に極めて便利**で、「どの時点でどのエージェントが何をしたか」が完全にトレース可能です。

---

# 実際の運用で学んだこと

## 完全自律ではない

誤解のないように書いておくと、これは「ボタン一つで完成品ができる」システムではありません。

実際の運用では以下のような人間の介入がありました。

- **Kaggle Secrets（APIキー）の設定**: ブラウザでの手動設定が必要
- **409 Conflictの対応**: Kaggle上でKernelが既に実行中のとき、Web UIから手動で停止する場面があった
- **方針の軌道修正**: 「実質金利はもう諦めて次に行こう」「メタモデルのAttempt 7をファイナルにしよう」といった大きな判断
- **PCの起動**: 学習完了後、結果取得→評価→次のイテレーションを始めるにはPCを起動する必要がある

逆に言うと、**それ以外は全てエージェントが自律的に回しました。**
設計書の作成・データ取得・前処理・学習スクリプト生成・Kaggle提出・結果取得・Gate評価・改善計画の立案と次イテレーションへの引き継ぎ。

## Windows cp932問題：Kaggle APIのUnicode地雷

Kaggle Python API v2.0.0は `open(file, "w")` をエンコーディング指定なしで呼びます。
Windowsではこれがcp932になり、Unicode文字（例: ⚠）でクラッシュします。

解決策として、Kaggle APIをインポートする前に `builtins.open` をパッチしてUTF-8をデフォルトにしました。

```python
_original_open = builtins.open

def _utf8_open(file, mode="r", *args, **kwargs):
    if isinstance(mode, str) and "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _original_open(file, mode, *args, **kwargs)

builtins.open = _utf8_open
# この後にimport kaggleする
```

力技ですが、確実に動きます。

## Kaggle API v2のデータセットパス問題

Kaggle Python API v2.0.0のデータセットマウントパスは `/kaggle/input/datasets/{owner}/{dataset-slug}/` です。
v1時代の `/kaggle/input/{dataset-slug}/` では動きません。
この差異に気づくまでに何度もNotebookがクラッシュしました。

最終的には、Notebook内で複数の候補パスを順番に試す防御的コードを入れています。

## エージェントの「思い込み」をファクトチェックで防ぐ

researcherが調査した内容を、architectがファクトチェックする設計にしたのは正解でした。
researcherが「FRED APIでG10全ての国のTIPS利回りが取得できる」と報告したことがありますが、実際にはそんなシリーズは存在しません。
architectがこれを検出し、researcherに再調査を指示するフローが何度か発動しています。

LLMは「もっともらしい嘘」をつくことがあるため、**調査と設計を別エージェントに分け、設計側でファクトチェックを挟む**というアーキテクチャは有効でした。

## 過学習との闘い

メタモデルのAttempt 1では、訓練時の方向精度94.3%に対してテストでは54.1%という、40ppの過学習が発生しました。
ここからAttempt 7で逆転（テストがtrainを上回る-5.28pp）に至るまでの改善プロセスは、全てエージェントが自律的に行っています。

|<font color=""> **過学習（Train-Test Gap）の改善推移** </font>|
|:-:|
|![過学習の改善推移](証跡画像/overfitting_progress.png)|

evaluatorが立案した改善策は、主に以下の3点でした。

1. **非定常な価格レベル特徴量の削除**（39特徴量→24特徴量）: 金価格$1,300（2018年）と$2,500（2024年）では分割閾値の意味が変わる
2. **方向性重み付き損失関数の廃止**: 訓練DAを94.3%にまで引き上げるが、バリデーションでは50%以下に崩壊する「記憶」を誘発していた
3. **正則化の大幅強化**: max_depth 3→2, min_child_weight 5→25, subsample 0.70→0.48

## 改善の限界を自律的に判断する

Attempt 7以降、8回連続でAttempt 7を下回る結果が出ました。
スタッキング、非対称損失関数、特徴量追加、特徴量削減、アンサンブルなど多様なアプローチを試しましたが、いずれもAttempt 7に及びません。

evaluatorが自律的に「これ以上の改善は困難」と判断し、Attempt 7をファイナルモデルとして確定しています。
この「いつ止めるか」の判断もパイプラインの一部として組み込まれています。

---

# 最終結果

|<font color=""> **ベースラインからの改善** </font>|
|:-:|
|![ベースライン vs 最終モデル](証跡画像/baseline_vs_final.png)|

<table>
  <caption>最終メタモデル（Attempt 7）の成績</caption>
  <thead>
    <tr>
      <th>指標</th>
      <th>目標</th>
      <th>ベースライン</th>
      <th>最終結果</th>
      <th>改善幅</th>
      <th>判定</th>
    </tr>
  </thead>
  <tr>
    <td>方向精度</td>
    <td>> 56%</td>
    <td>43.54%</td>
    <td><strong>60.04%</strong></td>
    <td>+16.50pp</td>
    <td><strong>PASS</strong></td>
  </tr>
  <tr>
    <td>高確信度方向精度</td>
    <td>> 60%</td>
    <td>42.74%</td>
    <td><strong>64.13%</strong></td>
    <td>+21.39pp</td>
    <td><strong>PASS</strong></td>
  </tr>
  <tr>
    <td>MAE</td>
    <td>< 0.75%</td>
    <td>0.714%</td>
    <td>0.943%</td>
    <td>-</td>
    <td>FAIL (免除)</td>
  </tr>
  <tr>
    <td>Sharpe Ratio</td>
    <td>> 0.80</td>
    <td>-1.70</td>
    <td><strong>2.46</strong></td>
    <td>+4.16</td>
    <td><strong>PASS</strong></td>
  </tr>
</table>

MAE目標だけ未達ですが、これはテストセットに2025-2026年の極端な金価格変動（日次リターン3%超が14日間）が含まれるため構造的に達成不可能と判断されました。

---

# まとめ：自動化パイプラインの設計パターン

今回のプロジェクトで得られた、ML自動化パイプラインの設計パターンをまとめます。

| パターン | 実装 | 効果 |
|---------|------|------|
| **エージェント分離** | 7エージェント × 専用プロンプト | 役割が明確で、差し戻し先の決定が容易 |
| **ファクトチェック層** | researcher → architect | LLMの「もっともらしい嘘」を設計前に検出 |
| **提出前検証** | validate_notebook.py（10項目） | Kaggleエラーによる数十分の無駄を防止 |
| **ナレッジベース** | docs/knowledge/ | 9サブモデルで共通知識を再利用 |
| **セッション間記憶** | MEMORY.md | 同じ罠に二度はまらない |
| **エラー自動分類** | kaggle_ops.py monitor | 人間の介入なしにリカバリサイクル |
| **状態ファイル** | state.json | PCオフ→再開でも正確に続きから |
| **全行程git記録** | エージェントごとのコミット | 完全なトレーサビリティ |
| **品質ゲート** | Gate 1/2/3 | 「理屈上は有用」と「実際に改善」のギャップを検出 |

---

# 今後の展望：物理デバイスからの完全解放

今回のシステムでは、学習をKaggleに委託することでPCを閉じられる時間を作りました。
しかし、学習結果の取得・評価・次のイテレーション開始にはPCの起動が必要です。

ここで注目しているのが **クラウド版のClaude Code** です。

```
[現在]
PC起動 → Claude Code起動 → 設計→スクリプト生成→Kaggle提出
→ PC閉じてOK → Kaggle学習 → PC再起動 → 結果取得→評価→次イテレーション

[将来]
指示を出す → クラウド版Claude Codeが全て自律実行
→ Kaggle学習待ち → 自動で結果取得→評価→次イテレーション
→ 完了通知を受け取る
```

物理デバイスの起動すら不要になり、「寝ている間にモデルが完成している」世界が見えてきます。
今回の `state.json` による状態管理と `kaggle_ops.py` によるAPI連携の設計は、そのための土台としてそのまま流用できるはずです。

AIエージェントによる自動開発は、まだ「半自動」の段階です。
しかし、クラウド実行基盤が整えば、人間の役割は「何を作るか」の意思決定だけになっていくのかもしれません。
