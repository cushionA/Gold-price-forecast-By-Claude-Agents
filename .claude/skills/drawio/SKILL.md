---
name: drawio
description: draw.io ダイアグラムを生成する。MCP ツール（open_drawio_xml / open_drawio_mermaid / open_drawio_csv）を使い分けてブラウザ上の draw.io エディタで開く。
disable-model-invocation: true
argument-hint: [diagram-type] [description]
---

# draw.io ダイアグラム生成スキル

MCP ツールを使ってダイアグラムを生成し、ブラウザ上の draw.io エディタで開く。
**状況に応じて 3つの MCP ツールを使い分ける。**

## 引数

```
$0 = 図の種別（省略可）: architecture / flowchart / sequence / er / network / state / class / mindmap / org / gantt
$1+ = 図の説明（自然言語）
```

例:
```
/drawio architecture AWS上のサーバーレスAPIシステム
/drawio flowchart ユーザー登録フロー
/drawio org 開発チームの組織図
/drawio 現在のプロジェクトのパイプライン図を作成
```

---

## 3つの MCP ツール

| MCP ツール | 引数 | 最適な用途 | 自動レイアウト |
|-----------|------|-----------|--------------|
| `open_drawio_xml` | `content`, `lightbox?`, `dark?` | AWSアーキテクチャ図、精密な配置、カスタムアイコン | なし（手動座標） |
| `open_drawio_mermaid` | `content`, `lightbox?`, `dark?` | フローチャート、シーケンス図、ER図、ステート図、クラス図 | あり（自動） |
| `open_drawio_csv` | `content`, `lightbox?`, `dark?` | 組織図、ツリー構造、データ駆動の図、大量ノード | あり（自動） |

### 共通引数

| パラメータ | 型 | 必須 | デフォルト | 説明 |
|-----------|-----|------|-----------|------|
| `content` | string | 必須 | — | XML / Mermaid / CSV の文字列 |
| `lightbox` | boolean | 任意 | false | true で読み取り専用ビューアー |
| `dark` | string | 任意 | `"auto"` | `"auto"` / `"true"` / `"false"` |

### 方式の選択フロー

```
AWSアイコン / カスタムスタイル / 精密な座標配置が必要？
  → YES → open_drawio_xml
  → NO ↓
ノード数が多い（20+）/ データ駆動 / 組織図・ツリー構造？
  → YES → open_drawio_csv
  → NO ↓
Mermaid がサポートする図？（flowchart/sequence/er/state/class/gantt/pie/mindmap）
  → YES → open_drawio_mermaid
  → NO → open_drawio_xml
```

### 図の種別ごとの推奨方式

| 図の種別 | 第1推奨 | 第2推奨 | 理由 |
|---------|---------|---------|------|
| AWSアーキテクチャ | xml | — | AWSアイコン必須 |
| フローチャート | mermaid | xml | 自動レイアウトが便利 |
| シーケンス図 | mermaid | — | Mermaid が最適 |
| ER図 | mermaid | xml | カーディナリティ表記が簡単 |
| ステート図 | mermaid | xml | 状態遷移の表記が簡潔 |
| クラス図 | mermaid | xml | UML記法サポート |
| 組織図 | csv | xml | 階層+自動レイアウト |
| ネットワーク図 | xml | csv | カスタムアイコンが必要 |
| マインドマップ | mermaid | — | Mermaid が最適 |
| データ駆動の図（20+ノード） | csv | — | 大量ノードの自動配置 |
| ガントチャート | mermaid | — | Mermaid が最適 |
| パイプライン図 | mermaid | xml | フロー系は Mermaid が簡潔 |

---

## 実行手順

### Step 1: 要件の確認

1. ユーザーの説明から図の種別・構成要素・関係性を特定
2. 不明点があれば質問してから作成開始
3. 上記の選択フローに従い **最適な方式を選択**

### Step 2: コンテンツの生成

選択した方式に応じて content 文字列を組み立てる（各方式の詳細は後述）。

### Step 3: MCP ツールで開く

```
# XML方式
open_drawio_xml(content="<mxfile>...</mxfile>")

# Mermaid方式
open_drawio_mermaid(content="flowchart TD\n    A-->B")

# CSV方式
open_drawio_csv(content="# label: %name%\n...")
```

ブラウザで draw.io が自動的に開き、ダイアグラムが表示される。

---

## open_drawio_xml — XML方式

**精密な配置・AWSアイコン・カスタムスタイルが必要な場合に使用。**

### AWSアイコンの取得

```bash
python .claude/skills/drawio/scripts/find_drawio_icon.py "lambda" "s3" "api gateway"
```

### XML 基本構造

```xml
<mxfile host="app.diagrams.net" version="24.0.0" type="device">
  <diagram name="Page-1" id="page1">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10"
                  guides="1" tooltips="1" connect="1" arrows="1"
                  fold="1" page="1" pageScale="1" pageWidth="1169"
                  pageHeight="827" math="0" shadow="0"
                  background="none" defaultFontFamily="Helvetica">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <!-- 矢印を先、図形を後に配置 -->
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

### XML 必須ルール

| ルール | 詳細 |
|--------|------|
| フォント | `defaultFontFamily` + 各セルの `style` に `fontFamily=Helvetica;` を **両方** 設定 |
| 矢印の配置順序 | `<root>` 内で矢印セルを図形セルより **前に** 記述 |
| テキスト幅 | 日本語: 1文字40px、英語: 1文字10px + パディング40px |
| マージン | グループ内30px、要素間40px、矢印ラベル距離20px |
| コネクタ | `edgeStyle=orthogonalEdgeStyle;rounded=1;` 推奨 |
| グループ | `container=1;collapsible=0;` + `parent` 属性で階層化 |
| AWSアイコン | `aspect=fixed;` 必須（歪み防止）、サイズ `78x78` 推奨 |

### 詳細な参照ファイル

| ファイル | 内容 |
|---------|------|
| [references/xml-templates.md](references/xml-templates.md) | 図形・矢印・グループの XML テンプレート |
| [references/aws-icons.md](references/aws-icons.md) | AWS アイコンカタログ（80+ サービス） |
| [references/layout-guidelines.md](references/layout-guidelines.md) | レイアウトパターン・配色パレット |

**重要**: 参照ファイルを丸ごと読み込まず、必要な部分のみ参照すること。
AWSアイコンは `find_drawio_icon.py` で必要なもののみ取得する。

---

## open_drawio_mermaid — Mermaid方式

**フローチャート・シーケンス図・ER図など、自動レイアウトが望ましい場合に使用。**

### フローチャート

```
flowchart TD
    A[Start] --> B{Condition?}
    B -->|Yes| C[Process A]
    B -->|No| D[Process B]
    C --> E[End]
    D --> E
```

方向: `TD`（上→下）、`LR`（左→右）、`BT`（下→上）、`RL`（右→左）

ノード形状:
- `[text]` → 長方形
- `(text)` → 角丸
- `{text}` → ひし形
- `([text])` → スタジアム
- `[(text)]` → シリンダー
- `((text))` → 円
- `{{text}}` → 六角形

矢印:
- `-->` 実線矢印 / `-.->` 破線矢印 / `==>` 太線矢印
- `-->|label|` ラベル付き / `---` 線のみ

サブグラフ:
```
subgraph Title
    A --> B
end
```

スタイル:
```
style A fill:#f9f,stroke:#333,stroke-width:2px
classDef highlight fill:#bbf,stroke:#33f
class B highlight
```

### シーケンス図

```
sequenceDiagram
    participant U as User
    participant A as API
    participant D as DB

    U->>A: POST /users
    A->>D: INSERT
    D-->>A: OK
    A-->>U: 201 Created

    Note over A,D: Data layer
    alt success
        A-->>U: 200
    else error
        A-->>U: 500
    end
```

矢印: `->>` 同期 / `-->>` 応答 / `-x` 失敗 / `-)` 非同期

### ER図

```
erDiagram
    USERS ||--o{ ORDERS : places
    USERS {
        int id PK
        string name
        string email
    }
    ORDERS ||--|{ ORDER_ITEMS : contains
    ORDERS {
        int id PK
        int user_id FK
        date created_at
    }
```

カーディナリティ: `||` (1) / `o{` (0以上) / `|{` (1以上) / `o|` (0 or 1)

### ステート図

```
stateDiagram-v2
    [*] --> Idle
    Idle --> Processing : submit
    Processing --> Success : complete
    Processing --> Error : fail
    Error --> Idle : retry
    Success --> [*]
```

### クラス図

```
classDiagram
    class Animal {
        +String name
        +int age
        +makeSound() void
    }
    class Dog {
        +fetch() void
    }
    Animal <|-- Dog
```

### マインドマップ

```
mindmap
    root((Central Topic))
        Branch A
            Leaf A1
            Leaf A2
        Branch B
            Leaf B1
```

### ガントチャート

```
gantt
    title Project Schedule
    dateFormat YYYY-MM-DD
    section Phase 1
    Task A :a1, 2024-01-01, 30d
    Task B :after a1, 20d
    section Phase 2
    Task C :2024-02-15, 25d
```

### Mermaid の注意点

- AWSアイコンは使えない → AWS図にはXML方式を使う
- 精密な座標指定は不可 → レイアウトは自動
- 特殊文字（`<`, `>`, `&`）はノードテキストで避けるか引用符で囲む
- content に含める文字列は Mermaid 構文のみ（```mermaid のフェンスは不要）

---

## open_drawio_csv — CSV方式

**組織図・ツリー構造・データ駆動の大量ノード図に使用。**

### CSV フォーマット

ヘッダーディレクティブ（`#` コメント）+ CSV データで構成。

```
# label: %name%<br><i style="font-size:11px;">%role%</i>
# style: shape=%shape%;fillColor=%fill%;strokeColor=%stroke%;fontFamily=Helvetica;fontSize=14;whiteSpace=wrap;html=1;rounded=1;
# namespace: csvimport-
# connect: {"from":"manager","to":"id","invert":true,"style":"curved=1;endArrow=blockThin;endFill=1;fontSize=11;edgeStyle=orthogonalEdgeStyle;"}
# width: auto
# height: auto
# padding: 15
# ignore: id,manager,shape,fill,stroke
# nodespacing: 40
# levelspacing: 100
# edgespacing: 40
# layout: verticaltree
## CSV data below
id,name,role,manager,shape,fill,stroke
1,CEO,Chief Executive,,rectangle,#dae8fc,#6c8ebf
2,CTO,Chief Technology,1,rectangle,#d5e8d4,#82b366
3,CFO,Chief Financial,1,rectangle,#fff2cc,#d6b656
4,Dev Lead,Development,2,rectangle,#d5e8d4,#82b366
5,Ops Lead,Operations,2,rectangle,#d5e8d4,#82b366
```

### 主要ディレクティブ

| ディレクティブ | 説明 | 例 |
|--------------|------|-----|
| `# label:` | 表示テキスト（`%col%` でプレースホルダ、HTML可） | `%name%<br><i>%title%</i>` |
| `# style:` | 全ノード共通スタイル（`%col%` でデータ駆動の色等） | `fillColor=%fill%;` |
| `# stylename:` | スタイルを列値で切り替える列名 | `type` |
| `# styles:` | stylename→スタイルのJSON | `{"server":"shape=image;..."}` |
| `# connect:` | 接続定義（JSON、複数行可） | `{"from":"parent","to":"id","invert":true}` |
| `# width:` / `# height:` | ノードサイズ（数値 or `auto`） | `auto` |
| `# padding:` | テキストパディング | `15` |
| `# ignore:` | 非表示にする列 | `id,parent,fill` |
| `# nodespacing:` | ノード間隔 | `40` |
| `# levelspacing:` | 階層間隔 | `100` |
| `# layout:` | レイアウト | `verticaltree` |

### layout の選択肢

| layout | 用途 |
|--------|------|
| `verticaltree` | 組織図（上→下） |
| `horizontaltree` | 横方向ツリー |
| `verticalflow` | フローチャート（上→下） |
| `horizontalflow` | フローチャート（左→右） |
| `organic` | ネットワーク図（力学的配置） |
| `circle` | 放射状配置 |
| `auto` | 自動選択 |
| `none` | レイアウトなし |

### connect の詳細

```json
{
  "from": "parent_id",
  "to": "id",
  "invert": true,
  "label": "manages",
  "style": "curved=1;endArrow=blockThin;endFill=1;edgeStyle=orthogonalEdgeStyle;"
}
```

### CSV の注意点

- `%column%` はラベルとスタイルの両方で使用可能（データ駆動の色分け）
- `# connect:` を複数行書けば複数種類の接続を定義可能
- `# stylename:` + `# styles:` でノード種別ごとにスタイル切替可能
- 信頼性は中程度 — `%column%` 記法が不安定な場合あり

---

## デザイン原則（全方式共通）

1. **標準記法を使う**: 広く認知されたアイコン・記号を使用
2. **全要素にラベルを付ける**: アイコン、グループ、コネクタすべてに明確なラベル
3. **一貫性を維持**: 色・サイズ・線の太さ・矢印の形状を統一
4. **正確性を優先**: 見た目のために正確さを犠牲にしない
5. **凡例を提供する**: 独自の表記ルールは必ず説明
6. **1枚に詰め込まない**: 複雑な図は段階的に分割
7. **コネクタ種別で意味を区別**: 実線=同期、破線=非同期、太線=メインフロー

---

## 参照ファイル一覧

| ファイル | 内容 | 方式 |
|---------|------|------|
| [references/xml-templates.md](references/xml-templates.md) | XML テンプレート集 | xml |
| [references/aws-icons.md](references/aws-icons.md) | AWSアイコンカタログ | xml |
| [references/layout-guidelines.md](references/layout-guidelines.md) | レイアウト・配色 | 全方式 |
| [scripts/find_drawio_icon.py](scripts/find_drawio_icon.py) | AWSアイコン検索 | xml |
