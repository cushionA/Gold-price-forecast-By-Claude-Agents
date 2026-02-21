# リサーチレポート: draw.io ダイアグラム作成方法 (Attempt 1)

## 調査事項

1. draw.io XML形式で直接開く方法（URLスキーム、CLI引数、VS Code拡張）
2. Mermaid記法からdraw.ioで開く方法（インポート機能、URLパラメータ）
3. CSVインポート機能（フォーマット仕様、ヘッダーコメント、レイアウト）

---

## 1. open_drawio_xml: XML形式で直接開く方法

### 1-A. ファイルを直接CLIで開く（デスクトップアプリ）

draw.io デスクトップアプリ（Electron製）はファイルパスを引数として受け取れる。

**Windows:**
```bash
# 通常起動（ただしコマンドプロンプトにアタッチされる）
"C:\Program Files\draw.io\draw.io.exe" "C:\path\to\diagram.drawio"

# プロセスをデタッチして開く（推奨）
start /b "C:\Program Files\draw.io\draw.io.exe" "C:\path\to\diagram.drawio"
```

**Mac:**
```bash
/Applications/draw.io.app/Contents/MacOS/draw.io "path/to/diagram.drawio"
```

**注意事項:**
- `--export` フラグなしで起動した場合、GUIでファイルを開く（エディタモード）
- Electron特有の制限で、Windows上ではプロセスがコマンドプロンプトにアタッチされる問題がある
- `start /b` で回避できるが、公式には未解決のissue（#72）

**Claude Codeからの呼び出し:**
```python
import subprocess
diagram_path = r"C:\path\to\diagram.drawio"
subprocess.Popen(
    [r"C:\Program Files\draw.io\draw.io.exe", diagram_path],
    creationflags=subprocess.DETACHED_PROCESS
)
```

### 1-B. エクスポート専用CLI（ヘッドレス）

draw.io デスクトップはエクスポート専用のCLIモードを持つ。GUIを開かずにXMLをPNG/SVG/PDFに変換できる。

```bash
# PNG変換
"C:\Program Files\draw.io\draw.io.exe" --export --format png --output output.png input.drawio

# SVG変換
"C:\Program Files\draw.io\draw.io.exe" -x -f svg -o output.svg input.drawio

# 主なCLIフラグ
# -x, --export        エクスポートモード
# -f, --format        出力フォーマット (pdf, png, jpg, svg, vsdx)
# -o, --output        出力ファイルパス
# -s, --scale         スケール倍率
# -t, --transparent   透過PNG
# -b, --border        余白ピクセル数
# -a, --all-pages     全ページを個別ファイルにエクスポート
# --page-index N      特定ページのみエクスポート
# -r                  フォルダを再帰的に処理
```

### 1-C. Web版URLでXMLを渡す方法

app.diagrams.net はURLパラメータでダイアグラムを開ける。

**URLにファイルパスを指定（GitHub等の公開URL）:**
```
https://app.diagrams.net/#Uhttps://raw.githubusercontent.com/user/repo/main/diagram.drawio
```
`#U` の後にURI-encodeされたURLを付加する。ローカルファイルには使えない。

**XMLをURLエンコードしてURLに埋め込む:**
draw.ioのエクスポート機能（File > Export as > URL）がXMLをdeflate圧縮してBase64エンコードし、URLを生成する。
- 大きなダイアグラムはURLが長くなりブラウザ制限に引っかかる可能性がある
- 変換ツール: https://jgraph.github.io/drawio-tools/tools/convert.html

### 1-D. VS Code拡張

hediet.vscode-drawio 拡張を使うと、`.drawio` / `.drawio.svg` / `.drawio.png` ファイルをVS Code内で直接編集できる。

```bash
# VS Codeでdrawioファイルを開く
code diagram.drawio
```

### 1-E. 各方式の比較

| 方式 | 用途 | メリット | デメリット |
|------|------|----------|------------|
| CLI open（デスクトップ） | 編集 | シンプル、フル機能 | プロセスアタッチ問題（Windows） |
| CLI export | 変換・自動化 | ヘッドレス、スクリプト組込み可 | GUI不要な場合のみ |
| URL #U パラメータ | 共有・閲覧 | ブラウザで即開ける | ローカルファイル不可 |
| VS Code拡張 | 編集・コーディング統合 | エディタ内完結 | 拡張インストール必要 |

---

## 2. open_drawio_mermaid: Mermaid記法から開く方法

### 2-A. GUIからのMermaidインポート

draw.io デスクトップ/Web版の両方で対応。

**手順:**
1. `Arrange > Insert > Advanced > Mermaid` を選択
   または ツールバーの `+` アイコン > Mermaid
2. テキストボックスにMermaid記法を貼り付ける
3. `Insert` をクリックしてダイアグラムを生成

**対応図種:**
- Flowchart (`graph LR`, `graph TD`)
- Sequence Diagram
- Class Diagram (UML)
- Gantt Chart
- Pie Chart
- ER Diagram

**編集:**
挿入後、シェイプを選択してEnterを押すとコードエディタが開き、Mermaidコードを直接編集してApplyで更新できる。

### 2-B. URLパラメータでMermaidを渡す方法

`create` パラメータを使用する（要確認: 動作確認推奨）。

```
https://app.diagrams.net/?create={"type":"mermaid","data":"graph LR; A-->B"}
```

実際のURLでは `data` 部分をURI-encodeする必要がある。ただし、このパラメータが現在の版で完全にサポートされているかは要確認。

### 2-C. CLIでMermaidをdraw.ioに変換

draw.io デスクトップのCLIは直接のMermaidインポートをサポートしていない（要確認）。

代替手法:
1. **mermaid-js/mermaid-cli** でMermaid → SVGに変換
2. draw.io CLIでSVG → PNG/PDFに変換

```bash
# mermaid-cliのインストール
npm install -g @mermaid-js/mermaid-cli

# Mermaid → SVG
mmdc -i diagram.mmd -o diagram.svg

# draw.io CLIでSVGをPNGに変換（drawioファイルではないので注意）
# この方法はdraw.ioのXML編集機能を使わない純粋なSVG変換
```

3. **プログラマブルな方法**: MermaidをXMLに変換してdraw.ioで開く（要カスタム変換ロジック）

### 2-D. 各方式の比較

| 方式 | 用途 | メリット | デメリット |
|------|------|----------|------------|
| GUIインポート | 手動作業 | 簡単、視覚的に確認可 | 自動化困難 |
| URLパラメータ create | 共有・埋め込み | URLで渡せる | 要確認、長いMermaidは困難 |
| mermaid-cli変換 | 自動化 | CI/CD組込み可 | draw.io固有機能は使えない |

---

## 3. CSVインポート: フォーマット仕様詳細

### 3-A. アクセス方法

- **GUIから**: `Arrange > Insert > Advanced > CSV`
- **Webから**: draw.ioのWeb版も同じメニュー
- **CLIから**: CSVインポートの直接CLI実行はサポートされていない（GUIが必要）

### 3-B. ファイル構造

```
[ヘッダーコメント行 (# で始まる)]
[CSVデータ (ヘッダー行 + データ行)]
```

- `#` で始まる行 = 設定ディレクティブ
- `##` で始まる行 = 無視されるコメント（説明用）
- CSVデータは設定行の後に記述、`#` なし

### 3-C. 全ヘッダーディレクティブ一覧

```
## ラベル・表示
# label: %name%<br><i>%position%</i>
# label: %name% (%id%)

## シェイプスタイル（固定スタイル）
# style: rounded=1;whiteSpace=wrap;html=1;fillColor=%fill%;strokeColor=%stroke%;

## シェイプスタイル（列値で切り替え）
# stylename: type
# styles: {"server": "shape=server;fillColor=#dae8fc;", \
            "db": "shape=cylinder;fillColor=#d5e8d4;", \
            "-": "ellipse;fillColor=#fff2cc;"}

## 接続設定
# connect: {"from":"refs", "to":"id"}
# connect: {"from":"manager", "to":"name", "invert":true, \
             "label":"manages", \
             "style":"curved=1;endArrow=block;"}

## 複数の接続スタイル
# connect: [{"from":"refs","to":"id","style":"dashed=1;"}, \
             {"from":"manager","to":"id","style":"solid=1;"}]

## 名前空間（IDのプレフィックス）
# namespace: csvimport-

## シェイプサイズ
# width: 120
# height: 60
# width: auto    (テキストに合わせて自動)
# padding: 20    (autosizeの余白)

## 除外列（ダイアグラムに表示しない）
# ignore: id,image,fill,stroke,refs,manager

## ハイパーリンク列
# link: url

## レイアウト間隔
# nodespacing: 40      (ノード間の水平距離)
# levelspacing: 100    (階層間の垂直距離)
# edgespacing: 40      (エッジ間の距離)

## レイアウトアルゴリズム
# layout: auto
# layout: none            (配置しない、手動)
# layout: verticaltree    (縦ツリー)
# layout: horizontaltree  (横ツリー)
# layout: verticalflow    (縦フロー)
# layout: horizontalflow  (横フロー)
# layout: organic         (有機的配置)
# layout: circle          (円形配置)
```

### 3-D. 予約済み列名（CSVの列名として使用不可）

- `id` - シェイプの一意識別子（使用可能だが特別な意味を持つ）
- `tooltip` - ツールチップテキスト
- `placeholder` / `placeholders` - プレースホルダー
- `link` - ハイパーリンクURL
- `label` - 表示ラベル（通常は `# label:` ディレクティブで制御）

### 3-E. 完全なCSV例（組織図）

```
## 組織図の例
# label: %name%<br><i style="color:gray;">%position%</i><br><a href="mailto:%email%">Email</a>
# style: label;image=%image%;whiteSpace=wrap;html=1;rounded=1;fillColor=%fill%;strokeColor=%stroke%;
# namespace: csvimport-
# connect: {"from":"manager", "to":"name", "invert":true, "label":"manages", \
             "style":"curved=0;endArrow=block;endFill=1;"}
# width: 180
# height: 60
# nodespacing: 40
# levelspacing: 100
# layout: verticaltree
# ignore: id,fill,stroke,image,manager

name,position,id,manager,email,fill,stroke,image
Alice Chen,CEO,1,,alice@co.com,#dae8fc,#6c8ebf,https://example.com/alice.png
Bob Smith,CTO,2,Alice Chen,bob@co.com,#d5e8d4,#82b366,https://example.com/bob.png
Carol Lee,CFO,3,Alice Chen,carol@co.com,#d5e8d4,#82b366,https://example.com/carol.png
Dave Kim,Engineer,4,Bob Smith,dave@co.com,#fff2cc,#d6b656,https://example.com/dave.png
```

### 3-F. 完全なCSV例（フローチャート）

```
## フローチャートの例
# label: %step%
# style: rounded=1;whiteSpace=wrap;html=1;fillColor=%fill%;
# connect: {"from":"next", "to":"id", "style":"endArrow=block;"}
# layout: verticaltree
# ignore: fill,next

step,id,fill,next
Start,s1,#d5e8d4,s2
Fetch Data,s2,#dae8fc,"s3,s4"
Process A,s3,#fff2cc,s5
Process B,s4,#fff2cc,s5
Output,s5,#f8cecc,
```

### 3-G. 多対多の接続（refs列）

```
## refs列でカンマ区切りの複数接続
# connect: {"from":"refs", "to":"id"}

name,id,refs
Node A,A,"B,C"
Node B,B,D
Node C,C,D
Node D,D,
```

### 3-H. CLIからCSVインポートを起動する方法

CSVインポートはGUIが必要なためCLIから直接実行できない。

**代替アプローチ（Claude Code対応）:**

1. **XMLに事前変換してdraw.io CLIで開く**: PythonでCSVをdraw.io XML形式に変換してから、CLIでファイルを開く

```python
# 疑似コード: CSVからdraw.io XMLへの変換
import pandas as pd

def csv_to_drawio_xml(csv_path, output_path):
    """CSVをdraw.io XML形式に変換"""
    df = pd.read_csv(csv_path)
    # draw.io XMLフォーマットを構築
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<mxGraphModel>
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
"""
    for i, row in df.iterrows():
        xml += f'    <mxCell id="{i+2}" value="{row["name"]}" style="rounded=1;" vertex="1" parent="1">\n'
        xml += f'      <mxGeometry x="{i*200}" y="100" width="120" height="60" as="geometry"/>\n'
        xml += '    </mxCell>\n'
    xml += "  </root>\n</mxGraphModel>"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml)
```

2. **Webスクレイピング経由（非推奨）**: ブラウザオートメーションでGUIのCSVインポートを操作する

3. **draw.io APIの利用（要確認）**: draw.ioのサーバー版やAPIエンドポイントの存在を確認する

---

## 4. 各方式の総合比較

| 方式 | 自動化 | 向いている図 | Claude Code対応 | 難易度 |
|------|--------|------------|----------------|--------|
| XML直接作成 | 高 | 任意 | PowerShell/Python経由 | 高（XMLを手書き） |
| XML CLI open | 中 | 任意 | subprocess.Popen | 低 |
| XML CLI export | 高 | 任意（変換用） | subprocess.run | 低 |
| URL #U | 中 | 任意（小さい図） | webbrowser.open | 中 |
| Mermaid GUIインポート | 低 | フローチャート、UML、シーケンス | 困難（GUI必須） | 低 |
| Mermaid URL create | 中 | フローチャート | webbrowser.open | 中（要確認） |
| CSV GUIインポート | 低 | 組織図、ツリー、データ駆動 | 困難（GUI必須） | 低 |
| CSV → XML変換 | 高 | 組織図、フロー | Python自動化 | 中 |

---

## 5. Claude Code（ローカルCLI）での推奨パターン

### パターン1: XMLファイルを生成してGUIで開く

最も実用的。Pythonでdraw.io XML形式を生成し、デスクトップアプリで開く。

```python
import subprocess

# 1. XMLを生成
xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<mxGraphModel><root>
  <mxCell id="0"/>
  <mxCell id="1" parent="0"/>
  <mxCell id="2" value="Hello" style="rounded=1;" vertex="1" parent="1">
    <mxGeometry x="100" y="100" width="120" height="60" as="geometry"/>
  </mxCell>
</root></mxGraphModel>"""

with open("diagram.drawio", "w", encoding="utf-8") as f:
    f.write(xml_content)

# 2. draw.ioで開く
subprocess.Popen(
    [r"C:\Program Files\draw.io\draw.io.exe", "diagram.drawio"],
    creationflags=subprocess.DETACHED_PROCESS
)
```

### パターン2: MermaidテキストをGUIに手渡しで貼り付け

Claude Code が Mermaid テキストをクリップボードにコピーし、ユーザーがdraw.ioのGUIにペースト。

```python
import subprocess

mermaid_code = """graph LR
    A[Start] --> B[Process]
    B --> C{Decision}
    C -->|Yes| D[End]
    C -->|No| B"""

# クリップボードにコピー（Windows）
subprocess.run(["clip"], input=mermaid_code.encode("utf-8"), check=True)
print("Mermaid code copied to clipboard.")
print("Open draw.io, go to Arrange > Insert > Advanced > Mermaid, and paste.")
```

### パターン3: エクスポートのみ（ヘッドレス変換）

既存の.drawioファイルをPNG/SVGに変換する自動化。

```python
import subprocess

subprocess.run([
    r"C:\Program Files\draw.io\draw.io.exe",
    "--export", "--format", "png",
    "--output", "output.png",
    "input.drawio"
], check=True)
```

---

## 注意事項

- draw.io デスクトップのWindowsインストールパスはバージョンによって異なる場合がある（要確認: `C:\Program Files\draw.io\draw.io.exe` または `C:\Users\{username}\AppData\Local\Programs\draw.io\draw.io.exe`）
- CSVインポートのCLI直接実行は現時点では非対応。GUIを経由するかXML直接生成が必要
- Mermaid URLパラメータ（`create` パラメータ）は公式ドキュメントで言及されているが、実際の動作は要確認
- draw.io XML形式はmxGraph形式。公式仕様: https://jgraph.github.io/mxgraph/
- CSVインポートの `# connect` ディレクティブの `from` / `to` は列名を参照する（`id` 列の値ではない点に注意）
- `# stylename` + `# styles` の組み合わせで条件付きスタイルが使えるが、`-` を未定義値のフォールバックとして使う

---

## 参考資料

- [draw.io supported URL parameters](https://www.drawio.com/doc/faq/supported-url-parameters)
- [draw.io CSV insert blog](https://www.drawio.com/blog/insert-from-csv)
- [draw.io CSV FAQ](https://www.drawio.com/doc/faq/insert-from-csv)
- [draw.io Mermaid diagrams blog](https://www.drawio.com/blog/mermaid-diagrams)
- [draw.io desktop GitHub - CLI discussion #1524](https://github.com/jgraph/drawio-desktop/discussions/1524)
- [draw.io desktop GitHub - Windows detach issue #72](https://github.com/jgraph/drawio-desktop/issues/72)
- [draw.io CSV examples (GitHub)](https://github.com/jgraph/drawio-diagrams/blob/dev/examples/csv/basic.txt)
- [draw.io CSV demos blog](https://crashlaker.github.io/2021/02/07/drawio_csv_demos.html)
- [draw.io encode to URL](https://www.drawio.com/doc/faq/export-to-url)
- [draw.io CLI usage (Tom Donohue)](https://tomd.xyz/how-i-use-drawio/)
- [draw.io CSV import example txt](https://drawio-app.com/wp-content/uploads/2018/03/drawio-example-csv-import.txt)
