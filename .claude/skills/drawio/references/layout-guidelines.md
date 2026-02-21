# draw.io レイアウトガイドライン

## レイアウトパターン

### 1. 水平フロー（左→右）

**用途**: データパイプライン、API処理フロー、マイクロサービス間通信

```
[Source] → [Process A] → [Process B] → [Destination]
```

座標計算:
- 各要素の Y 座標を統一（水平ライン上に配置）
- X 間隔: 要素幅 + 80px 以上
- グループがある場合は Y をグループ中央に揃える

### 2. 垂直フロー（上→下）

**用途**: フローチャート、シーケンス図、承認フロー

```
[Start]
  ↓
[Step 1]
  ↓
[Decision] → [Branch]
  ↓
[End]
```

座標計算:
- 各要素の X 座標を統一（垂直ライン上に配置）
- Y 間隔: 要素高さ + 60px 以上
- 分岐は X 方向にオフセット（±200px）

### 3. 階層型（ネスト）

**用途**: AWSアーキテクチャ図、ネットワーク図

```
┌─ AWS Cloud ──────────────────────┐
│ ┌─ VPC ────────────────────────┐ │
│ │ ┌─ Public Subnet ──────────┐ │ │
│ │ │ [ALB] [NAT]              │ │ │
│ │ └──────────────────────────┘ │ │
│ │ ┌─ Private Subnet ─────────┐ │ │
│ │ │ [ECS] [RDS]              │ │ │
│ │ └──────────────────────────┘ │ │
│ └──────────────────────────────┘ │
└──────────────────────────────────┘
```

座標計算:
- 外層パディング: 30px
- 層間パディング: 30px
- 内側から外側へサイズを積み上げて計算

### 4. グリッド配置

**用途**: マイクロサービス一覧、コンポーネント比較

```
[A] [B] [C]
[D] [E] [F]
[G] [H] [I]
```

座標計算:
- 列間隔: 要素幅 + 40px
- 行間隔: 要素高さ + 40px
- X = col * (width + gap) + offset
- Y = row * (height + gap) + offset

### 5. ハブ＆スポーク

**用途**: API Gateway中心の構成、イベントバス

```
        [Service A]
           ↑
[Service B] ← [Hub] → [Service C]
           ↓
        [Service D]
```

座標計算:
- 中心要素: (centerX, centerY)
- スポーク要素: 中心から半径 200px の円周上に等間隔配置
- 角度 = 360° / スポーク数 × index

---

## 座標計算の基本公式

### 要素中心の計算

```
centerX = x + width / 2
centerY = y + height / 2
```

### 等間隔配置（水平）

```
x[i] = startX + i * (elementWidth + gap)
y[i] = fixedY  (すべて同じ)
```

### 等間隔配置（垂直）

```
x[i] = fixedX  (すべて同じ)
y[i] = startY + i * (elementHeight + gap)
```

### グループサイズの計算

```
groupWidth  = max(子要素のx + width) - min(子要素のx) + padding * 2
groupHeight = max(子要素のy + height) - min(子要素のy) + padding * 2 + headerHeight
```

headerHeight = グループタイトル用のスペース（通常 30px）

---

## コネクタ（矢印）スタイル

### 直角線（推奨）

```
edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;
```

### 直線

```
edgeStyle=none;html=1;
```

### 曲線

```
edgeStyle=elbowEdgeStyle;elbow=vertical;rounded=1;html=1;
```

### 破線（非同期）

```
edgeStyle=orthogonalEdgeStyle;rounded=1;dashed=1;dashPattern=8 8;html=1;
```

### 矢印の種類

| endArrow | 見た目 | 用途 |
|----------|--------|------|
| `classic` | ▶ | 通常のフロー |
| `open` | > | 軽い関連 |
| `block` | ■▶ | 強い依存 |
| `diamond` | ◆ | 集約（UML） |
| `diamondThin` | ◇ | コンポジション（UML） |
| `none` | — | 関連線のみ |

### 双方向矢印

```
startArrow=classic;endArrow=classic;
```

---

## 配色パレット

### Material Design ベース（推奨）

| 用途 | fillColor | strokeColor | fontColor |
|------|-----------|-------------|-----------|
| 白背景 | `#FFFFFF` | `#333333` | `#333333` |
| 青系（Primary） | `#BBDEFB` | `#1565C0` | `#1565C0` |
| 緑系（Success） | `#C8E6C9` | `#2E7D32` | `#2E7D32` |
| 橙系（Warning） | `#FFE0B2` | `#E65100` | `#E65100` |
| 赤系（Error） | `#FFCDD2` | `#C62828` | `#C62828` |
| 紫系（Highlight） | `#E1BEE7` | `#6A1B9A` | `#6A1B9A` |
| グレー系（Neutral） | `#F5F5F5` | `#616161` | `#616161` |

### AWS公式カラー

| カテゴリ | fillColor |
|---------|-----------|
| Compute | `#ED7100` |
| Storage | `#3F8624` |
| Database | `#C925D1` |
| Networking | `#8C4FFF` |
| Security | `#DD344C` |
| Integration | `#E7157B` |
| AI/ML | `#01A88D` |

---

## アンチパターン（避けるべきこと）

1. **矢印の交差**: 可能な限り交差を避ける。避けられない場合はブリッジ表示
2. **色の使いすぎ**: 1図あたり 5色以内。意味のある色分けのみ
3. **フォントサイズ混在**: 同一レベルの要素は同じフォントサイズ
4. **ラベルなし矢印**: 意味が自明でない矢印には必ずラベル
5. **要素の重なり**: 要素が重なると可読性が著しく低下
6. **過密配置**: 余白を十分に取る（最小間隔 40px）
