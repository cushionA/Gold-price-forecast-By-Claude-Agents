# draw.io XML テンプレート集

## 基本図形

### 長方形（角丸）

```xml
<mxCell id="rect1" value="Label" vertex="1" parent="1"
  style="rounded=1;whiteSpace=wrap;html=1;
         fillColor=#FFFFFF;strokeColor=#333333;strokeWidth=2;
         fontFamily=Helvetica;fontSize=14;fontColor=#333333;
         arcSize=10;">
  <mxGeometry x="100" y="100" width="160" height="60" as="geometry" />
</mxCell>
```

### 角丸なし長方形

```xml
<mxCell id="rect2" value="Label" vertex="1" parent="1"
  style="rounded=0;whiteSpace=wrap;html=1;
         fillColor=#FFFFFF;strokeColor=#333333;strokeWidth=2;
         fontFamily=Helvetica;fontSize=14;fontColor=#333333;">
  <mxGeometry x="100" y="100" width="160" height="60" as="geometry" />
</mxCell>
```

### 円

```xml
<mxCell id="circle1" value="Label" vertex="1" parent="1"
  style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;
         fillColor=#FFFFFF;strokeColor=#333333;strokeWidth=2;
         fontFamily=Helvetica;fontSize=14;fontColor=#333333;">
  <mxGeometry x="100" y="100" width="80" height="80" as="geometry" />
</mxCell>
```

### ひし形（判断）

```xml
<mxCell id="diamond1" value="Yes/No?" vertex="1" parent="1"
  style="rhombus;whiteSpace=wrap;html=1;
         fillColor=#FFF9C4;strokeColor=#F57F17;strokeWidth=2;
         fontFamily=Helvetica;fontSize=14;fontColor=#333333;">
  <mxGeometry x="100" y="100" width="120" height="80" as="geometry" />
</mxCell>
```

### 平行四辺形（入出力）

```xml
<mxCell id="para1" value="Input/Output" vertex="1" parent="1"
  style="shape=parallelogram;perimeter=parallelogramPerimeter;
         whiteSpace=wrap;html=1;fixedSize=1;
         fillColor=#E3F2FD;strokeColor=#1565C0;strokeWidth=2;
         fontFamily=Helvetica;fontSize=14;fontColor=#333333;">
  <mxGeometry x="100" y="100" width="160" height="60" as="geometry" />
</mxCell>
```

### シリンダー（データベース）

```xml
<mxCell id="db1" value="Database" vertex="1" parent="1"
  style="shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;
         backgroundOutline=1;size=15;
         fillColor=#E1BEE7;strokeColor=#6A1B9A;strokeWidth=2;
         fontFamily=Helvetica;fontSize=14;fontColor=#333333;">
  <mxGeometry x="100" y="100" width="100" height="80" as="geometry" />
</mxCell>
```

### ドキュメント

```xml
<mxCell id="doc1" value="Document" vertex="1" parent="1"
  style="shape=document;whiteSpace=wrap;html=1;boundedLbl=1;
         backgroundOutline=1;size=0.27;
         fillColor=#FFFFFF;strokeColor=#333333;strokeWidth=2;
         fontFamily=Helvetica;fontSize=14;fontColor=#333333;">
  <mxGeometry x="100" y="100" width="120" height="80" as="geometry" />
</mxCell>
```

---

## コネクタ（矢印）

### 標準矢印（直角・ラベル付き）

```xml
<mxCell id="edge1" value="label" edge="1" source="node1" target="node2" parent="1"
  style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;
         jettySize=auto;html=1;strokeWidth=2;strokeColor=#333333;
         fontFamily=Helvetica;fontSize=11;fontColor=#666666;
         endArrow=classic;endFill=1;">
  <mxGeometry relative="1" as="geometry" />
</mxCell>
```

### 破線矢印（非同期）

```xml
<mxCell id="edge2" value="async" edge="1" source="node1" target="node2" parent="1"
  style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;
         jettySize=auto;html=1;strokeWidth=2;strokeColor=#666666;
         dashed=1;dashPattern=8 8;
         fontFamily=Helvetica;fontSize=11;fontColor=#666666;
         endArrow=open;endFill=0;">
  <mxGeometry relative="1" as="geometry" />
</mxCell>
```

### 直線矢印

```xml
<mxCell id="edge3" value="" edge="1" source="node1" target="node2" parent="1"
  style="html=1;strokeWidth=2;strokeColor=#333333;
         endArrow=classic;endFill=1;">
  <mxGeometry relative="1" as="geometry" />
</mxCell>
```

### 双方向矢印

```xml
<mxCell id="edge4" value="" edge="1" source="node1" target="node2" parent="1"
  style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;
         strokeWidth=2;strokeColor=#333333;
         startArrow=classic;startFill=1;
         endArrow=classic;endFill=1;">
  <mxGeometry relative="1" as="geometry" />
</mxCell>
```

---

## グループ（コンテナ）

### 標準グループ

```xml
<mxCell id="group1" value="Group Title" vertex="1" parent="1"
  style="rounded=1;whiteSpace=wrap;html=1;
         container=1;collapsible=0;recursiveResize=0;
         fillColor=#F5F5F5;strokeColor=#666666;strokeWidth=2;
         fontFamily=Helvetica;fontSize=14;fontStyle=1;fontColor=#333333;
         verticalAlign=top;align=left;spacingTop=10;spacingLeft=10;">
  <mxGeometry x="50" y="50" width="400" height="300" as="geometry" />
</mxCell>
```

### 破線グループ（論理境界）

```xml
<mxCell id="group2" value="Logical Boundary" vertex="1" parent="1"
  style="rounded=1;whiteSpace=wrap;html=1;
         container=1;collapsible=0;recursiveResize=0;
         fillColor=none;strokeColor=#999999;strokeWidth=2;
         dashed=1;dashPattern=8 8;
         fontFamily=Helvetica;fontSize=13;fontStyle=2;fontColor=#999999;
         verticalAlign=top;align=left;spacingTop=10;spacingLeft=10;">
  <mxGeometry x="50" y="50" width="400" height="300" as="geometry" />
</mxCell>
```

---

## 完全なテンプレート例

### フローチャート（最小構成）

```xml
<mxfile host="app.diagrams.net" modified="2026-01-01T00:00:00.000Z"
        version="24.0.0" type="device">
  <diagram name="Flowchart" id="flowchart1">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10"
                  guides="1" tooltips="1" connect="1" arrows="1"
                  fold="1" page="1" pageScale="1" pageWidth="1169"
                  pageHeight="827" math="0" shadow="0"
                  background="none" defaultFontFamily="Helvetica">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- Edges first -->
        <mxCell id="e1" edge="1" source="start" target="step1" parent="1"
          style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;
                 jettySize=auto;html=1;strokeWidth=2;strokeColor=#333333;
                 fontFamily=Helvetica;fontSize=11;endArrow=classic;endFill=1;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e2" edge="1" source="step1" target="decision" parent="1"
          style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;
                 jettySize=auto;html=1;strokeWidth=2;strokeColor=#333333;
                 fontFamily=Helvetica;fontSize=11;endArrow=classic;endFill=1;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e3" value="Yes" edge="1" source="decision" target="step2" parent="1"
          style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;
                 jettySize=auto;html=1;strokeWidth=2;strokeColor=#2E7D32;
                 fontFamily=Helvetica;fontSize=11;fontColor=#2E7D32;
                 endArrow=classic;endFill=1;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e4" value="No" edge="1" source="decision" target="end" parent="1"
          style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;
                 jettySize=auto;html=1;strokeWidth=2;strokeColor=#C62828;
                 fontFamily=Helvetica;fontSize=11;fontColor=#C62828;
                 endArrow=classic;endFill=1;exitX=1;exitY=0.5;exitDx=0;exitDy=0;">
          <mxGeometry relative="1" as="geometry">
            <Array as="points">
              <mxPoint x="560" y="290" />
              <mxPoint x="560" y="490" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="e5" edge="1" source="step2" target="end" parent="1"
          style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;
                 jettySize=auto;html=1;strokeWidth=2;strokeColor=#333333;
                 fontFamily=Helvetica;fontSize=11;endArrow=classic;endFill=1;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- Nodes -->
        <mxCell id="start" value="Start" vertex="1" parent="1"
          style="rounded=1;whiteSpace=wrap;html=1;arcSize=50;
                 fillColor=#C8E6C9;strokeColor=#2E7D32;strokeWidth=2;
                 fontFamily=Helvetica;fontSize=14;fontColor=#333333;">
          <mxGeometry x="340" y="60" width="120" height="40" as="geometry" />
        </mxCell>
        <mxCell id="step1" value="Process A" vertex="1" parent="1"
          style="rounded=1;whiteSpace=wrap;html=1;
                 fillColor=#FFFFFF;strokeColor=#333333;strokeWidth=2;
                 fontFamily=Helvetica;fontSize=14;fontColor=#333333;">
          <mxGeometry x="320" y="160" width="160" height="60" as="geometry" />
        </mxCell>
        <mxCell id="decision" value="Condition?" vertex="1" parent="1"
          style="rhombus;whiteSpace=wrap;html=1;
                 fillColor=#FFF9C4;strokeColor=#F57F17;strokeWidth=2;
                 fontFamily=Helvetica;fontSize=14;fontColor=#333333;">
          <mxGeometry x="340" y="260" width="120" height="60" as="geometry" />
        </mxCell>
        <mxCell id="step2" value="Process B" vertex="1" parent="1"
          style="rounded=1;whiteSpace=wrap;html=1;
                 fillColor=#FFFFFF;strokeColor=#333333;strokeWidth=2;
                 fontFamily=Helvetica;fontSize=14;fontColor=#333333;">
          <mxGeometry x="320" y="380" width="160" height="60" as="geometry" />
        </mxCell>
        <mxCell id="end" value="End" vertex="1" parent="1"
          style="rounded=1;whiteSpace=wrap;html=1;arcSize=50;
                 fillColor=#FFCDD2;strokeColor=#C62828;strokeWidth=2;
                 fontFamily=Helvetica;fontSize=14;fontColor=#333333;">
          <mxGeometry x="340" y="470" width="120" height="40" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

### AWS構成図（最小構成）

```xml
<mxfile host="app.diagrams.net" modified="2026-01-01T00:00:00.000Z"
        version="24.0.0" type="device">
  <diagram name="AWS Architecture" id="aws1">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10"
                  guides="1" tooltips="1" connect="1" arrows="1"
                  fold="1" page="1" pageScale="1" pageWidth="1169"
                  pageHeight="827" math="0" shadow="0"
                  background="none" defaultFontFamily="Helvetica">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- Edges first -->
        <mxCell id="e_user_cf" edge="1" source="user" target="cf" parent="1"
          style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;
                 strokeWidth=2;strokeColor=#333333;
                 fontFamily=Helvetica;fontSize=11;
                 endArrow=classic;endFill=1;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e_cf_apigw" edge="1" source="cf" target="apigw" parent="1"
          style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;
                 strokeWidth=2;strokeColor=#333333;
                 fontFamily=Helvetica;fontSize=11;
                 endArrow=classic;endFill=1;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e_apigw_lambda" edge="1" source="apigw" target="lambda" parent="1"
          style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;
                 strokeWidth=2;strokeColor=#333333;
                 fontFamily=Helvetica;fontSize=11;
                 endArrow=classic;endFill=1;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e_lambda_dynamo" edge="1" source="lambda" target="dynamo" parent="1"
          style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;
                 strokeWidth=2;strokeColor=#333333;
                 fontFamily=Helvetica;fontSize=11;
                 endArrow=classic;endFill=1;">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <!-- AWS Cloud group -->
        <mxCell id="cloud" value="AWS Cloud" vertex="1" parent="1"
          style="points=[[0,0],[0.25,0],[0.5,0],[0.75,0],[1,0],[1,0.25],[1,0.5],[1,0.75],[1,1],[0.75,1],[0.5,1],[0.25,1],[0,1],[0,0.75],[0,0.5],[0,0.25]];
                 outlineConnect=0;gradientColor=none;html=1;whiteSpace=wrap;
                 fontSize=14;fontStyle=1;fontFamily=Helvetica;
                 container=1;pointerEvents=0;collapsible=0;recursiveResize=0;
                 shape=mxgraph.aws4.group;grIcon=mxgraph.aws4.group_aws_cloud;
                 strokeColor=#242F3E;fillColor=none;
                 verticalAlign=top;align=left;spacingLeft=30;fontColor=#242F3E;">
          <mxGeometry x="200" y="50" width="700" height="400" as="geometry" />
        </mxCell>

        <!-- User (outside cloud) -->
        <mxCell id="user" value="User" vertex="1" parent="1"
          style="shape=mxgraph.aws4.user;outlineConnect=0;
                 fontColor=#232F3E;sketch=0;
                 fontFamily=Helvetica;fontSize=12;
                 verticalLabelPosition=bottom;verticalAlign=top;align=center;"
          >
          <mxGeometry x="60" y="200" width="60" height="60" as="geometry" />
        </mxCell>

        <!-- Services inside cloud -->
        <mxCell id="cf" value="CloudFront" vertex="1" parent="cloud"
          style="sketch=0;outlineConnect=0;fontColor=#232F3E;
                 gradientColor=none;fillColor=#8C4FFF;strokeColor=#ffffff;
                 fontFamily=Helvetica;fontSize=12;
                 verticalLabelPosition=bottom;verticalAlign=top;align=center;
                 html=1;aspect=fixed;
                 shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.cloudfront;">
          <mxGeometry x="40" y="140" width="78" height="78" as="geometry" />
        </mxCell>
        <mxCell id="apigw" value="API Gateway" vertex="1" parent="cloud"
          style="sketch=0;outlineConnect=0;fontColor=#232F3E;
                 gradientColor=none;fillColor=#8C4FFF;strokeColor=#ffffff;
                 fontFamily=Helvetica;fontSize=12;
                 verticalLabelPosition=bottom;verticalAlign=top;align=center;
                 html=1;aspect=fixed;
                 shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.api_gateway;">
          <mxGeometry x="200" y="140" width="78" height="78" as="geometry" />
        </mxCell>
        <mxCell id="lambda" value="Lambda" vertex="1" parent="cloud"
          style="sketch=0;outlineConnect=0;fontColor=#232F3E;
                 gradientColor=none;fillColor=#ED7100;strokeColor=#ffffff;
                 fontFamily=Helvetica;fontSize=12;
                 verticalLabelPosition=bottom;verticalAlign=top;align=center;
                 html=1;aspect=fixed;
                 shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.lambda;">
          <mxGeometry x="380" y="140" width="78" height="78" as="geometry" />
        </mxCell>
        <mxCell id="dynamo" value="DynamoDB" vertex="1" parent="cloud"
          style="sketch=0;outlineConnect=0;fontColor=#232F3E;
                 gradientColor=none;fillColor=#C925D1;strokeColor=#ffffff;
                 fontFamily=Helvetica;fontSize=12;
                 verticalLabelPosition=bottom;verticalAlign=top;align=center;
                 html=1;aspect=fixed;
                 shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.dynamodb;">
          <mxGeometry x="560" y="140" width="78" height="78" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

### 凡例（Legend）テンプレート

```xml
<!-- 凡例グループ -->
<mxCell id="legend" value="Legend" vertex="1" parent="1"
  style="rounded=1;whiteSpace=wrap;html=1;
         container=1;collapsible=0;
         fillColor=#FAFAFA;strokeColor=#BDBDBD;strokeWidth=1;
         fontFamily=Helvetica;fontSize=12;fontStyle=1;fontColor=#333333;
         verticalAlign=top;align=left;spacingTop=8;spacingLeft=8;">
  <mxGeometry x="900" y="600" width="220" height="120" as="geometry" />
</mxCell>

<!-- 凡例アイテム: 実線 -->
<mxCell id="leg1_line" edge="1" parent="legend"
  style="html=1;strokeWidth=2;strokeColor=#333333;endArrow=classic;endFill=1;
         fontFamily=Helvetica;fontSize=11;">
  <mxGeometry relative="1" as="geometry">
    <mxPoint x="10" y="45" as="sourcePoint" />
    <mxPoint x="60" y="45" as="targetPoint" />
  </mxGeometry>
</mxCell>
<mxCell id="leg1_label" value="Synchronous" vertex="1" parent="legend"
  style="text;html=1;fontFamily=Helvetica;fontSize=11;fontColor=#333333;
         align=left;verticalAlign=middle;">
  <mxGeometry x="70" y="35" width="130" height="20" as="geometry" />
</mxCell>

<!-- 凡例アイテム: 破線 -->
<mxCell id="leg2_line" edge="1" parent="legend"
  style="html=1;strokeWidth=2;strokeColor=#666666;
         dashed=1;dashPattern=8 8;endArrow=open;endFill=0;
         fontFamily=Helvetica;fontSize=11;">
  <mxGeometry relative="1" as="geometry">
    <mxPoint x="10" y="75" as="sourcePoint" />
    <mxPoint x="60" y="75" as="targetPoint" />
  </mxGeometry>
</mxCell>
<mxCell id="leg2_label" value="Asynchronous" vertex="1" parent="legend"
  style="text;html=1;fontFamily=Helvetica;fontSize=11;fontColor=#333333;
         align=left;verticalAlign=middle;">
  <mxGeometry x="70" y="65" width="130" height="20" as="geometry" />
</mxCell>
```

### テキストラベル（独立）

```xml
<mxCell id="label1" value="Title Text" vertex="1" parent="1"
  style="text;html=1;strokeColor=none;fillColor=none;
         fontFamily=Helvetica;fontSize=18;fontStyle=1;fontColor=#333333;
         align=center;verticalAlign=middle;">
  <mxGeometry x="400" y="10" width="200" height="30" as="geometry" />
</mxCell>
```

### 注釈（ノート）

```xml
<mxCell id="note1" value="Note: This is important" vertex="1" parent="1"
  style="shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;size=15;
         fillColor=#FFF9C4;strokeColor=#F57F17;strokeWidth=1;
         fontFamily=Helvetica;fontSize=11;fontColor=#333333;
         align=left;verticalAlign=top;spacingTop=5;spacingLeft=5;">
  <mxGeometry x="600" y="400" width="180" height="60" as="geometry" />
</mxCell>
```
