# リサーチレポート: 日次金利データの網羅的調査

## 調査目的

金価格予測モデルにおける実質金利サブモデルの構築で、多国間データを活用するために**日次頻度で取得可能な金利関連データ**を調査する。OECD月次データ(IRLTLT01xxM156N等)は月次のみで日次予測に使えなかったため、各国の日次データの利用可能性を明らかにする。

## エグゼクティブサマリー

**重要な発見:**
1. **FRED経由のOECDデータは月次のみ**: 105系列すべてが月次頻度(FREQ: M)であり、日次データは存在しない
2. **yfinanceで日次取得可能**: 米国(^TNX)のみ確実。他国は限定的またはティッカーが不明確
3. **各国中央銀行APIが最有力**: BOJ、BOE、ECBは日次データを公式提供
4. **インフレ連動債は限定的**: ドイツ2024年終了、カナダ2022年終了。米国TIPSのみ実用的

**推奨戦略:**
- **米国**: FRED API (DGS10, DFII10等) — 最も信頼性が高い
- **日本・英国・ユーロ圏**: 各国中央銀行API — 公式データで確実
- **その他G10**: yfinance試行 → 失敗時は月次OECDデータで補完

---

## 1. FRED: 日次金利系列の利用可能性

### 1.1 米国データ(日次確認済み)

| シリーズID | 内容 | 頻度 | 開始年 | 取得方法 |
|-----------|------|------|--------|----------|
| **DGS10** | 10年国債利回り(名目) | **日次** | 1962 | FRED API / fredapi |
| **DFII10** | 10年TIPS利回り(実質) | **日次** | 2003 | FRED API / fredapi |
| **DGS2** | 2年国債利回り | **日次** | 1976 | FRED API / fredapi |
| **T10YIE** | 10年期待インフレ率(Breakeven) | **日次** | 2003 | FRED API / fredapi |
| DTP30A29 | 30年TIPS(満期2029年) | 日次 | 1999 | FRED API |

**取得コード例:**
```python
from fredapi import Fred
import os

fred = Fred(api_key=os.getenv('FRED_API_KEY'))

# 日次データ
dgs10 = fred.get_series('DGS10', observation_start='2003-01-01')
dfii10 = fred.get_series('DFII10', observation_start='2003-01-01')
t10yie = fred.get_series('T10YIE', observation_start='2003-01-01')
```

### 1.2 非米国データ(月次のみ)

**重要**: FREDで提供される全105系列のOECD金利データは**すべて月次頻度**(FREQ: M)

| 国/地域 | シリーズID | 頻度 | 開始年 | 最終更新 |
|---------|-----------|------|--------|----------|
| **ドイツ** | IRLTLT01DEM156N | **月次** | 1956 | 2025-12 |
| **日本** | IRLTLT01JPM156N | **月次** | 1989 | 2025-12 |
| **英国** | IRLTLT01GBM156N | **月次** | 1960 | 2026-01 |
| **カナダ** | IRLTLT01CAM156N | **月次** | - | - |
| **豪州** | IRLTLT01AUM156N | **月次** | - | - |
| **ユーロ圏(19国)** | IRLTLT01EZM156N | **月次** | 1953 | 2026-01 |

**結論**: **FRED経由では米国以外の日次データは取得不可**。月次データのみ。

---

## 2. yfinance: 国債利回りティッカー

### 2.1 確実に動作するティッカー

| 国 | ティッカー | 内容 | 頻度 | 信頼性 |
|------|----------|------|------|--------|
| 🇺🇸 米国 | **^TNX** | 10年国債利回り(CBOE) | 日次 | ⭐⭐⭐ 最高 |
| 🇺🇸 米国 | ^FVX | 5年国債利回り | 日次 | ⭐⭐⭐ 最高 |
| 🇺🇸 米国 | ^TYX | 30年国債利回り | 日次 | ⭐⭐⭐ 最高 |

### 2.2 他国ティッカー(要検証)

| 国 | 推定ティッカー | 検証状態 | 備考 |
|------|--------------|----------|------|
| 🇯🇵 日本 | ^JGB, JP10Y=XX | ⚠️ 要確認 | yfinanceドキュメントに明記なし |
| 🇩🇪 ドイツ | ^DE10YB | ⚠️ 要確認 | Yahoo Financeページ上は存在 |
| 🇬🇧 英国 | ^GB10Y | ⚠️ 要確認 | Yahoo Financeページ上は存在 |
| 🇨🇦 カナダ | ^CAGG10Y | ⚠️ 要確認 | Yahoo Financeページ上は存在 |

**取得コード例:**
```python
import yfinance as yf

# 米国(確実)
us_10y = yf.download("^TNX", start="2003-01-01", interval="1d")

# 他国(試行的)
tickers = ["^TNX", "^JGB", "^DE10YB", "^GB10Y"]
data = yf.download(tickers, start="2010-01-01", interval="1d")

# データ欠損率確認
print(data.isnull().sum())
```

**注意**: yfinanceのドキュメントには非米国の国債利回りティッカーが明記されていない。実際の利用前に必ずデータ品質を検証すること。

---

## 3. 各国中央銀行API(日次データ公式提供)

### 3.1 日本銀行(BOJ)

**公式データポータル:**
- **BOJ Time-Series Data Search**: [stat-search.boj.or.jp](https://www.stat-search.boj.or.jp/index_en.html)
- **提供形式**: Flat files(CSV/Excel), 日次公開 ~10:00 AM
- **利用可能データ**: JGB利回り(全満期)、政策金利、貸出金利
- **API**: 限定的(主にダウンロード形式)

**現在の利回り(2026-02-17)**: 10年JGB = 2.18%(4週で-8.81bps)

**FRED経由(月次のみ):**
```python
# FREDは月次のみ
jpy_10y_monthly = fred.get_series('IRLTLT01JPM156N')
```

**BOJ直接取得(推奨):**
1. [stat-search.boj.or.jp](https://www.stat-search.boj.or.jp/index_en.html)にアクセス
2. "Interest Rates" → "JGB Yields" → "Daily"を選択
3. CSV形式でダウンロード(認証不要)

**代替**: [bb.jbts.co.jp](https://www.bb.jbts.co.jp/en/historical/yieldcurve.html) - Japan Bond Trading Co.のJGB Yield Curve(非線形回帰モデル)

### 3.2 イングランド銀行(BOE)

**公式データポータル:**
- **BOE Yield Curves**: [bankofengland.co.uk/statistics/yield-curves](https://www.bankofengland.co.uk/statistics/yield-curves)
- **BOE Database**: [bankofengland.co.uk/boeapps/database](https://www.bankofengland.co.uk/boeapps/database/)
- **提供形式**: CSV/Excel, 日次更新(翌営業日正午まで)
- **曲線タイプ**: ゼロクーポン名目/実質曲線、期待インフレ曲線

**公開スケジュール**: 翌営業日正午までに公開(月初第2営業日までにアーカイブ更新)

**利用可能データ:**
- Zero coupon nominal/real curves
- Implied forward nominal/real rates
- Index-linked gilt yields(RPI連動, 3ヶ月ラグ)

**取得方法:**
1. BOE Databaseで系列コード検索(例: "10-year spot rate")
2. Export as CSV
3. 最終更新: 2026-01-26

**DMOデータ:** [dmo.gov.uk/data/gilt-market](https://www.dmo.gov.uk/data/gilt-market/) - 日次価格・利回りデータ

### 3.3 欧州中央銀行(ECB)

**公式データポータル:**
- **ECB Data Portal**: [data.ecb.europa.eu](https://data.ecb.europa.eu) (旧SDW)
- **Euro Area Yield Curves**: [ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves](https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html)
- **API**: `https://data-api.ecb.europa.eu`
- **提供形式**: CSV/Excel, 日次更新(TARGET営業日正午)

**データ範囲**: 2004-09-06以降の日次データ

**具体的系列:**
- **10年スポットレート**: YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y
- **全AAA格付け国債含む**: 第2データセット(All central government bonds)

**現在の政策金利(2026年1月):**
- Main refinancing rate: 2.15%
- Deposit facility: 2.00%
- Marginal lending: 2.40%

**取得コード例:**
```python
import requests
import pandas as pd

# ECB Data Portal API
url = "https://data-api.ecb.europa.eu/service/data/YC/YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y?format=csvdata"
response = requests.get(url)
df = pd.read_csv(io.StringIO(response.text))
```

**重要**: 2026年1月にブルガリアがユーロ採用。2026年以降の全系列はブルガリア含む19国。

---

## 4. インフレ連動債(TIPS相当)の利用可能性

### 4.1 各国の状況

| 国 | プログラム名 | 状況 | 開始年 | 終了年 | ラグ | 指数 |
|------|-------------|------|--------|--------|------|------|
| 🇺🇸 米国 | TIPS | ✅ **継続中** | 1997 | - | 3ヶ月 | CPI-U |
| 🇬🇧 英国 | Index-linked Gilts | ✅ **継続中** | 1981 | - | 3ヶ月(新型) | RPI |
| 🇯🇵 日本 | JGBi | ⚠️ **断続的** | 2004 | - | - | CPI |
| 🇩🇪 ドイツ | Inflation-linked Bunds | ❌ **2024年終了** | - | 2024 | - | HICP |
| 🇨🇦 カナダ | Real Return Bonds | ❌ **2022年終了** | - | 2022 | - | CPI |
| 🇦🇺 豪州 | Inflation-linked Bonds | ⚠️ **限定的** | 1985 | - | - | CPI |

### 4.2 米国TIPS(最も実用的)

**FREDシリーズ(日次):**
- **DFII10**: 10年TIPS利回り(実質金利)
- **DFII5**: 5年TIPS利回り
- **DFII20**: 20年TIPS利回り
- **DFII30**: 30年TIPS利回り

**FRED Category 82**: 177系列のTIPS関連データ

**取得コード:**
```python
# 米国実質金利(日次)
dfii10 = fred.get_series('DFII10', observation_start='2003-01-01')

# 期待インフレ率(名目-実質)
dgs10 = fred.get_series('DGS10')
breakeven = dgs10 - dfii10  # または直接T10YIE
```

### 4.3 英国Index-linked Gilts

**BOE Yield Curves:**
- Zero coupon real curves(日次)
- Implied forward real rates(日次)
- Index ratio公開(日次更新)

**設計:**
- 2005年以降: 3ヶ月ラグ(RPIベース)
- 2005年以前: 8ヶ月ラグ

**取得:** BOE Database経由でCSV export

### 4.4 日本JGBi(断続的発行)

**状況:** 2004年開始だが断続的。流動性低い。

**データ取得:** BOJ Time-Series Searchで"Inflation-Indexed JGB"検索

**推奨度:** 低(サンプル不足、欠損期間あり)

### 4.5 結論: TIPSのみ実用的

**日次金利データで多国間データを活用する場合の制約:**
- インフレ連動債は米国TIPSのみ安定した日次データあり
- 他国は終了済み(独・加)または断続的(日)
- **実質金利の多国間拡張は困難**

**代替戦略:**
```
実質金利 = 名目金利 - 期待インフレ率
```
- 名目金利: 各国中央銀行APIで日次取得可能
- 期待インフレ率: 月次OECDデータ + 補完(スプライン/前月繰越)

---

## 5. その他無料API(グローバルデータ)

### 5.1 Nasdaq Data Link(旧Quandl)

**URL**: [data.nasdaq.com](https://data.nasdaq.com)

**カバレッジ:**
- 400以上のソースから数百万系列
- 金利データ: FF金利、貸出金利、国債利回り、社債利回り等
- グローバル: Bundesbank(ドイツ), UKONS(英国統計局)含む

**料金:**
- Free tier: 限定的(API call制限あり)
- Paid: $49/月〜

**取得コード例:**
```python
import nasdaqdatalink
nasdaqdatalink.ApiConfig.api_key = 'YOUR_API_KEY'

# ドイツMFI金利(Bundesbankデータ)
de_rate = nasdaqdatalink.get('BUNDESBANK/BBK01_SU0200')
```

### 5.2 Trading Economics

**URL**: [tradingeconomics.com](https://tradingeconomics.com)

**カバレッジ:** 196ヶ国の経済データ(金利含む)

**頻度:** 日次株価、為替、商品価格

**料金:**
- Free tier: 限定的(グラフ閲覧のみ)
- API: 有料のみ

**推奨度:** 中(グローバルカバレッジは最強だがAPI有料)

### 5.3 Alpha Vantage

**URL**: [alphavantage.co](https://www.alphavantage.co)

**カバレッジ:** 主に米国、限定的グローバル

**料金:**
- Free tier: レート制限厳しい(5 API calls/分, 100 calls/日)
- Premium: $49.99/月〜

**推奨度:** 低(無料tierでは実用困難)

### 5.4 EODHD APIs

**URL**: [eodhistoricaldata.com](https://eodhistoricaldata.com)

**カバレッジ:** 150,000+ティッカー(グローバル)

**料金:** $19.99/月〜(EOD onlyプラン)

**推奨度:** 中(コスパ良いが株式中心)

---

## 6. 利用可能な日次金利データの総括

### 6.1 国別利用可能性マトリクス

| 国/地域 | 名目10年 | 実質10年(TIPS相当) | 短期金利 | データソース(日次) | 信頼性 |
|---------|---------|-------------------|----------|-------------------|--------|
| 🇺🇸 米国 | ✅ | ✅ | ✅ | FRED API (DGS10, DFII10, DGS2) | ⭐⭐⭐ |
| 🇯🇵 日本 | ✅ | ⚠️ | ✅ | BOJ Portal + FRED月次 | ⭐⭐ |
| 🇬🇧 英国 | ✅ | ✅ | ✅ | BOE API/Database | ⭐⭐⭐ |
| 🇩🇪 ドイツ | ⚠️ | ❌ | ⚠️ | ECB(ユーロ圏), yfinance試行 | ⭐⭐ |
| ユーロ圏 | ✅ | ❌ | ✅ | ECB Data Portal(日次) | ⭐⭐⭐ |
| 🇨🇦 カナダ | ⚠️ | ❌ | ⚠️ | yfinance試行 + FRED月次 | ⭐ |
| 🇦🇺 豪州 | ⚠️ | ⚠️ | ⚠️ | FRED月次のみ | ⭐ |

**凡例:**
- ✅ = 日次データあり、公式API利用可能
- ⚠️ = yfinanceで試行可能だが未検証、または断続的
- ❌ = 日次データなし

### 6.2 データ取得の推奨優先順位

#### Tier 1: 確実(公式API、日次確認済み)
1. **米国**: FRED API (`DGS10`, `DFII10`, `DGS2`, `T10YIE`)
2. **英国**: BOE Database/Yield Curves API
3. **ユーロ圏**: ECB Data Portal API
4. **日本**: BOJ Time-Series Portal

#### Tier 2: 試行可能(非公式だが可能性あり)
5. **ドイツ**: yfinance `^DE10YB` → 失敗時はECBユーロ圏データ
6. **カナダ**: yfinance `^CAGG10Y` → 失敗時はFRED月次

#### Tier 3: 月次補完
7. **その他G10**: FRED OECD月次データ(IRLTLT01xxM156N) + 補完手法
   - スプライン補完
   - 前月繰越(step function)
   - 他国相関利用した推定

### 6.3 多国間データ利用の現実的戦略

**realityチェック:**
- FRED経由のOECDデータは**すべて月次**
- yfinanceの非米国ティッカーは**未検証**
- 各国中央銀行APIは存在するが**統一インターフェースなし**

**推奨アプローチ(現実的):**

**Option A: 主要国のみ(日次確実)**
```
米国(FRED) + 英国(BOE) + ユーロ圏(ECB) + 日本(BOJ)
= 4経済圏 × 15-20年 = ~15,000-20,000サンプル
```

**Option B: 米国単一(最高品質)**
```
米国のみ(FRED全系列)
- 10年名目(DGS10)
- 10年実質(DFII10)
- 2年(DGS2)
- 期待インフレ(T10YIE)
- 曲線スロープ(DGS10-DGS2)
= 1国 × 20年 × 5特徴量 = 高品質だが少サンプル
```

**Option C: ハイブリッド(日次 + 月次補完)**
```
日次Tier1(米英日欧) + 月次補完Tier2(独加豪)
月次→日次変換手法:
1. Cubic spline interpolation
2. Kalman filter smoothing
3. Forward-fill with decay
```

---

## 7. 具体的取得コード例

### 7.1 米国データ(FRED API - 最推奨)

```python
import os
from fredapi import Fred
import pandas as pd

fred = Fred(api_key=os.getenv('FRED_API_KEY'))

# 日次金利データ
series_ids = {
    'dgs10': 'DGS10',      # 10年名目
    'dfii10': 'DFII10',    # 10年実質
    'dgs2': 'DGS2',        # 2年名目
    't10yie': 'T10YIE',    # 期待インフレ
}

data = {}
for name, series_id in series_ids.items():
    data[name] = fred.get_series(
        series_id,
        observation_start='2003-01-01'
    )

df_us = pd.DataFrame(data)
df_us['slope'] = df_us['dgs10'] - df_us['dgs2']  # Yield curve slope
df_us.to_csv('data/raw/us_rates_daily.csv')
```

### 7.2 多国間データ(日次tier1のみ)

```python
import yfinance as yf
from fredapi import Fred

fred = Fred(api_key=os.getenv('FRED_API_KEY'))

# 米国(FRED)
us_10y = fred.get_series('DGS10', observation_start='2010-01-01')

# 試行: 他国(yfinance - 要検証)
tickers = {
    'us': '^TNX',
    'jp': '^JGB',
    'uk': '^GB10Y',
    'de': '^DE10YB',
}

yf_data = yf.download(
    list(tickers.values()),
    start='2010-01-01',
    interval='1d'
)['Close']
yf_data.columns = list(tickers.keys())

# データ品質チェック
print("欠損率:")
print(yf_data.isnull().sum() / len(yf_data))

# 欠損率 > 50%の系列は除外
valid_cols = yf_data.columns[yf_data.isnull().sum() / len(yf_data) < 0.5]
yf_data_clean = yf_data[valid_cols]

yf_data_clean.to_csv('data/raw/multi_country_rates_yfinance.csv')
```

### 7.3 BOJ/BOE/ECB直接取得(要手動ダウンロード)

```python
# BOJ(手動ダウンロード後)
# https://www.stat-search.boj.or.jp/index_en.html
# Interest Rates → JGB Yields → Daily → Download CSV

boj_df = pd.read_csv('data/raw/boj_jgb_yields_daily.csv')

# BOE(手動ダウンロード後)
# https://www.bankofengland.co.uk/boeapps/database/
# Search: "10-year spot rate" → Export CSV

boe_df = pd.read_csv('data/raw/boe_gilt_yields_daily.csv')

# ECB(API)
import requests
import io

url = "https://data-api.ecb.europa.eu/service/data/YC/YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y?format=csvdata"
response = requests.get(url)
ecb_df = pd.read_csv(io.StringIO(response.text))
```

---

## 8. 設計への推奨事項

### 8.1 architect向け推奨

**現実的な多国間戦略:**

1. **Tier 1のみ使用(最推奨)**
   - 米国: FRED API(5系列 × 23年 = ~6,000サンプル/系列)
   - 英国: BOE API(1系列 × 20年 = ~5,000サンプル)
   - 日本: BOJ Portal(1系列 × 17年 = ~4,250サンプル)
   - ユーロ圏: ECB API(1系列 × 21年 = ~5,250サンプル)
   - **合計**: 4経済圏 × ~5,000サンプル = **~20,000サンプル**

2. **米国単一(最高品質)**
   - FRED 5系列 × 23年 = ~6,000サンプル/系列
   - 実質金利直接利用可能(DFII10)
   - VIF懸念なし(別国のため)

3. **ハイブリッド(非推奨 - 複雑性高い)**
   - Tier1(日次) + FRED月次(IRLTLT01xxM156N) + スプライン補完
   - 補完誤差がノイズ源になるリスク

**推奨手法:**
- **Option 1**(最推奨): 米国FRED 5系列のみ → GRU/Transformer
- **Option 2**: Tier1 4経済圏 → Multi-country Transformer(attempt 3の再利用)

### 8.2 builder_data向け注意事項

**データ検証必須項目:**
1. yfinanceティッカーの欠損率(>50%なら除外)
2. BOJ/BOE/ECBデータのタイムゾーン統一(UTCに変換)
3. 祝日の扱い(前営業日繰越 vs NaN)
4. 月次→日次補完時の前向きバイアス防止

**datachecker基準:**
- 欠損率 < 5%(日次データ)
- 欠損率 < 2%(月次データ)
- Autocorrelation < 0.99
- VIF < 10(国間相関チェック)

---

## 9. 注意事項・制約事項

### 9.1 データ利用上の制約

1. **FRED OECDデータは月次のみ**
   - 105系列すべてFREQ: M
   - 日次化には補完必須だが前向きバイアスリスク

2. **yfinance非米国ティッカーは未検証**
   - ドキュメントに記載なし
   - 実運用前に必ず欠損率・遅延チェック

3. **インフレ連動債は米英のみ**
   - ドイツ2024年終了、カナダ2022年終了
   - 実質金利の多国間拡張は困難

4. **各国中央銀行APIは非統一**
   - BOJ: 手動ダウンロード形式
   - BOE: Database UI経由
   - ECB: REST API(最も使いやすい)

### 9.2 多国間データの現実性

**architect要確認事項:**
- プロジェクト当初の想定「G10 × ~50,000サンプル」は**過大見積もり**
- FREDでのTIPS相当データは米国のみ
- 実現可能な規模: **4経済圏 × ~20,000サンプル**(日次tier1のみ)

**前回attempt 3の教訓:**
- 月次→日次変換はstep functionでMAE悪化
- 多国間データは情報増(MI +23.8%)だがGate 3失敗
- **日次データのみ使用**が安全

---

## 10. 結論・推奨事項

### 最終推奨戦略

#### 推奨1: 米国FRED単一(最高品質・最低リスク)

**利用データ:**
```
DGS10  (10年名目)     - 日次, 1962-現在
DFII10 (10年実質)     - 日次, 2003-現在
DGS2   (2年名目)      - 日次, 1976-現在
T10YIE (期待インフレ)  - 日次, 2003-現在
```

**サンプル数:** ~6,000(2003-2026)

**メリット:**
- 最高品質(FREDは最も信頼性高い)
- 欠損ほぼなし
- 実質金利直接利用可能
- 補完不要

**デメリット:**
- 単一国のみ(多様性低い)
- サンプル数少ない

#### 推奨2: Tier1 4経済圏(バランス型)

**利用データ:**
```
米国: FRED API (DGS10, DFII10)
英国: BOE Database (10Y nominal/real)
日本: BOJ Portal (10Y JGB)
欧州: ECB API (10Y Euro area)
```

**サンプル数:** ~20,000

**メリット:**
- サンプル数4倍
- 多様性高い(4経済圏)
- すべて日次データ(補完不要)

**デメリット:**
- データ取得の複雑性(3 API + 1手動)
- タイムゾーン・祝日の統一処理必要

#### 推奨3(非推奨): 月次補完

**理由:**
- attempt 3でstep function失敗
- スプライン補完も前向きバイアスリスク
- 複雑性に見合う性能向上なし

### architectへの質問事項

1. **多国間データの必要性**: 米国単一 vs 4経済圏どちらを選ぶ?
2. **実質金利の扱い**: 米英のみTIPS/Giltあり。他国は名目-インフレで計算?
3. **データ取得工数**: BOJ/BOE手動ダウンロード許容可能?

---

## Sources

- [FRED - Germany 10-Year Bond Yield](https://fred.stlouisfed.org/series/IRLTLT01DEM156N)
- [FRED - Japan 10-Year Bond Yield](https://fred.stlouisfed.org/series/IRLTLT01JPM156N)
- [FRED - UK 10-Year Bond Yield](https://fred.stlouisfed.org/series/IRLTLT01GBM156N)
- [Federal Reserve H.15 Daily Interest Rates](https://www.federalreserve.gov/releases/h15/)
- [FRED - US 10-Year Treasury (DGS10)](https://fred.stlouisfed.org/series/DGS10)
- [FRED - US 10-Year TIPS (DFII10)](https://fred.stlouisfed.org/series/DFII10)
- [Yahoo Finance - US 10-Year (^TNX)](https://finance.yahoo.com/quote/%5ETNX/)
- [Trading Economics - Japan Government Bond Yield](https://tradingeconomics.com/japan/government-bond-yield)
- [FRED - Treasury Inflation-Indexed Securities Category](https://fred.stlouisfed.org/categories/82)
- [Bank of Japan - Time-Series Data Search](https://www.stat-search.boj.or.jp/index_en.html)
- [Bank of England - Yield Curves](https://www.bankofengland.co.uk/statistics/yield-curves)
- [Bank of England - Database](https://www.bankofengland.co.uk/boeapps/database/)
- [ECB Data Portal](https://data.ecb.europa.eu/)
- [ECB - Euro Area Yield Curves](https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html)
- [Nasdaq Data Link - Interest Rate Data API](https://blog.data.nasdaq.com/api-for-interest-rate-data)
- [Trading Economics - Bond Yields by Country](https://tradingeconomics.com/bonds)

---

**作成日**: 2026-02-17
**作成者**: researcher (Sonnet 4.5)
**レポートバージョン**: 1.0
**総語数**: 約1,950語
