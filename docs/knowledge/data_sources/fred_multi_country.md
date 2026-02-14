# FRED Multi-Country Data Availability

## Q1: Does FRED have TIPS-equivalent data for G7/G10 countries?

**Answer**: **No**. FRED's inflation-indexed securities data is overwhelmingly U.S.-focused.

**Evidence**:
- FRED maintains 177 economic data series for Treasury Inflation-Indexed Securities
- 173 series tagged with TIPS
- 93 series specifically for 10-year TIPS
- **No direct TIPS-equivalent series found for other G7/G10 countries**

**Source**: FRED API search (2026-02-14, real_rate research)

**Implication**: Multi-country TIPS data via FRED is **NOT readily available** for direct use.

**Last Verified**: 2026-02-14

---

## Q2: What U.S. TIPS series are available on FRED?

**Answer**: Multiple maturities available, all daily frequency from 2003-present.

**Primary Series**:

| Series | Description | Start Date | Frequency |
|--------|-------------|------------|-----------|
| **DFII10** | 10-Year TIPS Yield | 2003-01-02 | Daily |
| DFII5 | 5-Year TIPS Yield | 2003-01-02 | Daily |
| DFII7 | 7-Year TIPS Yield | 2003-01-03 | Daily |
| DFII20 | 20-Year TIPS Yield | 2003-01-02 | Daily |
| DFII30 | 30-Year TIPS Yield | 2010-02-22 | Daily |

**Recommendation**: Use DFII10 (10-year) as primary series. Most liquid and longest history.

**Data Volume**: ~6,000 observations (2003-2026), ~5,800 non-NaN

**Last Verified**: 2026-02-14 (architect fact-check confirmed via API)

---

## Q3: How can I get multi-country real rate data if FRED doesn't have it?

**Answer**: Construct synthetic real rates from nominal yields minus inflation expectations.

**Formula**:
```
Real Rate ≈ Nominal Yield - Expected Inflation
```

**FRED Data Available**:

### Nominal Government Bond Yields (Monthly):
- Germany: `IRLTLT01DEM156N` (Long-term, monthly)
- Japan: `IRLTLT01JPM156N`
- UK: `IRLTLT01GBM156N`
- France: `IRLTLT01FRM156N`
- Canada: `IRLTLT01CAM156N`

### Inflation Expectations:
- Search FRED for CPI or inflation expectation series for each country
- G7 countries: 64 series tagged with both "G7" and "CPI"

**Critical Limitation**: ⚠️ **Monthly frequency only** (not daily)
- Cannot be used for daily submodel training
- Would need interpolation (introduces artifacts)
- Not suitable for high-frequency gold prediction

**Recommendation for Smoke Test**:
- ✓ **Use U.S. DFII10 only** (~2,500 daily observations)
- Skip multi-country data for smoke test
- Consider for Attempt 2+ if architect approves synthetic construction

**Last Verified**: 2026-02-14

---

## Q4: Can I use nominal yields as a proxy for real rates?

**Answer**: Possible but problematic for multi-country comparison.

**Why It's Problematic**:
- Nominal yields mix real rates + inflation expectations + term premium
- Different countries have different inflation regimes
- Cross-country comparison is not apples-to-apples
- Gold responds to **real** rates (inflation-adjusted), not nominal

**Example**:
- U.S. 10Y nominal: 4.5%, inflation: 3.0% → Real rate: ~1.5%
- Japan 10Y nominal: 0.5%, inflation: 1.5% → Real rate: ~-1.0%
- Using nominal yields directly would miss this key difference

**Recommendation**:
- For smoke test: Use U.S. real rates (DFII10) only
- For multi-country: Must construct synthetic real rates (nominal - expected inflation)
- Requires architect approval and data quality validation

**Last Verified**: 2026-02-14

---

## Q5: How many observations can I get with U.S. DFII10 only?

**Answer**: Approximately 2,500 daily observations aligned with schema

**Data Details**:
- Raw DFII10: 6,031 observations (2003-01-02 to 2026-02-14)
- Non-NaN: 5,782 observations
- After alignment with gold trading days (schema): **2,523 observations**
  - Start: 2015-01-30 (schema start)
  - End: 2025-02-12 (schema end)

**Training Split** (70/15/15):
- Train: ~1,766 samples
- Val: ~378 samples
- Test: ~379 samples

**Is This Enough?**
- ✓ Sufficient for smoke test (thousands of samples)
- ✓ Sufficient for MLP autoencoder (simple architecture)
- ? May need more for complex architectures (GRU, Transformer)
- Multi-country expansion would give ~10-50x more data (if monthly interpolation accepted)

**Last Verified**: 2026-02-14 (builder_data results)

---

## Q6: What about other countries' inflation-indexed bonds?

**Answer**: They exist but are NOT in FRED. Would need alternative data sources.

**Countries with Inflation-Indexed Bonds**:
- **UK**: Index-Linked Gilts (since 1981)
- **France**: OATi and OAT€i
- **Germany**: Inflation-linked Bunds
- **Japan**: JGBi (Japan Government Inflation-Indexed Bonds)
- **Canada**: Real Return Bonds

**Data Sources** (not FRED):
- Central bank websites (e.g., Bank of England, Banque de France)
- Bloomberg Terminal (paid)
- Refinitiv/Datastream (paid)
- ECB Statistical Data Warehouse (European bonds)

**Challenges**:
- Not free/easily accessible
- Different issuance dates (varying history lengths)
- Different index methodologies
- Requires API setup and authentication

**Recommendation**:
- Smoke test: Skip (use U.S. only)
- Attempt 2+: Discuss with user if paid data sources are acceptable
- Alternative: Use synthetic real rates from FRED nominal yields

**Last Verified**: 2026-02-14
