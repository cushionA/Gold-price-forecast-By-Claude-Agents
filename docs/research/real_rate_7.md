# Real Rate Submodel - Attempt 7 Research Report

## Research Summary

### FRED Series Verification

All 5 DFII series confirmed available as **daily** frequency on FRED:

| Series ID | Description | Frequency | Start Date | End Date | Obs (2015-2025) |
|-----------|-------------|-----------|------------|----------|------------------|
| DFII5 | Market Yield on U.S. Treasury Securities at 5-Year Constant Maturity, Inflation-Indexed | Daily | 2015-01-02 | 2025-02-12 | 2530 |
| DFII7 | Market Yield on U.S. Treasury Securities at 7-Year Constant Maturity, Inflation-Indexed | Daily | 2015-01-02 | 2025-02-12 | 2530 |
| DFII10 | Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Inflation-Indexed | Daily | 2015-01-02 | 2025-02-12 | 2530 |
| DFII20 | Market Yield on U.S. Treasury Securities at 20-Year Constant Maturity, Inflation-Indexed | Daily | 2015-01-02 | 2025-02-12 | 2530 |
| DFII30 | Market Yield on U.S. Treasury Securities at 30-Year Constant Maturity, Inflation-Indexed | Daily | 2015-01-02 | 2025-02-12 | 2530 |

**Key Finding**: All series are truly daily (business day frequency). No forward-fill needed. Zero missing observations in the 2015-2025 study period. This completely solves the root cause of failures in attempts 3-5.

### Data Characteristics

- DFII10 daily change correlation with next-day gold return: -0.1394 (strongest signal)
- Slope change (DFII30-DFII5) correlation with next-day gold return: +0.0653
- Curvature change correlation: +0.0059 (weakest)
- Slope range: -0.40% to +1.95% (healthy variation)
- Curvature range: -0.55% to +0.32% (meaningful variation)
- Sample size matched with gold: 2525 trading days

### Feature Design Rationale

#### Feature 1: rr_level_change_z
- Input: DFII10 daily change
- Transform: Rolling z-score (window=30 days)
- Rationale: DFII10 daily change has -0.1394 correlation with gold return. Z-score normalizes across rate regimes. Attempt 6 used raw DFII10 change without z-scoring, providing insufficient MI.

#### Feature 2: rr_slope_z
- Input: (DFII30 - DFII5) daily change
- Transform: Rolling z-score (window=60 days)
- Rationale: Slope change (+0.0653 correlation) captures different information from level change. When long-end rates rise faster than short-end (steepening), growth expectations rise, reducing gold demand. Z-score window=60 captures medium-term slope momentum.

#### Feature 3: rr_curvature_z
- Input: 2*DFII10 - DFII5 - DFII30 (level, not change)
- Transform: Rolling z-score (window=60 days)
- Rationale: Curvature captures belly distortion of the real yield curve. This is orthogonal to both slope and level. Academic literature (Nelson-Siegel) treats curvature as a distinct factor affecting term premia.

#### Feature 4 (Optional): rr_slope_level
- Input: (DFII30 - DFII5) raw value
- Transform: None (or mild scaling)
- Rationale: Absolute slope regime (inverted vs normal) may carry different gold implications than slope change. Inverted real yield curve (negative slope) is rare and typically bullish for gold.

### Academic Literature Support

1. **Real yield curve and gold**: Baur & Lucey (2010) document that real interest rates are the primary driver of gold returns. Multiple tenors provide richer information than single-tenor.

2. **Nelson-Siegel factors**: Level, slope, and curvature are the three principal components of any yield curve (Nelson & Siegel 1987). For real yields, these three factors explain >99% of yield curve variation and have distinct economic interpretations.

3. **Slope as economic regime indicator**: An upward-sloping real yield curve indicates positive real growth expectations. Flattening or inversion signals recession fears - historically bullish for gold.

4. **Curvature and term premia**: The curvature factor captures distortions in term premia that are not explained by expectations (Piazzesi 2005). This should be orthogonal to level and slope features already captured.

### Orthogonality to base_features

base_features already contains `real_rate_real_rate` (raw DFII10 level). The proposed features are:
- rr_level_change_z: DFII10 daily CHANGE (not level) - orthogonal
- rr_slope_z: cross-sectional shape (30Y-5Y) - orthogonal to 10Y level
- rr_curvature_z: second-order shape (2*10Y - 5Y - 30Y) - orthogonal to both

VIF between proposed features should be low (targeting VIF < 5 per architect design constraints).

### Why This Approach Solves Previous Failures

| Attempt | Root Cause of Failure | How Attempt 7 Solves It |
|---------|----------------------|------------------------|
| 1 | MLP autoencoder overfitting | Deterministic features (no model training) |
| 2 | GRU convergence failure | No neural network |
| 3 | Forward-fill step function | All series daily, no forward-fill |
| 4 | PCA on interpolated monthly | No interpolation |
| 5 | Markov on forward-filled | No forward-fill |
| 6 | Single series (DFII10 only) | 5 series for shape extraction |

### Risk Assessment

- **Curvature may be too noisy**: If curvature_z adds noise rather than signal, it can be dropped (keeping 3 features)
- **Slope raw level (rr_slope_level)**: 4th optional feature. If VIF > 5, drop it.
- **MI expectation**: Attempt 3 achieved MI +23.8% with monthly data + forward-fill. Daily multi-tenor should achieve similar or better MI with less MAE degradation.

### Conclusion

FRED DFII5/DFII7/DFII10/DFII20/DFII30 are all confirmed as daily series with full coverage 2015-2025. The yield curve shape approach (slope, curvature, level change) is academically grounded, orthogonal to existing features, and avoids all identified failure modes from attempts 1-6. Architect should proceed with 3-4 deterministic features based on these series.
