# Trader Performance vs Market Sentiment

This repository contains a complete solution for the Data Science Intern assignment:
analyzing how Fear/Greed sentiment relates to trader behavior and performance.

## Files
- `analysis.py` - end-to-end analysis pipeline
- `historical_data.csv` - Hyperliquid trader data
- `fear_greed_index.csv` - market sentiment data
- `outputs/analysis_report.md` - concise findings + strategy suggestions
- `outputs/*.csv` - processed metrics and segment tables
- `outputs/*.png` - charts used as evidence

## How to Run
```bash
python analysis.py
```

## What the script does
1. Loads and validates both datasets (shape, missing values, duplicates).
2. Parses timestamps and aligns records at daily granularity.
3. Builds key metrics:
   - daily/account-day PnL
   - win-day rate
   - trade activity and size
   - buy/sell directional bias
   - drawdown proxy (negative account-day PnL)
4. Compares metrics across sentiment classes.
5. Creates trader segments:
   - frequent vs infrequent
   - large-size vs small-size
   - consistent winners vs inconsistent
6. Exports evidence tables/charts and a short write-up.

## Notes
- The provided historical dataset does not include an explicit leverage column, so the analysis uses trade size and activity as risk-behavior proxies.
- The overlap period used between both datasets is handled automatically by the script.
