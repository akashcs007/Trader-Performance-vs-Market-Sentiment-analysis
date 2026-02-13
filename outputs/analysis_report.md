# Trader Performance vs Market Sentiment

## Methodology
- Loaded `historical_data.csv` and `fear_greed_index.csv`.
- Deduplicated rows, parsed timestamps, and aligned both datasets on daily dates.
- Computed day-level and account-day-level metrics (PnL, win rate, activity, size, buy/sell bias).
- Built 3 segment families: frequency, trade size, and consistency.

## Data Preparation Checks
- Trades rows/cols (before): 211224 / 16
- Sentiment rows/cols (before): 2644 / 4
- Trades duplicates removed: 0
- Sentiment duplicates removed: 0
- Trades missing values (before -> after): 0 -> 0
- Sentiment missing values (before -> after): 0 -> 0
- Overlap window used for merge: 2023-05-01 to 2025-05-01

## Fear vs Greed Findings
- Mean win-day rate on fear-like days: 0.602
- Mean win-day rate on greed-like days: 0.647
- Avg daily trades on fear-like days: 1,104.0
- Avg daily trades on greed-like days: 305.7
- Drawdown proxy (avg negative account-day PnL) on fear-like days: -8,678.78
- Drawdown proxy on greed-like days: -7,860.08

## Segment Highlights
- Top consistency segment by avg total PnL:
  - {'freq_segment': nan, 'traders': 13, 'avg_total_pnl': 410335.9900119231, 'avg_win_day_rate': 0.7535975110247399, 'avg_size_usd': nan, 'segment_type': 'Consistency', 'size_segment': nan, 'avg_trades_per_day': 161.97388564074234, 'consistency_segment': 'Consistent_Winner'}
- Top frequency segment by avg total PnL:
  - {'freq_segment': 'Frequent', 'traders': 16, 'avg_total_pnl': 383853.3369425625, 'avg_win_day_rate': 0.6576067585323723, 'avg_size_usd': 11743.66362565366, 'segment_type': 'Frequency', 'size_segment': nan, 'avg_trades_per_day': nan, 'consistency_segment': nan}

## 3 Actionable Insights
1. Sentiment regimes are associated with clear changes in trader outcomes (win rate and account-day PnL differ by class).
2. Activity intensity shifts with sentiment (trade count and volume vary materially across Fear/Greed classes).
3. Segment behavior matters: performance differences across frequent/infrequent and consistency segments are substantial enough for strategy personalization.

## Strategy Rules of Thumb
1. **Risk scaling by sentiment:** On fear-like days, reduce position size for weaker segments and prioritize tighter stop/risk limits to control downside tails.
2. **Segment-aware activity:** Allow higher trade frequency only for stronger consistency segments; cap frequency for mixed/inconsistent segments when sentiment is adverse.

## Files Generated
- `outputs/daily_metrics.csv`
- `outputs/account_day_metrics.csv`
- `outputs/sentiment_comparison.csv`
- `outputs/trader_segments.csv`
- `outputs/segment_summary.csv`
- `outputs/pnl_distribution_by_sentiment.png`
- `outputs/win_rate_by_sentiment.png`
- `outputs/activity_by_sentiment.png`
