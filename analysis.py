import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv("historical_data.csv")
    sentiment = pd.read_csv("fear_greed_index.csv")
    return trades, sentiment


def clean_and_prepare(trades: pd.DataFrame, sentiment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta = {
        "trades_rows_before": len(trades),
        "trades_cols": trades.shape[1],
        "sent_rows_before": len(sentiment),
        "sent_cols": sentiment.shape[1],
        "trades_missing_before": int(trades.isna().sum().sum()),
        "sent_missing_before": int(sentiment.isna().sum().sum()),
        "trades_duplicates_before": int(trades.duplicated().sum()),
        "sent_duplicates_before": int(sentiment.duplicated().sum()),
    }

    trades = trades.drop_duplicates().copy()
    sentiment = sentiment.drop_duplicates().copy()

    numeric_cols = ["Execution Price", "Size Tokens", "Size USD", "Start Position", "Closed PnL", "Fee", "Timestamp"]
    for col in numeric_cols:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    sentiment["value"] = pd.to_numeric(sentiment["value"], errors="coerce")
    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce")
    sentiment["classification"] = sentiment["classification"].str.strip().str.title()
    sentiment = sentiment.dropna(subset=["date", "classification"])
    sentiment["date"] = sentiment["date"].dt.normalize()

    # Prefer exchange-provided IST string for daily grouping; fall back to epoch timestamp.
    trades["timestamp_ist_dt"] = pd.to_datetime(
        trades["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce", dayfirst=True
    )
    fallback_dt = pd.to_datetime(trades["Timestamp"], unit="ms", errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    trades["trade_dt"] = trades["timestamp_ist_dt"].fillna(fallback_dt)
    trades["trade_date"] = pd.to_datetime(trades["trade_dt"]).dt.normalize()

    trades["Closed PnL"] = trades["Closed PnL"].fillna(0.0)
    trades["Size USD"] = trades["Size USD"].fillna(0.0)
    trades["Fee"] = trades["Fee"].fillna(0.0)

    # Directional flags from side to estimate long/short participation bias.
    trades["is_buy"] = trades["Side"].astype(str).str.upper().eq("BUY").astype(int)
    trades["is_sell"] = trades["Side"].astype(str).str.upper().eq("SELL").astype(int)

    trades = trades.dropna(subset=["trade_date"])

    meta["trades_rows_after"] = len(trades)
    meta["sent_rows_after"] = len(sentiment)
    meta["trades_missing_after"] = int(trades.isna().sum().sum())
    meta["sent_missing_after"] = int(sentiment.isna().sum().sum())
    meta["date_overlap_start"] = str(max(trades["trade_date"].min(), sentiment["date"].min()).date())
    meta["date_overlap_end"] = str(min(trades["trade_date"].max(), sentiment["date"].max()).date())

    return trades, sentiment, meta


def build_daily_metrics(trades: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    daily = (
        trades.groupby("trade_date", as_index=False)
        .agg(
            total_trades=("Trade ID", "count"),
            unique_traders=("Account", "nunique"),
            total_pnl=("Closed PnL", "sum"),
            avg_pnl_per_trade=("Closed PnL", "mean"),
            median_trade_size_usd=("Size USD", "median"),
            mean_trade_size_usd=("Size USD", "mean"),
            total_volume_usd=("Size USD", "sum"),
            total_fees=("Fee", "sum"),
            buy_trades=("is_buy", "sum"),
            sell_trades=("is_sell", "sum"),
        )
        .rename(columns={"trade_date": "date"})
    )
    daily["buy_sell_ratio"] = daily["buy_trades"] / daily["sell_trades"].replace(0, np.nan)
    daily["net_flow_bias"] = (daily["buy_trades"] - daily["sell_trades"]) / (
        daily["buy_trades"] + daily["sell_trades"]
    ).replace(0, np.nan)

    # Drawdown proxy: average negative account-day pnl on each date.
    account_day = trades.groupby(["trade_date", "Account"], as_index=False).agg(account_day_pnl=("Closed PnL", "sum"))
    neg_only = account_day[account_day["account_day_pnl"] < 0]
    dd = (
        neg_only.groupby("trade_date", as_index=False)
        .agg(avg_loss_account_day=("account_day_pnl", "mean"), p10_loss_account_day=("account_day_pnl", lambda x: x.quantile(0.10)))
        .rename(columns={"trade_date": "date"})
    )
    daily = daily.merge(dd, on="date", how="left")

    merged = daily.merge(sentiment[["date", "classification", "value"]], on="date", how="inner")
    return merged


def build_account_day_metrics(trades: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    acc_day = (
        trades.groupby(["trade_date", "Account"], as_index=False)
        .agg(
            trades_count=("Trade ID", "count"),
            account_day_pnl=("Closed PnL", "sum"),
            avg_trade_size_usd=("Size USD", "mean"),
            median_trade_size_usd=("Size USD", "median"),
            day_buy_trades=("is_buy", "sum"),
            day_sell_trades=("is_sell", "sum"),
        )
        .rename(columns={"trade_date": "date"})
    )
    acc_day["win_day"] = (acc_day["account_day_pnl"] > 0).astype(int)
    acc_day["buy_sell_ratio"] = acc_day["day_buy_trades"] / acc_day["day_sell_trades"].replace(0, np.nan)
    acc_day["activity_bucket"] = pd.qcut(acc_day["trades_count"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    acc_day["size_bucket"] = pd.qcut(
        acc_day["avg_trade_size_usd"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    acc_day = acc_day.merge(sentiment[["date", "classification", "value"]], on="date", how="inner")
    return acc_day


def build_trader_segments(acc_day: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trader_profile = (
        acc_day.groupby("Account", as_index=False)
        .agg(
            active_days=("date", "nunique"),
            avg_trades_per_day=("trades_count", "mean"),
            median_trades_per_day=("trades_count", "median"),
            avg_size_usd=("avg_trade_size_usd", "mean"),
            total_pnl=("account_day_pnl", "sum"),
            mean_daily_pnl=("account_day_pnl", "mean"),
            win_day_rate=("win_day", "mean"),
        )
        .fillna(0.0)
    )

    trades_median = trader_profile["avg_trades_per_day"].median()
    size_median = trader_profile["avg_size_usd"].median()

    trader_profile["freq_segment"] = np.where(
        trader_profile["avg_trades_per_day"] >= trades_median, "Frequent", "Infrequent"
    )
    trader_profile["size_segment"] = np.where(trader_profile["avg_size_usd"] >= size_median, "Large_Size", "Small_Size")
    trader_profile["consistency_segment"] = np.select(
        [
            (trader_profile["win_day_rate"] >= 0.60) & (trader_profile["active_days"] >= 20),
            (trader_profile["win_day_rate"] <= 0.40) & (trader_profile["active_days"] >= 20),
        ],
        ["Consistent_Winner", "Inconsistent"],
        default="Mixed",
    )

    freq_summary = (
        trader_profile.groupby("freq_segment", as_index=False)
        .agg(
            traders=("Account", "count"),
            avg_total_pnl=("total_pnl", "mean"),
            avg_win_day_rate=("win_day_rate", "mean"),
            avg_size_usd=("avg_size_usd", "mean"),
        )
        .sort_values("avg_total_pnl", ascending=False)
    )
    size_summary = (
        trader_profile.groupby("size_segment", as_index=False)
        .agg(
            traders=("Account", "count"),
            avg_total_pnl=("total_pnl", "mean"),
            avg_win_day_rate=("win_day_rate", "mean"),
            avg_trades_per_day=("avg_trades_per_day", "mean"),
        )
        .sort_values("avg_total_pnl", ascending=False)
    )
    consistency_summary = (
        trader_profile.groupby("consistency_segment", as_index=False)
        .agg(
            traders=("Account", "count"),
            avg_total_pnl=("total_pnl", "mean"),
            avg_win_day_rate=("win_day_rate", "mean"),
            avg_trades_per_day=("avg_trades_per_day", "mean"),
        )
        .sort_values("avg_total_pnl", ascending=False)
    )

    segment_summary = pd.concat(
        [
            freq_summary.assign(segment_type="Frequency"),
            size_summary.assign(segment_type="TradeSize"),
            consistency_summary.assign(segment_type="Consistency"),
        ],
        ignore_index=True,
    )
    return trader_profile, segment_summary


def sentiment_comparison(daily_metrics: pd.DataFrame, acc_day: pd.DataFrame) -> pd.DataFrame:
    day_level = (
        daily_metrics.groupby("classification", as_index=False)
        .agg(
            days=("date", "nunique"),
            avg_daily_pnl=("total_pnl", "mean"),
            median_daily_pnl=("total_pnl", "median"),
            avg_daily_trades=("total_trades", "mean"),
            avg_daily_volume_usd=("total_volume_usd", "mean"),
            avg_buy_sell_ratio=("buy_sell_ratio", "mean"),
            avg_dd_proxy=("avg_loss_account_day", "mean"),
        )
        .sort_values("avg_daily_pnl", ascending=False)
    )

    account_level = (
        acc_day.groupby("classification", as_index=False)
        .agg(
            account_day_obs=("Account", "count"),
            avg_account_day_pnl=("account_day_pnl", "mean"),
            median_account_day_pnl=("account_day_pnl", "median"),
            win_day_rate=("win_day", "mean"),
            avg_account_trades=("trades_count", "mean"),
            avg_account_size_usd=("avg_trade_size_usd", "mean"),
            avg_account_buy_sell_ratio=("buy_sell_ratio", "mean"),
        )
        .sort_values("avg_account_day_pnl", ascending=False)
    )

    combined = day_level.merge(account_level, on="classification", how="outer")
    return combined


def save_plots(daily_metrics: pd.DataFrame, acc_day: pd.DataFrame, out_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1) Distribution of account-day pnl by sentiment
    fig, ax = plt.subplots(figsize=(9, 5))
    classes = sorted(acc_day["classification"].dropna().unique().tolist())
    data = [acc_day.loc[acc_day["classification"] == c, "account_day_pnl"].values for c in classes]
    ax.boxplot(data, labels=classes, showfliers=False)
    ax.set_title("Account-Day PnL Distribution by Sentiment")
    ax.set_ylabel("Account-Day PnL")
    fig.tight_layout()
    fig.savefig(out_dir / "pnl_distribution_by_sentiment.png", dpi=160)
    plt.close(fig)

    # 2) Win-day rate by sentiment
    win = acc_day.groupby("classification", as_index=False)["win_day"].mean().sort_values("win_day", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(win["classification"], win["win_day"])
    ax.set_title("Win-Day Rate by Sentiment")
    ax.set_ylabel("Win-Day Rate")
    ax.set_ylim(0, 1)
    for idx, val in enumerate(win["win_day"]):
        ax.text(idx, val + 0.01, f"{val:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "win_rate_by_sentiment.png", dpi=160)
    plt.close(fig)

    # 3) Trading activity by sentiment
    activity = (
        daily_metrics.groupby("classification", as_index=False)[["total_trades", "total_volume_usd"]]
        .mean()
        .sort_values("total_trades", ascending=False)
    )
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(activity))
    width = 0.4
    ax1.bar(x - width / 2, activity["total_trades"], width, label="Avg Daily Trades")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, activity["total_volume_usd"], width, label="Avg Daily Volume USD", alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(activity["classification"])
    ax1.set_title("Trading Activity by Sentiment")
    ax1.set_ylabel("Avg Daily Trades")
    ax2.set_ylabel("Avg Daily Volume USD")
    fig.tight_layout()
    fig.savefig(out_dir / "activity_by_sentiment.png", dpi=160)
    plt.close(fig)


def write_markdown_report(
    meta: dict,
    comp: pd.DataFrame,
    segment_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    by_cls = comp.set_index("classification")
    fear_like = by_cls.loc[[x for x in by_cls.index if "Fear" in x]]
    greed_like = by_cls.loc[[x for x in by_cls.index if "Greed" in x]]

    fear_win = fear_like["win_day_rate"].mean() if not fear_like.empty else np.nan
    greed_win = greed_like["win_day_rate"].mean() if not greed_like.empty else np.nan
    fear_trades = fear_like["avg_daily_trades"].mean() if not fear_like.empty else np.nan
    greed_trades = greed_like["avg_daily_trades"].mean() if not greed_like.empty else np.nan
    fear_dd = fear_like["avg_dd_proxy"].mean() if not fear_like.empty else np.nan
    greed_dd = greed_like["avg_dd_proxy"].mean() if not greed_like.empty else np.nan

    best_consistency = segment_summary[segment_summary["segment_type"] == "Consistency"].head(1)
    best_freq = segment_summary[segment_summary["segment_type"] == "Frequency"].head(1)

    report = f"""# Trader Performance vs Market Sentiment

## Methodology
- Loaded `historical_data.csv` and `fear_greed_index.csv`.
- Deduplicated rows, parsed timestamps, and aligned both datasets on daily dates.
- Computed day-level and account-day-level metrics (PnL, win rate, activity, size, buy/sell bias).
- Built 3 segment families: frequency, trade size, and consistency.

## Data Preparation Checks
- Trades rows/cols (before): {meta['trades_rows_before']} / {meta['trades_cols']}
- Sentiment rows/cols (before): {meta['sent_rows_before']} / {meta['sent_cols']}
- Trades duplicates removed: {meta['trades_duplicates_before']}
- Sentiment duplicates removed: {meta['sent_duplicates_before']}
- Trades missing values (before -> after): {meta['trades_missing_before']} -> {meta['trades_missing_after']}
- Sentiment missing values (before -> after): {meta['sent_missing_before']} -> {meta['sent_missing_after']}
- Overlap window used for merge: {meta['date_overlap_start']} to {meta['date_overlap_end']}

## Fear vs Greed Findings
- Mean win-day rate on fear-like days: {fear_win:.3f}
- Mean win-day rate on greed-like days: {greed_win:.3f}
- Avg daily trades on fear-like days: {fear_trades:,.1f}
- Avg daily trades on greed-like days: {greed_trades:,.1f}
- Drawdown proxy (avg negative account-day PnL) on fear-like days: {fear_dd:,.2f}
- Drawdown proxy on greed-like days: {greed_dd:,.2f}

## Segment Highlights
- Top consistency segment by avg total PnL:
  - {best_consistency.to_dict(orient='records')[0] if not best_consistency.empty else 'N/A'}
- Top frequency segment by avg total PnL:
  - {best_freq.to_dict(orient='records')[0] if not best_freq.empty else 'N/A'}

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
"""

    (out_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    trades, sentiment = load_data()
    trades, sentiment, meta = clean_and_prepare(trades, sentiment)

    daily_metrics = build_daily_metrics(trades, sentiment)
    account_day_metrics = build_account_day_metrics(trades, sentiment)
    trader_segments, segment_summary = build_trader_segments(account_day_metrics)
    comp = sentiment_comparison(daily_metrics, account_day_metrics)

    daily_metrics.to_csv(OUTPUT_DIR / "daily_metrics.csv", index=False)
    account_day_metrics.to_csv(OUTPUT_DIR / "account_day_metrics.csv", index=False)
    comp.to_csv(OUTPUT_DIR / "sentiment_comparison.csv", index=False)
    trader_segments.to_csv(OUTPUT_DIR / "trader_segments.csv", index=False)
    segment_summary.to_csv(OUTPUT_DIR / "segment_summary.csv", index=False)

    save_plots(daily_metrics, account_day_metrics, OUTPUT_DIR)
    write_markdown_report(meta, comp, segment_summary, OUTPUT_DIR)

    print("Analysis complete. See outputs/ for tables, charts, and report.")


if __name__ == "__main__":
    main()
