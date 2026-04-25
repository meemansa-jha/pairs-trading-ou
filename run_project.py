"""
run_project.py
--------------
End-to-end pipeline:

  1. Load prices from prices.csv (created by data_loader.py).
  2. Split into training (first 75%) and test (last 25%) windows.
  3. For each candidate pair: test cointegration, fit OU, backtest.
  4. Print summary tables, save plots, save a results CSV.

Run:
    python run_project.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pairs_analysis import (engle_granger_test, fit_ou, backtest_pair,
                            EG_CRITICAL_VALUES)

# Candidate pairs must match the ones you downloaded in data_loader.py.
CANDIDATE_PAIRS = [
    ("KO",   "PEP"),
    ("XOM",  "CVX"),
    ("MA",   "V"),
    ("GLD",  "SLV"),
    ("HD",   "LOW"),
    ("HDFCBANK.NS", "ICICIBANK.NS"),
    ("TCS.NS",      "INFY.NS"),
    ("RELIANCE.NS", "ONGC.NS"),
]

PRICES_FILE = "prices.csv"
OUTPUT_DIR = "output"
TRAIN_FRAC = 0.75      # first 75% of data for fitting, last 25% for backtest


def load_prices():
    df = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df


def split_train_test(df, train_frac=TRAIN_FRAC):
    n = len(df)
    cut = int(n * train_frac)
    return df.iloc[:cut], df.iloc[cut:]


def analyse_pair(ticker_a, ticker_b, train, test):
    """Run cointegration test, OU fit, and backtest for one pair."""
    if ticker_a not in train.columns or ticker_b not in train.columns:
        return None
    log_a_tr = np.log(train[ticker_a].values)
    log_b_tr = np.log(train[ticker_b].values)
    log_a_te = np.log(test[ticker_a].values)
    log_b_te = np.log(test[ticker_b].values)

    # --- Step 1: cointegration on training set ---
    eg = engle_granger_test(log_a_tr, log_b_tr)

    # --- Step 2: fit OU on training-set residuals ---
    try:
        ou = fit_ou(eg["residuals"], dt=1/252)
    except ValueError as err:
        return {"pair": (ticker_a, ticker_b), "skipped": str(err), "eg": eg}

    # --- Step 3: backtest on test set using training-set parameters ---
    result = backtest_pair(
        dates=test.index,
        log_a=log_a_te, log_b=log_b_te,
        alpha=eg["alpha"], beta=eg["beta"], ou=ou,
        entry_z=2.0, exit_z=0.5, stop_z=3.5,
        cost_per_leg=0.0010,
        pair_label=(ticker_a, ticker_b),
    )
    return {
        "pair": (ticker_a, ticker_b),
        "eg": eg,
        "ou": ou,
        "backtest": result,
        "train": train, "test": test,
        "log_a_te": log_a_te, "log_b_te": log_b_te,
    }


def plot_best_pair(res, outpath):
    """Plot the spread of the best pair with z-bands and trade markers."""
    eg = res["eg"]; ou = res["ou"]; bt = res["backtest"]
    test = res["test"]
    spread_te = res["log_a_te"] - eg["alpha"] - eg["beta"] * res["log_b_te"]
    z_te = (spread_te - ou.mu) / ou.stationary_std

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top panel: raw spread with ±n sigma bands
    axes[0].plot(test.index, spread_te, color="steelblue", linewidth=1.2, label="Spread (test)")
    axes[0].axhline(ou.mu, color="black", linestyle="--", linewidth=0.8, label="μ (long-run mean)")
    for k, style in [(2, "-"), (3.5, ":")]:
        axes[0].axhline(ou.mu + k * ou.stationary_std, color="red", linestyle=style, linewidth=0.7,
                        label=f"±{k}σ" if k == 2 else f"±{k}σ (stop)")
        axes[0].axhline(ou.mu - k * ou.stationary_std, color="red", linestyle=style, linewidth=0.7)
    axes[0].set_ylabel("Spread  (log $P_A$ − α − β log $P_B$)")
    axes[0].set_title(f"Pair: {res['pair'][0]} / {res['pair'][1]}   "
                      f"θ={ou.theta:.2f}/yr, half-life={ou.half_life*252:.1f} days")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    # Bottom panel: z-score with entry/exit markers
    axes[1].plot(test.index, z_te, color="darkgreen", linewidth=1.2, label="z-score")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axhline(2.0, color="red", linestyle="--", linewidth=0.7, label="entry ±2")
    axes[1].axhline(-2.0, color="red", linestyle="--", linewidth=0.7)
    axes[1].axhline(0.5, color="blue", linestyle=":", linewidth=0.7, label="exit ±0.5")
    axes[1].axhline(-0.5, color="blue", linestyle=":", linewidth=0.7)

    for t in bt.trades:
        color = "green" if t.pnl > 0 else "crimson"
        axes[1].plot(t.entry_date, t.entry_z, "o", color=color, markersize=6)
        axes[1].plot(t.exit_date,  t.exit_z,  "x", color=color, markersize=6)
    axes[1].set_ylabel("z-score")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> plot saved to {outpath}")


def plot_equity_curves(all_results, outpath):
    """Plot cumulative P&L for every pair that was traded."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for res in all_results:
        bt = res["backtest"]
        if bt.n_trades == 0:
            continue
        cum = np.cumsum([t.pnl for t in bt.trades])
        dates = [t.exit_date for t in bt.trades]
        ax.plot(dates, cum * 100,
                marker="o", markersize=3,
                label=f"{res['pair'][0]}/{res['pair'][1]} "
                      f"({bt.success_ratio*100:.0f}% win)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative P&L (% log-return units)")
    ax.set_xlabel("Trade exit date")
    ax.set_title("Out-of-sample equity curves by pair")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> equity curves saved to {outpath}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_prices()
    train, test = split_train_test(df)
    print(f"Training:  {train.index.min().date()}  ->  {train.index.max().date()}  ({len(train)} days)")
    print(f"Test:      {test.index.min().date()}  ->  {test.index.max().date()}  ({len(test)} days)")

    # ---- Pair-selection table ----
    screen_rows = []
    all_results = []
    for a, b in CANDIDATE_PAIRS:
        res = analyse_pair(a, b, train, test)
        if res is None:
            print(f"[skip] {a}/{b}: missing data")
            continue
        if "skipped" in res:
            print(f"[skip] {a}/{b}: {res['skipped']}")
            continue
        eg = res["eg"]; ou = res["ou"]
        row = {
            "pair": f"{a}/{b}",
            "beta": eg["beta"],
            "adf_stat": eg["adf_stat"],
            "cointegrated_5%": eg["is_cointegrated_5pct"],
            "theta_/yr": ou.theta,
            "mu": ou.mu,
            "sigma_/sqrt(yr)": ou.sigma,
            "half_life_days": ou.half_life * 252,
        }
        screen_rows.append(row)
        if eg["is_cointegrated_5pct"] and 1 <= ou.half_life * 252 <= 60:
            all_results.append(res)

    screen_df = pd.DataFrame(screen_rows).round(4)
    print("\n=== Pair screening (training window) ===")
    print(screen_df.to_string(index=False))
    screen_df.to_csv(os.path.join(OUTPUT_DIR, "pair_screening.csv"), index=False)

    # ---- Backtest results ----
    print(f"\n{len(all_results)} pair(s) passed both cointegration and half-life filters.")
    bt_rows = []
    for res in all_results:
        bt = res["backtest"]
        bt_rows.append({
            "pair": f"{res['pair'][0]}/{res['pair'][1]}",
            "n_trades": bt.n_trades,
            "wins": bt.n_wins,
            "success_ratio": bt.success_ratio,
            "avg_pnl_%": bt.avg_pnl * 100,
            "total_pnl_%": bt.total_pnl * 100,
            "sharpe": bt.sharpe,
            "max_drawdown_%": bt.max_drawdown * 100,
            "avg_hold_days": bt.avg_holding_days,
        })
    bt_df = pd.DataFrame(bt_rows).round(3)
    print("\n=== Backtest results (out-of-sample) ===")
    print(bt_df.to_string(index=False))
    bt_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_results.csv"), index=False)

    # ---- Plots ----
    if all_results:
        # best pair = highest total PnL
        best = max(all_results, key=lambda r: r["backtest"].total_pnl)
        print(f"\nBest pair: {best['pair'][0]}/{best['pair'][1]}  "
              f"(total P&L = {best['backtest'].total_pnl*100:.2f}%)")
        plot_best_pair(best, os.path.join(OUTPUT_DIR, "best_pair_spread.png"))
        plot_equity_curves(all_results, os.path.join(OUTPUT_DIR, "equity_curves.png"))

    # ---- Trade log for the best pair ----
    if all_results:
        best = max(all_results, key=lambda r: r["backtest"].total_pnl)
        trade_rows = [{
            "entry_date": t.entry_date.date(),
            "exit_date":  t.exit_date.date(),
            "direction":  "long spread" if t.direction > 0 else "short spread",
            "entry_z":    round(t.entry_z, 2),
            "exit_z":     round(t.exit_z, 2),
            "hold_days":  t.holding_days,
            "pnl_%":      round(t.pnl * 100, 3),
            "exit":       t.exit_reason,
        } for t in best["backtest"].trades]
        trade_df = pd.DataFrame(trade_rows)
        trade_df.to_csv(os.path.join(OUTPUT_DIR, "best_pair_trades.csv"), index=False)
        print(f"\nBest pair trade log ({len(trade_df)} trades) saved.")


if __name__ == "__main__":
    main()
