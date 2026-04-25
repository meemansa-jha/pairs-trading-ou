# Pairs Trading via Ornstein–Uhlenbeck Mean Reversion

A continuous-time stochastic-calculus approach to pairs trading,
developed for the Stochastic Calculus for Finance course.

The full theoretical derivation, methodology, and empirical results
are in **[final_report.pdf](final_report.pdf)**.

## Authors

- Vedika Jain- 2024B4A81111G
- Meemansa Jha- 2024B4A71096G

## Method

We model the log-price spread between two cointegrated stocks as an
Ornstein–Uhlenbeck process

    dX(t) = θ(μ − X(t)) dt + σ dW(t),

derive its closed-form solution via the Itô–Doeblin formula, calibrate
its parameters by AR(1) regression on training data, and trade a z-score
mean-reversion rule out-of-sample.

## Repository contents

| File | Description |
|------|-------------|
| `report.pdf` / `report.tex` | The main project report |
| `data_loader.py` | Downloads daily prices from Yahoo Finance |
| `pairs_analysis.py` | Cointegration test, OU calibration, backtest engine |
| `run_project.py` | End-to-end pipeline that produces all results |
| `prices.csv` | The historical price data used in the analysis |
| `output/` | Results: tables, plots, trade logs |

## How to reproduce

```bash
pip install yfinance pandas numpy matplotlib scipy
python data_loader.py     # downloads fresh data into prices.csv
python run_project.py     # runs full pipeline, populates output/
```

## Key results

- 2 of 8 candidate pairs passed the Engle–Granger cointegration test at 5%:
  **MA/V** and **RELIANCE.NS/ONGC.NS**.
- Out-of-sample backtest (1 year): 3 trades, 2 wins (success ratio 66.7%),
  net P&L +2.07% after 10 bp/leg transaction costs.
- The MA/V trade history demonstrates both successful mean reversion
  (Trade 1, +4.7%) and a controlled stop-loss on a temporary cointegration
  breakdown (Trade 2, −2.7%).

See `final_report.pdf` for the full discussion.

## References

- Shreve, S. E. (2004). *Stochastic Calculus for Finance II: Continuous-Time Models.* Springer.
- Engle & Granger (1987). Co-integration and error correction. *Econometrica* 55(2).
- Gatev, Goetzmann & Rouwenhorst (2006). Pairs trading. *RFS* 19(3).
