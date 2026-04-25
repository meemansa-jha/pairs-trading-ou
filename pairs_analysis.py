"""
pairs_analysis.py
-----------------
Core functions for the pairs-trading project:

  1. engle_granger_test : test whether two log-price series are cointegrated.
  2. fit_ou             : fit the Ornstein-Uhlenbeck parameters (theta, mu, sigma)
                          to a spread series using the AR(1) equivalence.
  3. backtest_pair      : walk-forward backtest of the z-score trading rule,
                          returning a list of trades and summary metrics.

Theory reference: Shreve, "Stochastic Calculus for Finance II", Ch. 3-4.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# 1. Cointegration test (Engle-Granger two-step, ADF on residuals)
# ---------------------------------------------------------------------------
# We implement the ADF test from scratch so we do not need statsmodels.
# The test regresses delta_y_t on y_{t-1} plus lagged deltas and returns the
# t-statistic of the lagged-level coefficient. A more negative value means
# stronger evidence of stationarity.

# MacKinnon 1% / 5% / 10% critical values for the Engle-Granger residual ADF
# test with 2 variables (no constant or trend in the test regression).
# Source: MacKinnon (2010) Table 2, n=2, finite-sample corrected.
EG_CRITICAL_VALUES = {"1%": -3.90, "5%": -3.34, "10%": -3.04}


def _adf_tstat(y, lags=1):
    """Augmented Dickey-Fuller t-statistic on the lagged-level coefficient."""
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)                   # first differences
    n = len(dy)
    # Build regressor matrix: rows correspond to dy[lags:]; columns are
    # [y_{t-1}, dy_{t-1}, ..., dy_{t-lags}].
    y_lag = y[lags:-1] if lags > 0 else y[:-1]
    cols = [y_lag]
    for k in range(1, lags + 1):
        cols.append(dy[lags - k : n - k])
    X = np.column_stack(cols)
    target = dy[lags:]
    # OLS
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    dof = len(target) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    # Covariance of beta = sigma^2 * (X'X)^{-1}
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(cov[0, 0])
    return beta[0] / se


def engle_granger_test(log_price_a, log_price_b):
    """
    Two-step Engle-Granger cointegration test.
    Returns a dict with:
        beta      : hedge ratio from OLS of log_a on log_b (with intercept)
        alpha     : intercept
        residuals : the spread series (log_a - alpha - beta*log_b)
        adf_stat  : ADF t-statistic on residuals
        is_cointegrated_5pct : True if adf_stat < -3.34
    """
    a = np.asarray(log_price_a, dtype=float)
    b = np.asarray(log_price_b, dtype=float)
    # Regress a on [1, b]
    X = np.column_stack([np.ones_like(b), b])
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    alpha, beta = coef
    residuals = a - alpha - beta * b
    adf_stat = _adf_tstat(residuals, lags=1)
    return {
        "alpha": alpha,
        "beta": beta,
        "residuals": residuals,
        "adf_stat": adf_stat,
        "is_cointegrated_5pct": adf_stat < EG_CRITICAL_VALUES["5%"],
    }


# ---------------------------------------------------------------------------
# 2. Ornstein-Uhlenbeck parameter estimation
# ---------------------------------------------------------------------------
# The OU SDE:     dX_t = theta (mu - X_t) dt + sigma dW_t
#
# Exact discretisation at interval dt:
#     X_{t+dt} = X_t * exp(-theta*dt) + mu * (1 - exp(-theta*dt)) + noise
#
# So an AR(1) regression X_{t+dt} = a * X_t + b + eps gives:
#     theta = -ln(a) / dt
#     mu    = b / (1 - a)
#     sigma = sqrt( Var(eps) * 2*theta / (1 - a^2) )

@dataclass
class OUParams:
    theta: float        # mean reversion speed (per year if dt in years)
    mu: float           # long run mean of the spread
    sigma: float        # volatility (per sqrt(year))
    half_life: float    # ln(2) / theta, in the same time units
    stationary_std: float   # sigma / sqrt(2*theta): std of the equilibrium distribution


def fit_ou(spread, dt=1/252):
    """
    Fit OU parameters to a spread series by AR(1) regression.
    dt is the sampling interval in years (1/252 for daily data).
    """
    x = np.asarray(spread, dtype=float)
    x_prev = x[:-1]
    x_next = x[1:]
    # Regress x_next on [1, x_prev]
    X = np.column_stack([np.ones_like(x_prev), x_prev])
    coef, *_ = np.linalg.lstsq(X, x_next, rcond=None)
    b, a = coef
    # Guard against a >= 1 (no mean reversion detected) or a <= 0 (noise)
    if not (0 < a < 1):
        raise ValueError(f"AR(1) slope {a:.4f} outside (0,1): no usable mean reversion.")
    theta = -np.log(a) / dt
    mu = b / (1 - a)
    resid = x_next - (a * x_prev + b)
    eps_var = resid.var(ddof=2)
    sigma = np.sqrt(eps_var * 2 * theta / (1 - a ** 2))
    half_life = np.log(2) / theta
    stationary_std = sigma / np.sqrt(2 * theta)
    return OUParams(theta=theta, mu=mu, sigma=sigma,
                    half_life=half_life, stationary_std=stationary_std)


# ---------------------------------------------------------------------------
# 3. Backtest engine
# ---------------------------------------------------------------------------
# Trading rule:
#   - Open SHORT spread when z > entry_z: short A, long beta units of B.
#   - Open LONG spread when z < -entry_z: long A, short beta units of B.
#   - Exit when |z| < exit_z (take profit) OR |z| > stop_z (stop loss)
#     OR holding > max_hold_days (time stop).
#   - Only one position open at a time.
#
# P&L of a unit spread position over the holding period is computed from
# log-price changes: going long the spread earns (log_A_end - log_A_start)
# minus beta*(log_B_end - log_B_start). Multiplying by notional gives dollar
# P&L; we report in log-return units since it scales cleanly.

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int              # +1 = long spread, -1 = short spread
    entry_z: float
    exit_z: float
    pnl: float                  # net P&L after costs, in log-return units
    exit_reason: str
    holding_days: int


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    pair: tuple = ()
    n_trades: int = 0
    n_wins: int = 0
    success_ratio: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    avg_holding_days: float = 0.0


def backtest_pair(dates, log_a, log_b, alpha, beta, ou: OUParams,
                  entry_z=2.0, exit_z=0.5, stop_z=3.5,
                  max_hold_days=None, cost_per_leg=0.0010,
                  pair_label=("A", "B")):
    """
    Walk forward day-by-day over the test set and execute trades.

    Parameters
    ----------
    dates   : DatetimeIndex of the test set.
    log_a, log_b : log prices of the two legs over the test set.
    alpha, beta  : cointegration intercept and slope (from TRAINING set).
    ou          : OUParams fitted on the TRAINING set.
    entry_z, exit_z, stop_z : z-score thresholds.
    max_hold_days : time stop in trading days. If None, uses 2x half-life.
    cost_per_leg  : transaction cost per leg per trade, as a decimal
                    (0.0010 = 10 basis points).
    """
    spread = np.asarray(log_a) - alpha - beta * np.asarray(log_b)
    z = (spread - ou.mu) / ou.stationary_std
    if max_hold_days is None:
        # Half-life is in years (if dt was 1/252), so convert to days.
        max_hold_days = int(max(5, round(2 * ou.half_life * 252)))

    trades: List[Trade] = []
    position = 0           # 0 flat, +1 long spread, -1 short spread
    entry_i = None
    entry_z_val = None
    # After exiting a trade, we must see |z| drop below entry_z at least once
    # before re-entering. Prevents churning on stop-loss whipsaws.
    armed = True

    for i in range(len(dates)):
        zi = z[i]
        if position == 0:
            # Re-arm once z comes back inside the entry band.
            if not armed and abs(zi) < entry_z:
                armed = True
            # Look for an entry signal (only if armed).
            if armed:
                if zi > entry_z:
                    position = -1      # short the spread (spread too high)
                    entry_i = i
                    entry_z_val = zi
                elif zi < -entry_z:
                    position = 1       # long the spread (spread too low)
                    entry_i = i
                    entry_z_val = zi
        else:
            # Check exit conditions.
            holding = i - entry_i
            exit_now = False
            reason = ""
            if abs(zi) < exit_z:
                exit_now, reason = True, "take_profit"
            elif abs(zi) > stop_z:
                exit_now, reason = True, "stop_loss"
            elif holding >= max_hold_days:
                exit_now, reason = True, "time_stop"

            if exit_now:
                # P&L of long-spread position: d(log_a) - beta*d(log_b)
                d_log_a = log_a[i] - log_a[entry_i]
                d_log_b = log_b[i] - log_b[entry_i]
                gross = position * (d_log_a - beta * d_log_b)
                # Two legs traded at entry, two at exit => 4 * cost_per_leg
                # But we scale by (1 + beta) for the size of leg B.
                cost = 2 * cost_per_leg * (1 + abs(beta))
                net = gross - cost
                trades.append(Trade(
                    entry_date=dates[entry_i], exit_date=dates[i],
                    direction=position, entry_z=entry_z_val, exit_z=zi,
                    pnl=net, exit_reason=reason, holding_days=holding,
                ))
                position = 0
                entry_i = None
                entry_z_val = None
                armed = False      # require z to re-enter the ±entry_z band before next trade

    # Force-close any open position at the end of the test window.
    if position != 0 and entry_i is not None:
        i = len(dates) - 1
        d_log_a = log_a[i] - log_a[entry_i]
        d_log_b = log_b[i] - log_b[entry_i]
        gross = position * (d_log_a - beta * d_log_b)
        cost = 2 * cost_per_leg * (1 + abs(beta))
        net = gross - cost
        trades.append(Trade(
            entry_date=dates[entry_i], exit_date=dates[i],
            direction=position, entry_z=entry_z_val, exit_z=z[i],
            pnl=net, exit_reason="end_of_backtest",
            holding_days=i - entry_i,
        ))

    return _summarise(trades, pair_label)


def _summarise(trades, pair_label):
    r = BacktestResult(trades=trades, pair=pair_label)
    r.n_trades = len(trades)
    if r.n_trades == 0:
        return r
    pnls = np.array([t.pnl for t in trades])
    holds = np.array([t.holding_days for t in trades])
    r.n_wins = int(np.sum(pnls > 0))
    r.success_ratio = r.n_wins / r.n_trades
    r.avg_pnl = float(pnls.mean())
    r.total_pnl = float(pnls.sum())
    r.avg_holding_days = float(holds.mean())
    # Sharpe from per-trade returns (rough, annualised assuming ~n_trades/year)
    if r.n_trades >= 2 and pnls.std(ddof=1) > 1e-12:
        # trades per year = 252 / avg_holding_days (very rough)
        trades_per_year = 252 / max(r.avg_holding_days, 1)
        r.sharpe = float(pnls.mean() / pnls.std(ddof=1) * np.sqrt(trades_per_year))
    # Max drawdown on the cumulative P&L curve
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    r.max_drawdown = float(dd.min())
    return r
