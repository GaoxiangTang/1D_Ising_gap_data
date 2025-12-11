import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- helpers ----------
def log_mean_stats(df, col="eps2"):
    rows=[]
    for L, g in df.groupby("L"):
        v = g[col].to_numpy()
        v = v[v > 0]
        if v.size == 0:
            continue
        logs = np.log(v)
        n = logs.size
        mu = logs.mean()
        se = logs.std(ddof=1)/np.sqrt(n) if n > 1 else np.nan
        rows.append((float(L), mu, se, n))
    out = pd.DataFrame(rows, columns=["L","mu","se","n"]).sort_values("L").reset_index(drop=True)
    return out

def weights_from_se(se):
    w = np.zeros_like(se, dtype=float)
    finite = np.isfinite(se) & (se > 0)
    if finite.any():
        w[finite] = 1.0/(se[finite]**2)
        medw = np.median(w[finite])
        w[~finite] = medw
    else:
        w[:] = 1.0
    return w

# ---------- models ----------
# Hypothesis 1 (Activated): GM ≈ A/√L * exp(-c√L)  -> mu = lnA - c√L - 0.5 ln L
def fit_activated(L, mu, w=None):
    X = np.sqrt(L)
    y = mu + 0.5*np.log(L)
    if w is None:
        b, a = np.polyfit(X, y, 1)      # y = a + b X
    else:
        b, a = np.polyfit(X, y, 1, w=w)
    lnA, c = a, -b
    return lnA, c

def pred_activated(L, lnA, c):
    return lnA - c*np.sqrt(L) - 0.5*np.log(L)

# Hypothesis 2 (Power law): GM ≈ B * L^{-psi}  -> mu = lnB - psi ln L
def fit_powerlaw(L, mu, w=None):
    X = np.log(L)
    y = mu
    if w is None:
        b, a = np.polyfit(X, y, 1)      # y = a + b X
    else:
        b, a = np.polyfit(X, y, 1, w=w)
    lnB, psi = a, -b
    return lnB, psi

def pred_powerlaw(L, lnB, psi):
    return lnB - psi*np.log(L)

# ---------- errors ----------
def rmse_log(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))

def wrmse_log(y, yhat, w):
    return np.sqrt(np.sum(w*(y - yhat)**2) / np.sum(w))

# ---------- cumulative learning curves ----------
def cumulative_curves(L, mu, se, min_pts=3, use_weights=False):
    order = np.argsort(L)
    Ls, mus, ses = L[order], mu[order], se[order]

    cum_L, err_act, err_pow = [], [], []
    for k in range(min_pts, len(Ls)+1):
        L_train, mu_train, se_train = Ls[:k], mus[:k], ses[:k]
        if use_weights:
            w = weights_from_se(se_train)
            lnA, c   = fit_activated(L_train, mu_train, w=w)
            lnB, psi = fit_powerlaw(L_train,  mu_train, w=w)
            e_act = wrmse_log(mu_train, pred_activated(L_train, lnA, c), w)
            e_pow = wrmse_log(mu_train, pred_powerlaw(L_train, lnB, psi), w)
        else:
            lnA, c   = fit_activated(L_train, mu_train, w=None)
            lnB, psi = fit_powerlaw(L_train,  mu_train, w=None)
            e_act = rmse_log(mu_train, pred_activated(L_train, lnA, c))
            e_pow = rmse_log(mu_train, pred_powerlaw(L_train, lnB, psi))
        cum_L.append(L_train[-1]); err_act.append(e_act); err_pow.append(e_pow)

    return pd.DataFrame({
        "L": np.array(cum_L),
        "RMSE_activated_log": np.array(err_act),
        "RMSE_powerlaw_log":  np.array(err_pow)
    })

# ---------- compute L* for one sigma file ----------
def compute_L_star_for_file(csv_path, min_pts=3, use_weights=False):
    df = pd.read_csv(csv_path)
    tbl = log_mean_stats(df, "eps2")
    L, mu, se = tbl["L"].to_numpy(), tbl["mu"].to_numpy(), tbl["se"].to_numpy()
    curves = cumulative_curves(L, mu, se, min_pts=min_pts, use_weights=use_weights)
    better = curves["RMSE_powerlaw_log"] < curves["RMSE_activated_log"]
    if better.any():
        return float(curves.loc[better, "L"].max())
    else:
        return np.nan  # no L where power-law beats activated
    

def estimate_L_star(cum_df):
    """First L where activated beats power law (or closest by diff)."""
    diff = cum_df["RMSE_powerlaw_log"].to_numpy() - cum_df["RMSE_activated_log"].to_numpy()
    idx = np.where(diff < 0)[0]
    if idx.size > 0:
        return float(cum_df["L"].iloc[idx[-1]])
    # fallback: where the (power-activated) diff is minimal
    return float(cum_df["L"].iloc[np.argmin(diff)])


# ---------- R^2 metrics on log-scale ----------
def r2_log(y, yhat):
    """
    Unweighted R^2 on the log targets (mu).
    Returns 1 - SSE/SST. If SST==0, returns 1.0 if perfect fit else np.nan.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    sse = np.sum((y - yhat)**2)
    ybar = np.mean(y)
    sst = np.sum((y - ybar)**2)
    if not np.isfinite(sst) or sst <= 0:
        return 1.0 if np.isclose(sse, 0.0) else np.nan
    return 1.0 - sse / sst

def wr2_log(y, yhat, w):
    """
    Weighted R^2 on the log targets (mu).
    Uses weighted mean for SST and w-SSE/w-SST.
    If w-SST==0, returns 1.0 if perfect fit else np.nan.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, np.inf)
    wsum = np.sum(w)
    if not np.isfinite(wsum) or wsum <= 0:
        return np.nan
    ybar = np.sum(w * y) / wsum
    sse = np.sum(w * (y - yhat)**2)
    sst = np.sum(w * (y - ybar)**2)
    if not np.isfinite(sst) or sst <= 0:
        return 1.0 if np.isclose(sse, 0.0) else np.nan
    return 1.0 - sse / sst

# ---------- cumulative learning curves (R^2 version) ----------
def cumulative_curves_r2(L, mu, se, min_pts=3, use_weights=False):
    """
    Same as cumulative_curves but computes R^2 instead of RMSE.
    Returns a DataFrame with columns: L, R2_activated_log, R2_powerlaw_log.
    """
    order = np.argsort(L)
    Ls, mus, ses = L[order], mu[order], se[order]

    cum_L, r2_act, r2_pow = [], [], []
    for k in range(min_pts, len(Ls)+1):
        L_train, mu_train, se_train = Ls[:k], mus[:k], ses[:k]
        if use_weights:
            w = weights_from_se(se_train)
            lnA, c   = fit_activated(L_train, mu_train, w=w)
            lnB, psi = fit_powerlaw(L_train,  mu_train, w=w)
            yhat_act = pred_activated(L_train, lnA, c)
            yhat_pow = pred_powerlaw(L_train, lnB, psi)
            r_act = wr2_log(mu_train, yhat_act, w)
            r_pow = wr2_log(mu_train, yhat_pow, w)
        else:
            lnA, c   = fit_activated(L_train, mu_train, w=None)
            lnB, psi = fit_powerlaw(L_train,  mu_train, w=None)
            yhat_act = pred_activated(L_train, lnA, c)
            yhat_pow = pred_powerlaw(L_train, lnB, psi)
            r_act = r2_log(mu_train, yhat_act)
            r_pow = r2_log(mu_train, yhat_pow)
        cum_L.append(L_train[-1]); r2_act.append(r_act); r2_pow.append(r_pow)

    return pd.DataFrame({
        "L": np.array(cum_L),
        "R2_activated_log": np.array(r2_act),
        "R2_powerlaw_log":  np.array(r2_pow)
    })

# ---------- compute L* for one sigma file (R^2 version) ----------
def compute_L_star_for_file_r2(csv_path, min_pts=3, use_weights=False):
    """
    Largest L where power law has higher R^2 than activated.
    Mirrors compute_L_star_for_file but with R^2 (higher is better).
    """
    df = pd.read_csv(csv_path)
    tbl = log_mean_stats(df, "eps2")
    L, mu, se = tbl["L"].to_numpy(), tbl["mu"].to_numpy(), tbl["se"].to_numpy()
    curves = cumulative_curves_r2(L, mu, se, min_pts=min_pts, use_weights=use_weights)
    better = curves["R2_powerlaw_log"] > curves["R2_activated_log"]
    if better.any():
        return float(curves.loc[better, "L"].max())
    else:
        return np.nan

# ---------- estimate L* from a cumulative R^2 table ----------
def estimate_L_star_r2(cum_df):
    """
    First L where activated beats power law by R^2.
    Fallback: pick L where (R2_activated - R2_powerlaw) is maximal.
    """
    diff = (cum_df["R2_activated_log"] - cum_df["R2_powerlaw_log"]).to_numpy()
    idx = np.where(diff < 0)[0]
    if idx.size > 0:
        return float(cum_df["L"].iloc[idx[-1]])
    return float(cum_df["L"].iloc[np.argmax(diff)])


def plot_r2_from_df(df, *, ax=None, min_pts=3, use_weights=False,
                    title=None, xlabel="Largest size in fit (≤ L)"):
    """
    Plot cumulative R^2 curves (activated vs power-law) from a raw eps2 dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ['L', 'eps2'] with multiple reps per L.
    ax : matplotlib.axes.Axes or None
        If None, creates a new figure/axes.
    min_pts : int
        Minimum number of distinct L values to start cumulative fits.
    use_weights : bool
        Passed to cumulative_curves_r2 (weights from 'se' if True).
    title : str or None
        Optional axes title. If None, no title is set.
    xlabel : str
        X-axis label.

    Returns
    -------
    Lstar : float
        Estimated L* from estimate_L_star_r2(cum_df).
    cum_df : pd.DataFrame
        Output from cumulative_curves_r2 with columns:
        ['L','R2_activated_log','R2_powerlaw_log', ...]
    ax : matplotlib.axes.Axes
        The axes used for plotting.
    """
    # 1) aggregate per-L on the log scale
    tbl = log_mean_stats(df, "eps2")
    L   = tbl["L"].to_numpy()
    mu  = tbl["mu"].to_numpy()
    se  = tbl["se"].to_numpy()

    # 2) cumulative R^2 tables and L*
    cum_df = cumulative_curves_r2(L, mu, se, min_pts=min_pts, use_weights=use_weights)
    Lstar  = estimate_L_star_r2(cum_df)

    # 3) plotting
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
        created_ax = True

    ax.plot(cum_df["L"], cum_df["R2_activated_log"],
            marker="o", ms=3, lw=1.5, label=r"Activated  $A/\sqrt{L}\cdot e^{-c\sqrt{L}}$")
    ax.plot(cum_df["L"], cum_df["R2_powerlaw_log"],
            marker="s", ms=3, lw=1.5, label=r"Power law  $B\cdot L^{-\psi}$")

    # vertical line at L*
    # inside your plotting code
    ax.axvline(Lstar, linestyle="--", linewidth=1)
    ax.text(Lstar, ax.get_ylim()[1]*0.92, f"L*≈{int(Lstar)}",
        ha="center", va="top", fontsize=9, transform=ax.transData)  # inside axes


    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    if created_ax:
        plt.tight_layout()

    return Lstar, cum_df, ax
