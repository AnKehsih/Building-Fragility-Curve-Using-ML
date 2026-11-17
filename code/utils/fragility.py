"""
Baselines (CONTEXT 4).

MajorityModel   : predicts the training-fold grade distribution for every building.
FragilityModel  : empirical lognormal fragility curves, one set per structural type c,
                  fitted by maximum likelihood on training data only:

    P(DG >= k | PGA, c) = Phi( (ln PGA - ln theta_{c,k}) / beta_c ),  k = 2..5
    P(DG = k)           = P(DG >= k) - P(DG >= k+1),  with P(DG >= 1) = 1, P(DG >= 6) = 0

  theta_{c,k} = median PGA (g) for reaching grade k (increasing in k); beta_c = one dispersion per
  type (parallel curves on a log axis). This is an ordered probit with ln PGA as the only input.
  Because PGA is constant within a municipality, the likelihood is computed on counts per
  (PGA value, grade); this gives exactly the same estimate as building-by-building, much faster.
  Types with fewer than `min_n` training buildings fall back to one curve set pooled over all types.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

K = 5
FLAT_BETA = 3.0


class MajorityModel:
    def fit(self, y):
        counts = np.bincount(np.asarray(y, int), minlength=K + 1)[1:]
        self.p_ = counts / counts.sum()
        return self

    def predict_proba(self, n):
        return np.tile(self.p_, (n, 1))


def _unpack(x):
    """x = [log(1/beta), ln theta_2, log gaps 3..5] -> beta, increasing ln theta_2..5."""
    beta = np.exp(-x[0])
    ln_theta = x[1] + np.concatenate([[0.0], np.cumsum(np.exp(x[2:]))])
    return beta, ln_theta


def grade_probs(ln_pga, beta, ln_theta):
    exceed = norm.cdf((np.asarray(ln_pga)[:, None] - ln_theta[None, :]) / beta)   # (n, 4): k = 2..5
    ge = np.column_stack([np.ones(len(exceed)), exceed, np.zeros(len(exceed))])      # P(DG >= 1..6)
    return np.clip(ge[:, :-1] - ge[:, 1:], 1e-12, 1.0)


def fit_curves(ln_pga, counts):
    """MLE of (beta, ln theta_2..5) from distinct ln PGA values (m,) and grade counts (m, 5)."""
    def nll(x):
        beta, ln_theta = _unpack(x)
        return -(counts * np.log(grade_probs(ln_pga, beta, ln_theta))).sum() / counts.sum()

    # start: beta = 0.6; thresholds spread around the mean ln PGA
    x0 = np.array([np.log(1 / 0.6), ln_pga.mean() - 1.0, np.log(0.5), np.log(0.5), np.log(0.5)])
    res = minimize(nll, x0, method="L-BFGS-B", bounds=[(-5, 5), (-10, 5), (-8, 3), (-8, 3), (-8, 3)])
    beta, ln_theta = _unpack(res.x)
    return {"beta": beta, "ln_theta": ln_theta, "nll_per_building": res.fun, "converged": bool(res.success),
            "beta_at_bound": bool(np.isclose(res.x[0], -5) or np.isclose(res.x[0], 5))}


def _aggregate(df):
    t = pd.crosstab(np.log(df["pga"]).round(10), df["damage_grade"]).reindex(columns=range(1, K + 1), fill_value=0)
    return t.index.to_numpy(float), t.to_numpy(float)


class FragilityModel:
    def __init__(self, type_col="structural_type", min_n=200):
        self.type_col, self.min_n = type_col, min_n

    def fit(self, df):
        self.params_, self.n_ = {}, df[self.type_col].value_counts().to_dict()
        self.pooled_ = fit_curves(*_aggregate(df))
        for c, sub in df.groupby(self.type_col, observed=True):
            if len(sub) >= self.min_n:
                self.params_[c] = fit_curves(*_aggregate(sub))
        return self

    def params_for(self, c):
        return self.params_.get(c, self.pooled_)

    def predict_proba(self, df):
        P = np.empty((len(df), K))
        types = df[self.type_col].astype(object).to_numpy()
        ln_pga = np.log(df["pga"].to_numpy(float))
        for c in pd.unique(types):
            m = types == c
            p = self.params_for(c)
            P[m] = grade_probs(ln_pga[m], p["beta"], p["ln_theta"])
        return P / P.sum(axis=1, keepdims=True)

    def table(self):
        rows = []
        for c, p in list(self.params_.items()) + [("_pooled", self.pooled_)]:
            rows.append({"structural_type": c, "n_train": self.n_.get(c, sum(self.n_.values())),
                         "beta": p["beta"], **{f"theta_g{k}": np.exp(t) for k, t in zip(range(2, 6), p["ln_theta"])},
                         "converged": p["converged"], "beta_at_bound": p["beta_at_bound"],
                         # beta > FLAT_BETA: damage barely changes with PGA over the observed range,
                         # so the fit encodes near-constant probabilities and theta has no physical meaning
                         "flat_curve": p["beta"] > FLAT_BETA})
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Parameter-recovery self-test on simulated data with known curves.
    rng = np.random.default_rng(0)
    beta_true, theta_true = 0.5, np.array([0.10, 0.20, 0.35, 0.60])
    ln_pga = np.log(rng.uniform(0.05, 0.9, 60))
    P = grade_probs(ln_pga, beta_true, np.log(theta_true))
    counts = np.array([rng.multinomial(2000, p) for p in P], float)
    fit = fit_curves(ln_pga, counts)
    print(f"beta {fit['beta']:.3f} (true {beta_true}); theta {np.round(np.exp(fit['ln_theta']), 3)} (true {theta_true})")
    assert abs(fit["beta"] - beta_true) < 0.03 and np.allclose(np.exp(fit["ln_theta"]), theta_true, rtol=0.05)
    print("fragility self-test passed")
