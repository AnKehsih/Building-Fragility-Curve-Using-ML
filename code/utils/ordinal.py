"""
Ridge-penalised ordinal logistic regression (proportional-odds / cumulative logit model).

    P(y <= k | x) = sigmoid(theta_k - x @ beta),   k = 1..K-1,   theta_1 < ... < theta_{K-1}
    P(y = k)      = P(y <= k) - P(y <= k-1)

Fitted by minimising   sum_i -log P(y_i | x_i)  +  (alpha / 2) * ||beta||^2
(thresholds theta are not penalised) with L-BFGS and the exact gradient. Thresholds are
parameterised as theta_1 + cumulative exp(gaps), so they stay ordered. x should be standardised
(the penalty treats all coefficients alike).

Run `python code/utils/ordinal.py` for a gradient check and a parameter-recovery test.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class OrdinalLogit:
    def __init__(self, alpha=1.0, n_classes=5, max_iter=1000, tol=1e-7):
        self.alpha, self.K, self.max_iter, self.tol = alpha, n_classes, max_iter, tol

    # ---- parameters <-> thresholds
    def _thetas(self, t):
        return t[0] + np.concatenate([[0.0], np.cumsum(np.exp(t[1:]))])

    def _loss_grad(self, w, X, y0):
        """y0 = class index 0..K-1. Returns (loss, gradient) for w = [beta (p), t (K-1)]."""
        p, K = X.shape[1], self.K
        beta, t = w[:p], w[p:]
        theta = self._thetas(t)
        eta = X @ beta
        th_ext = np.concatenate([[-np.inf], theta, [np.inf]])        # theta_0 = -inf, theta_K = +inf
        a, b = th_ext[y0 + 1] - eta, th_ext[y0] - eta                 # upper and lower cut, per row
        Fa, Fb = expit(a), expit(b)
        prob = np.clip(Fa - Fb, 1e-300, None)
        fa, fb = Fa * (1 - Fa), Fb * (1 - Fb)                          # logistic density (0 at +-inf)
        loss = -np.log(prob).sum() + 0.5 * self.alpha * beta @ beta

        g_eta = (fa - fb) / prob                                        # d(-log p)/d eta
        g_beta = X.T @ g_eta + self.alpha * beta
        # d(-log p)/d theta_j: -fa/p where j is the upper cut, +fb/p where j is the lower cut
        g_theta = (np.bincount(y0, weights=-fa / prob, minlength=K + 1)[:K - 1]
                   + np.bincount(y0, weights=fb / prob, minlength=K + 1)[1:K])
        # chain rule to t: theta_j = t0 + sum_{i<j} exp(t_i)
        g_t = np.empty(K - 1)
        g_t[0] = g_theta.sum()
        tail = np.cumsum(g_theta[::-1])[::-1]                           # sum_{j >= i} g_theta_j
        g_t[1:] = np.exp(t[1:]) * tail[1:]
        return loss, np.concatenate([g_beta, g_t])

    def fit(self, X, y):
        X = np.asarray(X, float)
        y0 = np.asarray(y, int) - 1
        counts = np.bincount(y0, minlength=self.K)
        cum = np.cumsum(counts)[:-1] / counts.sum()
        theta0 = np.log(cum / (1 - cum))                                 # thresholds of the null model
        t0 = np.concatenate([[theta0[0]], np.log(np.maximum(np.diff(theta0), 1e-3))])
        w0 = np.concatenate([np.zeros(X.shape[1]), t0])
        res = minimize(self._loss_grad, w0, args=(X, y0), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter, "gtol": self.tol})
        self.coef_, self.theta_ = res.x[:X.shape[1]], self._thetas(res.x[X.shape[1]:])
        self.classes_ = np.arange(1, self.K + 1)
        self.converged_, self.n_iter_ = bool(res.success), int(res.nit)
        return self

    def predict_proba(self, X):
        eta = np.asarray(X, float) @ self.coef_
        cdf = expit(self.theta_[None, :] - eta[:, None])
        cdf = np.column_stack([np.zeros(len(eta)), cdf, np.ones(len(eta))])
        return np.clip(np.diff(cdf, axis=1), 1e-12, 1)


if __name__ == "__main__":
    from scipy.optimize import check_grad
    rng = np.random.default_rng(0)
    n, p = 20000, 4
    X = rng.normal(size=(n, p))
    beta_true, theta_true = np.array([1.0, -0.5, 0.25, 0.0]), np.array([-1.5, -0.3, 0.6, 1.8])
    eta = X @ beta_true
    u = rng.logistic(size=n)
    y = 1 + (eta + u > theta_true[:, None]).sum(0)                       # latent-variable simulation

    m = OrdinalLogit(alpha=1e-6)
    y0 = y - 1
    w = rng.normal(size=p + 4) * 0.3
    err = check_grad(lambda v: m._loss_grad(v, X[:500], y0[:500])[0],
                     lambda v: m._loss_grad(v, X[:500], y0[:500])[1], w)
    print(f"gradient check error: {err:.2e}")
    assert err < 1e-4

    m.fit(X, y)
    print("beta ", np.round(m.coef_, 3), " true", beta_true)
    print("theta", np.round(m.theta_, 3), " true", theta_true)
    assert np.allclose(m.coef_, beta_true, atol=0.05) and np.allclose(m.theta_, theta_true, atol=0.07)
    P = m.predict_proba(X)
    assert np.allclose(P.sum(1), 1)
    print("ordinal logit self-test passed")
