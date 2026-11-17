"""
Evaluation metrics (CONTEXT 5.3). P is an (n, 5) array of predicted grade probabilities
(columns = grades 1..5, rows sum to 1); y holds observed grades as integers 1..5.

  rps            ranked probability score per building (primary; lower is better)
  brier_severe   Brier score for severe damage, P(grade >= 4)
  evaluate       all metrics for one validation fold, as a dict
Run `python code/utils/metrics.py` for a hand-checked self-test.
"""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss

K = 5
SEVERE_MIN_GRADE = 4


def _check(P, y):
    P, y = np.asarray(P, float), np.asarray(y, int)
    assert P.ndim == 2 and P.shape[1] == K and len(P) == len(y)
    assert np.allclose(P.sum(1), 1, atol=1e-6), "probabilities must sum to 1"
    assert y.min() >= 1 and y.max() <= K
    return P, y


def rps(P, y):
    """RPS_i = 1/(K-1) * sum_k (F_ik - O_ik)^2, F = predicted cumulative P(grade <= k),
    O = observed cumulative indicator. The k = K term is always 0 and is dropped."""
    P, y = _check(P, y)
    F = np.cumsum(P, axis=1)[:, :K - 1]
    O = (np.arange(1, K)[None, :] >= y[:, None]).astype(float)
    return ((F - O) ** 2).sum(axis=1) / (K - 1)


def p_severe(P):
    return np.asarray(P)[:, SEVERE_MIN_GRADE - 1:].sum(axis=1)


def brier_severe(P, y):
    P, y = _check(P, y)
    return float(np.mean((p_severe(P) - (y >= SEVERE_MIN_GRADE)) ** 2))


def evaluate(P, y):
    """All CONTEXT 5.3 metrics for one set of predictions. Hard labels = most probable grade."""
    P, y = _check(P, y)
    yhat = P.argmax(axis=1) + 1
    return {
        "n": len(y),
        "rps": float(rps(P, y).mean()),
        "brier_severe": brier_severe(P, y),
        "log_loss": float(log_loss(y, np.clip(P, 1e-12, 1), labels=list(range(1, K + 1)))),
        "qwk": float(cohen_kappa_score(y, yhat, weights="quadratic")),
        "macro_f1": float(f1_score(y, yhat, average="macro", labels=list(range(1, K + 1)), zero_division=0)),
        "accuracy": float(accuracy_score(y, yhat)),
    }


if __name__ == "__main__":
    # Hand-checked cases.
    perfect = np.eye(K)[[0, 4]]                       # certain and correct
    assert np.allclose(rps(perfect, [1, 5]), 0)
    uniform = np.full((1, K), 0.2)                     # y = 1: F = .2,.4,.6,.8, O = 1,1,1,1
    assert np.isclose(rps(uniform, [1])[0], (0.64 + 0.36 + 0.16 + 0.04) / 4)   # 0.30
    assert np.isclose(rps(uniform, [3])[0], (0.04 + 0.16 + 0.16 + 0.04) / 4)   # 0.10
    wrong = np.eye(K)[[4]]                             # certain grade 5, truth grade 1: worst case
    assert np.isclose(rps(wrong, [1])[0], 1.0)
    near = np.eye(K)[[1]]                              # certain grade 2, truth 1: ordinal-aware, small
    assert np.isclose(rps(near, [1])[0], 0.25)
    assert np.isclose(brier_severe(uniform, [5]), (0.4 - 1) ** 2)
    print("metrics self-test passed")
