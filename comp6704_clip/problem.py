"""Regularized softmax regression with a text-subspace constraint."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def ball(a, radius):
    norm = np.linalg.norm(a)
    return a if norm <= radius else a * (radius / norm)


def project(d, basis, radius):
    a = basis.T @ d
    return d + basis @ (ball(a, radius) - a)


class Problem:
    def __init__(self, x, y, text, scale, lam):
        # text is stored with one class per row. d uses one class per column.
        self.x = scale * np.asarray(x, dtype=np.float64)
        self.y = y
        self.lam = lam
        self.base = self.x @ text.T
        self.shape = (x.shape[1], text.shape[0])
        self.rows = np.arange(len(y))
        self.L = lam + np.linalg.norm(self.x, ord=2) ** 2 / (2 * len(y))

    def value_grad(self, d):
        logits = self.base + self.x @ d
        normalizer = logsumexp(logits, axis=1)
        loss = np.mean(normalizer - logits[self.rows, self.y])
        p = np.exp(logits - normalizer[:, None])
        p[self.rows, self.y] -= 1.
        grad = self.x.T @ p / len(self.y) + self.lam * d
        return loss + 0.5 * self.lam * np.vdot(d, d), grad

    def free_solution(self):
        def flat(v):
            loss, grad = self.value_grad(v.reshape(self.shape))
            return loss, grad.ravel()

        opt = minimize(flat, np.zeros(self.shape).ravel(), jac=True, method="L-BFGS-B",
                       options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 2000, "maxls": 50})
        # Strong convexity gives F(d)-F* <= ||grad F(d)||^2 / (2 lambda).
        bound = np.vdot(opt.jac, opt.jac) / (2 * self.lam)
        if not np.isfinite(bound) or bound > 1e-7:
            raise RuntimeError(f"Unconstrained reference inaccurate (gap bound {bound:.3g}): {opt.message}")
        return opt.x.reshape(self.shape)


def text_basis(text, rank):
    u, singular_values, _ = np.linalg.svd(text.T, full_matrices=False)
    cutoff = np.finfo(float).eps * max(text.shape) * singular_values[0]
    if not 1 <= rank <= np.sum(singular_values > cutoff):
        raise ValueError("Requested basis rank exceeds the numerical text rank.")
    return u[:, :rank].copy()


def accuracy(x, y, weights):
    return 100. * np.mean(np.argmax(x @ weights, axis=1) == y)
