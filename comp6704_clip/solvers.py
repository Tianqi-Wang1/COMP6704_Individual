"""Three solvers with a common feasible-objective accuracy certificate."""
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from problem import ball, project


def solve(p, basis, radius, method, tolerance, max_iter, log_every, rho):
    start = perf_counter()
    d = np.zeros(p.shape)
    trace = []
    evaluations = 0

    def value_grad(w):
        nonlocal evaluations
        evaluations += 1
        return p.value_grad(w)

    def record(k):
        feasible = project(d, basis, radius)
        violation = max(0., np.linalg.norm(basis.T @ d) - radius)
        value, grad = value_grad(feasible)
        # Minimize the strong-convexity quadratic lower bound on the feasible set.
        delta = project(feasible - grad / p.lam, basis, radius) - feasible
        gap = max(0., -np.vdot(grad, delta) - 0.5 * p.lam * np.vdot(delta, delta))
        if not np.all(np.isfinite([value, gap, violation])):
            raise FloatingPointError(f"Nonfinite solver diagnostics in {method}")
        trace.append(dict(iteration=k, seconds=perf_counter() - start,
                          grad_evals=evaluations, objective=value,
                          gap_upper=gap, violation=violation))
        return gap <= tolerance and violation <= tolerance

    if record(0):
        return d, trace, True
    converged = False
    if method in ("PGD", "APG"):
        v = d
        momentum = 1.
        for k in range(1, max_iter + 1):
            _, grad = value_grad(v)
            previous = d
            d = project(v - grad / p.L, basis, radius)
            if method == "APG":
                next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
                v = d + ((momentum - 1) / next_momentum) * (d - previous)
                momentum = next_momentum
            else:
                v = d
            if k % log_every == 0 or k == max_iter:
                converged = record(k)
                if converged:
                    break
    elif method == "ADMM":
        h = np.zeros((basis.shape[1], d.shape[1]))
        dual = np.zeros_like(h)
        for k in range(1, max_iter + 1):
            def inner(flat):
                w = flat.reshape(p.shape)
                loss, grad = value_grad(w)
                residual = basis.T @ w - h + dual
                loss += 0.5 * rho * np.vdot(residual, residual)
                grad += rho * (basis @ residual)
                return loss, grad.ravel()

            opt = minimize(inner, d.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": 200, "maxls": 50, "gtol": 1e-9, "ftol": 1e-14})
            inner_bound = np.vdot(opt.jac, opt.jac) / (2 * p.lam)
            if not opt.success and (not np.isfinite(inner_bound) or inner_bound > min(1e-8, tolerance / 10)):
                raise RuntimeError(f"ADMM subproblem at iteration {k}: {opt.message}")
            d = opt.x.reshape(p.shape)
            projected_coords = basis.T @ d
            h = ball(projected_coords + dual, radius)
            dual += projected_coords - h
            if k % log_every == 0 or k == max_iter:
                converged = record(k)
                if converged:
                    break
    else:
        raise ValueError(f"Unknown solver {method}")
    return project(d, basis, radius), trace, converged
