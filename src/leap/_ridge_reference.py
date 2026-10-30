"""Source-standardized dual Ridge fitting and rank-based evaluation metrics."""
import math, time, sys
import numpy as np
FEATURE_NAME="concat_block30_block32"
KINDS=("log","sqrt","identity")
ALPHAS=(.0001,.001,.01,.1,1.,10.)
def peak_rss_kib():
    try:
        import resource
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value/1024 if sys.platform=="darwin" else value)
    except ImportError: return None


def rank(values):
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        out[order[start:end]] = (start + end - 1) / 2
        start = end
    return out

def metric(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    rho = None if len(truth) < 2 or np.ptp(truth) == 0 or np.ptp(prediction) == 0 else float(np.corrcoef(rank(truth), rank(prediction))[0, 1])
    numerator = truth_pairs = prediction_pairs = 0
    for i in range(len(truth) - 1):
        dy = np.sign(truth[i + 1:] - truth[i])
        dp = np.sign(prediction[i + 1:] - prediction[i])
        numerator += float((dy * dp).sum())
        truth_pairs += int(np.count_nonzero(dy))
        prediction_pairs += int(np.count_nonzero(dp))
    return dict(n=len(truth), mae=float(np.abs(truth-prediction).mean()) if len(truth) else None,
                rmse=float(np.sqrt(np.mean((truth-prediction)**2))) if len(truth) else None,
                spearman=rho, kendall_tau_b=numerator/math.sqrt(truth_pairs*prediction_pairs) if truth_pairs*prediction_pairs else None)

def inverse(z, kind, budget=2048):
    if kind not in KINDS or not 1 <= budget <= 2048:
        raise ValueError("Invalid transform or output budget")
    z = np.asarray(z)
    if not np.isfinite(z).all():
        raise ValueError("Prediction must be finite")
    if kind == "log":
        return np.exp(np.clip(z, 0, np.log(budget)))
    if kind == "sqrt":
        return np.clip(z, 1, math.sqrt(budget))**2
    return np.clip(z, 1, budget)

def fit_heads(raw, train_indices, val_indices, ytrain, yval, alphas=ALPHAS):
    """Fit all target mappings with shared source standardization and eigensolve."""
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    x = raw[train_indices].astype(np.float64)
    mean, scale = x.mean(0), x.std(0, ddof=1) + 1e-6
    x = (x - mean) / scale
    targets = np.column_stack([np.log(ytrain), np.sqrt(ytrain), ytrain])
    center = targets.mean(0)
    eigenvalues, eigenvectors = np.linalg.eigh(x @ x.T / x.shape[1])
    projection = eigenvectors.T @ (targets-center)
    xval = (raw[val_indices].astype(np.float64)-mean)/scale
    candidates, best = [], [None] * 3
    for alpha in alphas:
        dual = eigenvectors @ (projection / (np.maximum(eigenvalues, 0)+alpha)[:, None])
        coefficients = x.T @ dual / x.shape[1]
        val_z = xval @ coefficients + center
        for j, kind in enumerate(KINDS):
            score = float(np.abs(inverse(val_z[:, j], kind)-yval).mean())
            candidates.append(dict(feature=FEATURE_NAME, mapping=kind, alpha=alpha, validation_mean20_mae=score))
            if best[j] is None or score < best[j]["validation_mean20_mae"]:
                raw_w = coefficients[:, j] / scale
                best[j] = dict(mapping=kind, alpha=alpha, validation_mean20_mae=score,
                               w=raw_w.copy(), b=float(center[j]-mean @ raw_w))
    costs = dict(fit_all_three_mappings_shared_wall_s=time.perf_counter()-wall_start,
                 fit_all_three_mappings_shared_cpu_s=time.process_time()-cpu_start,
                 process_peak_rss_kib=peak_rss_kib())
    return best, candidates, costs, mean, scale
