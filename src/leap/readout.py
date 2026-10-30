"""Portable interfaces around the original source-standardized dual Ridge fit.

Regularization uses the research implementation's alpha * feature_dimension.
The deployed coefficients operate on raw features; never standardize them twice.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import numpy as np
from ._ridge_reference import fit_heads, inverse


def matrix(value, width=None):
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite feature matrix')
    if width is not None and x.shape[1] != width:
        raise ValueError('Feature dimension differs from the trained head')
    return x


def labels(value, count, budget=2048):
    y = np.asarray(value, dtype=np.float64)
    if y.shape != (count,) or not np.isfinite(y).all() or (y < 1).any() or (y > budget).any():
        raise ValueError('Length labels must be finite and within the declared output budget')
    return y


@dataclass(frozen=True)
class LinearReadout:
    w: np.ndarray
    b: float
    mean: np.ndarray
    scale: np.ndarray
    mapping: str = 'sqrt'
    budget: int = 2048
    metadata: dict = None

    def __post_init__(self):
        w = np.asarray(self.w, dtype=np.float64).copy()
        mean = np.asarray(self.mean, dtype=np.float64).copy()
        scale = np.asarray(self.scale, dtype=np.float64).copy()
        if w.ndim != 1 or not len(w) or mean.shape != w.shape or scale.shape != w.shape:
            raise ValueError('Coefficient and source-statistics shapes differ')
        if not all(np.isfinite(x).all() for x in (w, mean, scale)) or not np.isfinite(self.b) or (scale <= 0).any():
            raise ValueError('Invalid coefficients or source standardization')
        inverse(np.array([1.]), self.mapping, self.budget)
        for x in (w, mean, scale): x.setflags(write=False)
        object.__setattr__(self, 'w', w)
        object.__setattr__(self, 'mean', mean)
        object.__setattr__(self, 'scale', scale)
        object.__setattr__(self, 'b', float(self.b))
        object.__setattr__(self, 'metadata', dict(self.metadata or {}))

    def coordinates(self, x):
        return matrix(x, len(self.w)) @ self.w + self.b

    def predict(self, x):
        # Final bound also matches the native ABI and removes one-ulp sqrt^2 overshoot.
        return np.clip(inverse(self.coordinates(x), self.mapping, self.budget), 1., self.budget)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            np.savez(f, w=self.w, b=self.b, mean=self.mean, scale=self.scale,
                     mapping=self.mapping, budget=self.budget,
                     metadata_json=json.dumps(self.metadata, sort_keys=True, allow_nan=False))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z['w'], float(z['b']), z['mean'], z['scale'],
                       str(z['mapping']), int(z['budget']), json.loads(str(z['metadata_json'])))


def fit_readout(train_x, train_y, val_x, val_y, *, mapping='sqrt',
                alphas=(.0001, .001, .01, .1, 1., 10.), feature_id='last_prompt_hidden',
                export_dtype='float32', metadata=None):
    """Select alpha on validation labels only; test inputs are not accepted."""
    train_x = matrix(train_x)
    val_x = matrix(val_x, train_x.shape[1])
    if len(train_x) < 2: raise ValueError('At least two source-training examples required')
    train_y = labels(train_y, len(train_x))
    val_y = labels(val_y, len(val_x))
    alphas = tuple(float(a) for a in alphas)
    if not alphas or not all(np.isfinite(a) and a > 0 for a in alphas):
        raise ValueError('Ridge candidates must be positive finite values')
    if mapping not in ('sqrt', 'log', 'identity') or export_dtype not in ('float32', 'float64'):
        raise ValueError('Unsupported map or export precision')
    raw = np.concatenate([train_x, val_x])
    fitted, candidates, costs, mean, scale = fit_heads(
        raw, np.arange(len(train_x)), np.arange(len(train_x), len(raw)), train_y, val_y, alphas)
    selected = next(row for row in fitted if row['mapping'] == mapping)
    for row in candidates:
        row['feature'] = feature_id
        # The public interface accepts an explicitly supplied label convention.
        row['validation_mae'] = row.pop('validation_mean20_mae')
    # The paper's source export is FP32, promoted to FP64 for CPU accumulation.
    w = selected['w'].astype(export_dtype).astype(np.float64)
    b = float(np.asarray(selected['b'], dtype=export_dtype))
    info = dict(metadata or {})
    info.update(feature_id=feature_id, train_examples=len(train_x), val_examples=len(val_x),
                alpha=selected['alpha'], regularization_lambda=selected['alpha'] * len(w),
                validation_mae=selected['validation_mean20_mae'], validation_used_for_selection=True,
                test_used_for_selection=False, standardization='source train only, folded into w/b',
                exported_parameter_dtype=export_dtype, accumulation_dtype='float64', parameter_count=len(w)+1,
                source_train_features_sha256=hashlib.sha256(np.ascontiguousarray(train_x, dtype='<f8').tobytes()).hexdigest())
    return LinearReadout(w, b, mean, scale, mapping, 2048, info), candidates, costs


def make_rls_prior(head, source_train_x):
    """Create the original FP32 inverse source-feature precision and deployed prior mean.

    This is an offline preparation step: the dense matrix is O(d^2) storage.
    RLS remains anchored to the actual (possibly rounded) deployed coefficients.
    """
    if head.mapping != 'sqrt' or head.budget != 2048:
        raise ValueError('Published RLS worker requires sqrt mapping and budget 2048')
    x = matrix(source_train_x, len(head.w))
    if len(x) != head.metadata.get('train_examples'):
        raise ValueError('The full source-training feature matrix is required')
    if hashlib.sha256(np.ascontiguousarray(x, dtype='<f8').tobytes()).hexdigest() != head.metadata['source_train_features_sha256']:
        raise ValueError('RLS prior must use exactly the fitted source-training features')
    u = np.c_[(x-head.mean)/head.scale, np.ones(len(x))]
    precision = u.T @ u
    d = len(head.w)
    precision[np.arange(d), np.arange(d)] += float(head.metadata['regularization_lambda'])
    coef = np.r_[head.w*head.scale, head.b+head.mean@head.w]
    return dict(P=np.linalg.inv(precision).astype(np.float32), coef=coef,
                mean=head.mean.copy(), scale=head.scale.copy())

