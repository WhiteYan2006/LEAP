"""LEAP's NumPy-first readout and completion-feedback implementation."""
from .readout import LinearReadout, fit_readout, make_rls_prior
from ._ridge_reference import metric
from .rls_worker import RLSFeedbackCore, RLSWorker

__version__ = '0.1.0.dev0'
__all__ = ['LinearReadout', 'fit_readout', 'make_rls_prior', 'metric',
           'RLSFeedbackCore', 'RLSWorker']

