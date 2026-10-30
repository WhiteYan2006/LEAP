"""Optional wrapper around the original single-block 4096-coordinate C++ head."""
import ctypes as C
from pathlib import Path
import numpy as np


class NativeReadout:
    def __init__(self, head, library, max_batch=512):
        if len(head.w) != 4096 or head.mapping != 'sqrt' or head.budget != 2048:
            raise ValueError('Original native ABI is fixed to 4096/sqrt/2048')
        if type(max_batch) is not int or not 1 <= max_batch <= 512:
            raise ValueError('Invalid native batch bound')
        self.max_batch = max_batch
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.lie30_create.argtypes = [C.POINTER(C.c_double), C.c_double, C.c_int]
        self.lib.lie30_create.restype = C.c_void_p
        self.lib.lie30_run.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_int, C.POINTER(C.c_double)]
        self.lib.lie30_run.restype = C.c_int
        self.lib.lie30_destroy.argtypes = [C.c_void_p]
        self.lib.lie30_destroy.restype = None
        self.w = np.ascontiguousarray(head.w, dtype=np.float64)
        self.handle = self.lib.lie30_create(self.w.ctypes.data_as(C.POINTER(C.c_double)), head.b, max_batch)
        if not self.handle: raise ValueError('Native head rejected its coefficients')

    def predict(self, x):
        if not self.handle: raise RuntimeError('Native head is closed')
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 4096 or not 1 <= len(x) <= self.max_batch or not np.isfinite(x).all():
            raise ValueError('Invalid native feature matrix')
        out = np.empty(len(x), dtype=np.float64)
        rc = self.lib.lie30_run(self.handle, x.ctypes.data_as(C.POINTER(C.c_float)), len(x), out.ctypes.data_as(C.POINTER(C.c_double)))
        if rc: raise RuntimeError('Native head failed: '+str(rc))
        return out

    run = predict

    def close(self):
        if self.handle:
            self.lib.lie30_destroy(self.handle)
            self.handle = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

