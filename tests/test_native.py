from pathlib import Path
import os, shutil, subprocess, sys, tempfile, unittest
import numpy as np
from leap import LinearReadout
from leap.native import NativeReadout

class NativeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'),'Existing g++ compiler required for original native ABI test')
    def test_original_cpp_parity_and_bounds(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as t:
            output=Path(t)/('readout.dll' if os.name=='nt' else 'readout.so')
            subprocess.run([sys.executable,str(root/'scripts/build_native.py'),'--output',str(output)],check=True,capture_output=True)
            # A child owns the DLL so Windows releases its file before temp cleanup.
            code='''import sys,numpy as np
from leap import LinearReadout
from leap.native import NativeReadout
g=np.random.default_rng(4);w=g.normal(scale=.01,size=4096);x=g.normal(size=(5,4096)).astype(np.float32)
h=LinearReadout(w,10.,np.zeros(4096),np.ones(4096));native=NativeReadout(h,sys.argv[1],8)
try:
 np.testing.assert_allclose(native.predict(x),h.predict(x),rtol=1e-12,atol=1e-10)
 try:native.predict(np.zeros((9,4096)))
 except ValueError:pass
 else:raise AssertionError('Oversized batch accepted')
finally:native.close()
try:native.predict(x)
except RuntimeError:pass
else:raise AssertionError('Closed head accepted a call')
'''
            subprocess.run([sys.executable,'-B','-c',code,str(output)],check=True,capture_output=True)

if __name__=='__main__':unittest.main()

