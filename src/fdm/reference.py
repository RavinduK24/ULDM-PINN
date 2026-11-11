"""
Loader for the external PyUltraLight reference solver.

PyUltraLight (Mocz et al. 2017, arXiv:1807.04037) is **not** part of this
repository — see NOTICE.md.  It must be cloned from upstream into `external/`:

    mkdir -p external && cd external
    git clone https://github.com/auckland-cosmology/PyUltraLight.git

This module loads that upstream file at run time and applies two import-time,
non-physics compatibility transformations, without modifying the source:

  1. numexpr >= 2.14 rejects the dotted name `np.pi` inside expression strings,
     so it is rewritten to a plain module-level name `PI` (identical value).
  2. On Windows the output-directory timestamp uses `:`, which is not a legal
     character in a file name, so it is replaced by `-`.
"""
from __future__ import annotations

import os
import sys
import types


def _shim_ipython():
    if "IPython" in sys.modules:
        return
    m = types.ModuleType("IPython")
    core = types.ModuleType("IPython.core")
    disp = types.ModuleType("IPython.core.display")
    disp.clear_output = lambda *a, **k: None
    core.display = disp
    m.core = core
    sys.modules["IPython"] = m
    sys.modules["IPython.core"] = core
    sys.modules["IPython.core.display"] = disp


class _NoJit:
    """numba.jit passthrough: PyUltraLight's initsoliton() uses np.ndindex and
    is not nopython-safe, so it is run as plain Python."""

    @staticmethod
    def jit(f):
        return f


def load(pyultralight_dir: str):
    """Load PyUltraLight.py from *pyultralight_dir* and return the module."""
    src_path = os.path.join(pyultralight_dir, "PyUltraLight.py")
    with open(src_path, encoding="utf-8") as fh:
        src = fh.read()

    # numexpr >= 2.14 rejects dotted names inside expression strings.
    src = src.replace("np.pi", "PI")
    src = src.replace("import numpy as np\n",
                      "import numpy as np\nPI = 3.141592653589793\n", 1)

    # Windows: ':' is not a legal character in a file name.
    src = src.replace("':', talt[1], ':'", "'-', talt[1], '-'")

    _shim_ipython()
    module = types.ModuleType("PyUltraLight_loaded")
    module.__dict__["__file__"] = src_path
    module.__dict__["numba"] = _NoJit
    exec(compile(src, src_path, "exec"), module.__dict__)
    module.numba = _NoJit
    return module
