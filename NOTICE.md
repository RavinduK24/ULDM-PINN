# Third-party software notice

This repository **does not redistribute** third-party numerical codes.

## PyUltraLight

The reference solver used to validate the PINN is **PyUltraLight**:

> Mocz, Vogelsberger, Robles, Smith, Hernquist & Marinacci (2017),
> *Galaxy formation with BECDM*, MNRAS 471, 4559. Code paper: arXiv:1807.04037.
> Upstream repository: https://github.com/auckland-cosmology/PyUltraLight

PyUltraLight is developed and maintained by another research group (Auckland
Cosmology). It is **not** part of this project, and it is intentionally excluded
from version control. If you use it, please:

1. Clone it from the upstream repository into `external/` (which is git-ignored):
   ```bash
   mkdir -p external && cd external
   git clone https://github.com/auckland-cosmology/PyUltraLight.git
   ```
2. Read and comply with its licence in that repository.
3. Cite Mocz et al. (2017) in any publication that uses results generated with it.

`src/fdm/reference.py` only *loads* the upstream file at run time and applies two
import-time, non-physics compatibility fixes (numexpr ≥ 2.14 rejects `np.pi`
inside expression strings; Windows forbids `:` in file names). It never modifies
or copies the upstream source into this repository.

## Reference papers (physics background, not code)

- Mishra & Tolley (2025), *SPINN*, [arXiv:2506.02957](https://arxiv.org/abs/2506.02957)
- Hui, Ostriker, Tremaine & Witten (2017), *Ultralight scalars as cosmological dark matter*, [arXiv:1610.08297](https://arxiv.org/abs/1610.08297)
