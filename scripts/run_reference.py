"""
Generate reference data with the external PyUltraLight solver.

Two runs (all in PyUltraLight's code units, hbar = m = 1, 4 pi G = 1):

  A  one isolated ground-state soliton  -> must be stationary
  B  two solitons approaching            -> merger + interference

For each run the script writes, under --out:
  masslist.npy, egylist.npy, rho_#N.npy, line_#N.npy
and prints the mass/energy drift and the profile change.

Note: PyUltraLight's `central_mass` argument must be 0 for a pure self-gravity
run — with a non-zero value it ADDS a Keplerian point mass of that mass on top
of the density grid's own gravity (which collapses the soliton).

Usage:
    python scripts/run_reference.py --pyul external/PyUltraLight --out data/reference
"""
import argparse
import glob
import io
import json
import os
import sys


def _run(pul, out_root, name, resol, length, duration, central_mass, solitons,
         save_number):
    pyul_dir = os.path.dirname(pul.__file__)
    save_path = os.path.relpath(os.path.join(out_root, name), pyul_dir)
    os.makedirs(os.path.join(out_root, name), exist_ok=True)
    save_options = [True, False, False, True, True]  # rho, psi, plane, energies, line
    print(f"\n=== {name}: resol={resol} L={length} T={duration} M={central_mass} ===")

    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = buf
    try:
        pul.evolve(central_mass, 4, length, '', resol, duration, '', 1,
                   save_number, save_options, save_path, False, True, False,
                   '', '', '', solitons, 0.0)
    finally:
        sys.stdout = real

    stamp = open(os.path.join(pyul_dir, save_path, "timestamp.txt")).read().strip()
    return os.path.join(pyul_dir, save_path, stamp)


def _analyse(loc, resol, length):
    rho_files = sorted(glob.glob(os.path.join(loc, "rho_#*.npy")),
                       key=lambda p: int(p.split("#")[-1].split(".")[0]))
    line_files = sorted(glob.glob(os.path.join(loc, "line_#*.npy")),
                        key=lambda p: int(p.split("#")[-1].split(".")[0]))
    import numpy as np
    mass = np.load(os.path.join(loc, "masslist.npy"))
    egy = np.load(os.path.join(loc, "egylist.npy"))
    d = dict(mass=float(mass[-1]), mass_drift=float(abs(mass[-1] - mass[0]) / mass[0]),
             energy_drift=float(abs(egy[-1] - egy[0]) / abs(egy[0])))
    if line_files:
        lines = np.array([np.load(f) for f in line_files])
        r = np.abs((np.arange(resol) + 0.5) * (length / resol) - length / 2)
        order = np.argsort(r)
        p0, pT = lines[0][order], lines[-1][order]
        d["profile_rel_change"] = float(np.linalg.norm(pT - p0) / np.linalg.norm(p0))
        d["rho_c"] = float(p0.max())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyul", required=True, help="path to the cloned PyUltraLight dir")
    ap.add_argument("--out", default="data/reference")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from fdm.reference import load

    pul = load(args.pyul)
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    cases = [
        dict(name="A_stationary_soliton", resol=128, length=16.0, duration=2.5,
             central_mass=0.0, save_number=6,
             solitons=[[3.883, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0]]),
        dict(name="B_two_soliton_merger", resol=96, length=20.0, duration=6.0,
             central_mass=0.0, save_number=12,
             solitons=[[3.883, [-3.0, 0.0, 0.0], [0.15, 0.0, 0.0], 0.0],
                       [3.883, [3.0, 0.0, 0.0], [-0.15, 0.0, 0.0], 0.0]]),
    ]
    results = {}
    for c in cases:
        name = c.pop("name")
        loc = _run(pul, out_root, name, **c)
        d = _analyse(loc, c["resol"], c["length"])
        results[name] = dict(**{k: v for k, v in c.items()}, dir=loc, **d)
        print(f"    mass drift = {d['mass_drift']:.3e}  "
              f"energy drift = {d['energy_drift']:.3e}")
        if "profile_rel_change" in d:
            print(f"    profile change = {d['profile_rel_change']:.3e}  "
                  f"rho_c = {d['rho_c']:.4f}")

    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[io] wrote {os.path.join(out_root, 'summary.json')}")


if __name__ == "__main__":
    main()
