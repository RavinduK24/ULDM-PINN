"""
Re-run the notebook diagnostics from the code review.

Usage:
    python scripts/run_diagnostics.py
    python scripts/run_diagnostics.py --profile <path/to/initial_f.npy>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fdm import diagnostics as diag  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=diag.DEFAULT_PROFILE,
                    help="path to the soliton profile initial_f.npy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = {}
    print("=" * 80)
    print("1. Poisson compatibility floor  (periodic V'' = rho  requires <rho> = 0)")
    print("=" * 80)
    rows = []
    for c in diag.notebook_cases():
        r = diag.poisson_compatibility(c["rho"])
        r["name"] = c["name"]
        rows.append(r)
        print(f"{c['name']:46s} <rho>={r['rho_mean']:+.3e}  "
              f"floor||r||={r['floor']:.3e}  rel={r['floor_rel']:.3f}")
    out["poisson_compatibility"] = rows

    print()
    print("=" * 80)
    print("2. 1-D sheet Poisson vs 3-D Newtonian Poisson (same Gaussian lump)")
    print("=" * 80)
    g = diag.one_d_vs_three_d()
    print(f"1-D periodic depth = {g['V1_depth']:.4e}   "
          f"3-D Newtonian depth = {g['Phi3_depth']:.4e}")
    print("force (should fall as 1/r^2 in 3-D, stay flat in 1-D):")
    for f in g["force"]:
        print(f"  r={f['r']:4.1f}   |V'|_1D = {f['f1']:.5e}   |Phi'|_3D = {f['f3']:.5e}")
    out["poisson_geometry"] = g

    print()
    print("=" * 80)
    print("3. Box dependence of the 1-D potential (physics must not depend on L)")
    print("=" * 80)
    bd = diag.box_dependence_1d()
    for b in bd:
        print(f"  L = {b['L']:6.1f}   depth = {b['depth']:10.5f}   depth*L = {b['depth_times_L']:10.5f}")
    out["box_dependence_1d"] = bd

    print()
    print("=" * 80)
    print("4. 'trapz over a random batch' as a surrogate for int rho dx")
    print("=" * 80)
    m = diag.madelung_trapz()
    print(f"true integral               = {m['true_integral']:.6f}")
    print(f"notebook estimator (N=1e4)  = {m['interior_estimate']:.6f}  "
          f"(bias {m['bias']:.3f} x)")
    print(f"boundary estimator (N=1e3)  = {m['boundary_estimate']:.6f}  "
          f"(mismatch {m['interior_boundary_mismatch']:.2f} x)")
    out["madelung_trapz"] = m

    print()
    print("=" * 80)
    print("5. Ground-state soliton profile")
    print("=" * 80)
    if os.path.exists(args.profile):
        s = diag.soliton_profile_metrics(args.profile)
        print(f"M_code = {s['M_code']:.6f}   r_half = {s['r_half']:.6f}   "
              f"M*r_half = {s['M_times_r']:.6f}")
        out["soliton"] = s
    else:
        print(f"[skip] profile not found: {args.profile}")
        print("       clone PyUltraLight into external/ (see NOTICE.md) to run this check.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[io] wrote {args.out}")


if __name__ == "__main__":
    main()
