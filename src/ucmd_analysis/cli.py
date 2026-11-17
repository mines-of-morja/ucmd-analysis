"""`ucmd-analyze`: transport, structure and density of a spec H5MD trajectory (JSON output).

    ucmd-analyze transport TRAJ.h5 --temperature-K 293.15 --cation Li --anion PF6 [--skip-ps 0]
    ucmd-analyze rdf TRAJ.h5 --a Li:Li --b EC:O:1 [--r-max-nm 1.0] [--skip-ps 0]
    ucmd-analyze density TRAJ.h5 [--skip-ps 0]

Atom groups are MOLECULE:ELEMENT[:BONDS], e.g. EC:O:1 for the carbonyl oxygens of EC.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import constants as c
from .io import read_h5md
from .structure import coordination, rdf
from .transport import FitWindow, transport
from .uncertainty import blocks, mean_sem


def _skip(traj, skip_ps: float):
    return int(np.searchsorted(traj.times_ps, traj.times_ps[0] + skip_ps))


def _group(system, spec: str) -> np.ndarray:
    parts = spec.split(":")
    bonded = int(parts[2]) if len(parts) > 2 else None
    return system.atoms_where(parts[0], parts[1], bonded=bonded)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ucmd-analyze", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("transport")
    r = sub.add_parser("rdf")
    d = sub.add_parser("density")
    for p in (t, r, d):
        p.add_argument("trajectory")
        p.add_argument("--skip-ps", type=float, default=0.0)
    t.add_argument("--temperature-K", type=float, required=True)
    t.add_argument("--cation", default="Li")
    t.add_argument("--anion", default="PF6")
    t.add_argument("--blocks", type=int, default=5)
    t.add_argument("--fit", type=float, nargs=2, default=(0.1, 0.5))
    t.add_argument("--charges", choices=["formal", "scaled"], default="formal")
    r.add_argument("--a", required=True)
    r.add_argument("--b", required=True)
    r.add_argument("--r-max-nm", type=float, default=1.0)
    r.add_argument("--bins", type=int, default=200)
    args = ap.parse_args(argv)
    traj = read_h5md(args.trajectory)
    k = _skip(traj, args.skip_ps)
    if k:
        traj = read_h5md(args.trajectory, start=k)
    if args.command == "transport":
        out = transport(
            traj, args.temperature_K, args.cation, args.anion, n_blocks=args.blocks,
            window=FitWindow(*args.fit), charges=args.charges,
        )  # fmt: skip
    elif args.command == "rdf":
        a, b = _group(traj.system, args.a), _group(traj.system, args.b)
        rr, g = rdf(traj.positions_nm, traj.boxes_nm, a, b, r_max_nm=args.r_max_nm,
                    bins=args.bins)  # fmt: skip
        rho_b = len(b) / float(traj.volumes_nm3().mean())
        n, r_cut = coordination(rr, g, rho_b)
        out = {"a": args.a, "b": args.b, "n_a": len(a), "n_b": len(b), "frames": len(traj),
               "first_peak_nm": float(rr[np.argmax(g)]), "g_max": float(g.max()),
               "coordination_number": n, "r_cut_nm": r_cut,
               "r_nm": rr.round(5).tolist(), "g": g.round(5).tolist()}  # fmt: skip
    else:
        mass_g = traj.system.total_mass_amu() * c.AMU_IN_G
        rho = mass_g / (traj.volumes_nm3() * c.NM3_IN_CM3)
        values = [rho[s].mean() for s in blocks(len(rho), 5)]
        mean, sem = mean_sem(values)
        out = {"density_g_cm3": {"mean": mean, "sem": sem}, "frames": len(traj)}
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
