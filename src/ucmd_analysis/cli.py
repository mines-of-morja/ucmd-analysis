"""`ucmd-analyze`: transport, structure and density of a spec H5MD trajectory (JSON output).

    ucmd-analyze transport TRAJ.h5 --temperature-K 293.15 --cation Li --anion PF6 [--skip-ps 0]
    ucmd-analyze rdf TRAJ.h5 --a Li:Li --b EC:O:1 [--r-max-nm 1.0] [--skip-ps 0]
    ucmd-analyze density TRAJ.h5 [--skip-ps 0]
    ucmd-analyze clusters TRAJ.h5 --cation Li:Li --anion PF6:P [--r-cut-nm auto]
    ucmd-analyze shell TRAJ.h5 --a Li:Li --b EC:O:1 --b EMC:O:1 --b PF6:P [--r-cut-nm auto ...]
    ucmd-analyze residence TRAJ.h5 --a Li:Li --b EC:O:1 [--r-cut-nm auto] [--tstar-ps 0]

Atom groups are MOLECULE:ELEMENT[:BONDS], e.g. EC:O:1 for the carbonyl oxygens of EC.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import constants as c
from .association import clusters_with_blocks, contacts, shell_composition, with_blocks
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


def _r_cut(traj, a: np.ndarray, b: np.ndarray, value: str) -> float:
    """A number, or "auto": the first minimum of g(r) after its first peak, on ≤ 200 frames."""
    if value != "auto":
        return float(value)
    k = max(1, len(traj) // 200)
    rr, g = rdf(traj.positions_nm[::k], traj.boxes_nm[::k], a, b, r_max_nm=0.8, bins=160)
    return coordination(rr, g, 1.0)[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ucmd-analyze", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("transport")
    r = sub.add_parser("rdf")
    d = sub.add_parser("density")
    cl = sub.add_parser("clusters")
    sh = sub.add_parser("shell")
    rs = sub.add_parser("residence")
    for p in (t, r, d, cl, sh, rs):
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
    for p in (cl, rs):
        p.add_argument("--r-cut-nm", default="auto")
    for p in (cl, sh, rs):
        p.add_argument("--blocks", type=int, default=5)
    cl.add_argument("--cation", default="Li:Li")
    cl.add_argument("--anion", default="PF6:P")
    sh.add_argument("--a", default="Li:Li")
    sh.add_argument("--b", action="append", required=True)
    sh.add_argument("--r-cut-nm", action="append", help="one per --b, or auto (default)")
    rs.add_argument("--a", default="Li:Li")
    rs.add_argument("--b", required=True)
    rs.add_argument("--tstar-ps", type=float, default=0.0)
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
    elif args.command == "clusters":
        a, b = _group(traj.system, args.cation), _group(traj.system, args.anion)
        r_cut = _r_cut(traj, a, b, args.r_cut_nm)
        out = clusters_with_blocks(contacts(traj.positions_nm, traj.boxes_nm, a, b, r_cut),
                                   args.blocks)  # fmt: skip
        out.update({"cation": args.cation, "anion": args.anion, "r_cut_nm": r_cut,
                    "frames": len(traj)})  # fmt: skip
    elif args.command == "shell":
        a = _group(traj.system, args.a)
        cuts = args.r_cut_nm or ["auto"] * len(args.b)
        if len(cuts) != len(args.b):
            ap.error("give --r-cut-nm once per --b, or not at all")
        per, used = {}, {}
        for spec, cut in zip(args.b, cuts, strict=True):
            b = _group(traj.system, spec)
            used[spec] = _r_cut(traj, a, b, cut)
            per[spec] = contacts(traj.positions_nm, traj.boxes_nm, a, b, used[spec])
        out = shell_composition(per)
        out.update({"a": args.a, "r_cut_nm": used, "frames": len(traj)})
    elif args.command == "residence":
        a, b = _group(traj.system, args.a), _group(traj.system, args.b)
        r_cut = _r_cut(traj, a, b, args.r_cut_nm)
        dt = float(np.diff(traj.times_ps).mean())
        out = with_blocks(contacts(traj.positions_nm, traj.boxes_nm, a, b, r_cut), dt,
                          args.blocks, tstar_ps=args.tstar_ps)  # fmt: skip
        out.update({"a": args.a, "b": args.b, "r_cut_nm": r_cut, "frames": len(traj),
                    "frame_interval_ps": dt})  # fmt: skip
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
