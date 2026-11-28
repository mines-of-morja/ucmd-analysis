"""Readers (PLAN.md A1.1): H5MD trajectories of the spec profile and their embedded system.json.

A trajectory carries the system it was produced from (spec/trajectory-h5md.md), so analysis needs
no other file. Positions are unwrapped and molecules whole.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Molecules:
    """The molecules of one species (molecule type) in canonical order."""

    name: str
    atoms: np.ndarray  # (n_molecules, atoms_per_molecule) global atom indices
    masses: np.ndarray  # (atoms_per_molecule,) amu
    formal_charge: int  # e, the type's formal charge (unscaled)
    charge: float  # e, the sum of the (possibly scaled) partial charges

    @property
    def count(self) -> int:
        return len(self.atoms)


@dataclass(frozen=True)
class System:
    doc: dict
    masses: np.ndarray
    charges: np.ndarray
    elements: list[str]
    species: dict[str, Molecules]
    bonds: list[tuple[int, int]]  # global atom indices

    @property
    def n_atoms(self) -> int:
        return len(self.masses)

    def total_mass_amu(self) -> float:
        return float(self.masses.sum())

    def atoms_where(self, molecule: str, element: str, *, bonded: int | None = None) -> np.ndarray:
        """Atoms of `element` in molecules of type `molecule`, optionally with exactly `bonded`
        bonds (e.g. carbonyl O: O with one bond)."""
        degree = np.zeros(self.n_atoms, dtype=int)
        for i, j in self.bonds:
            degree[i] += 1
            degree[j] += 1
        mol = self.species[molecule]
        local = [k for k, e in enumerate(self._type_elements[molecule]) if e == element]
        picked = mol.atoms[:, local].ravel()
        if bonded is not None:
            picked = picked[degree[picked] == bonded]
        return np.sort(picked)

    @property
    def _type_elements(self) -> dict[str, list[str]]:
        return {
            name: [a.get("element") or "X" for a in t["atoms"]]
            for name, t in self.doc["molecule_types"].items()
        }


def system_from_json(doc: dict) -> System:
    masses, charges, elements, bonds = [], [], [], []
    species: dict[str, list] = {}
    offset = 0
    for component in doc["components"]:
        name = component["molecule_type"]
        mtype = doc["molecule_types"][name]
        n = len(mtype["atoms"])
        for _ in range(component["count"]):
            species.setdefault(name, []).append(list(range(offset, offset + n)))
            for atom in mtype["atoms"]:
                masses.append(atom["mass_amu"])
                charges.append(atom.get("charge_e") or 0.0)
                elements.append(atom.get("element") or "X")
            bonds += [(offset + b["atoms"][0], offset + b["atoms"][1]) for b in mtype["bonds"]]
            offset += n
    out = {}
    for name, rows in species.items():
        t = doc["molecule_types"][name]
        out[name] = Molecules(
            name,
            np.array(rows, dtype=np.int64),
            np.array([a["mass_amu"] for a in t["atoms"]]),
            int(t.get("formal_charge_e", 0) or 0),
            float(sum(a.get("charge_e") or 0.0 for a in t["atoms"])),
        )
    return System(doc, np.array(masses), np.array(charges), elements, out, bonds)


@dataclass(frozen=True)
class Trajectory:
    system: System
    steps: np.ndarray
    times_ps: np.ndarray
    positions_nm: np.ndarray  # (frames, atoms, 3), unwrapped
    boxes_nm: np.ndarray  # (frames, 3, 3), rows a, b, c
    provenance: dict
    velocities_nm_per_ps: np.ndarray | None = None  # (frames, atoms, 3), only if asked for

    def __len__(self) -> int:
        return len(self.steps)

    def volumes_nm3(self) -> np.ndarray:
        return np.abs(np.linalg.det(self.boxes_nm))


def read_h5md(path: Path | str, *, start: int = 0, stop: int | None = None,
              stride: int = 1, velocities: bool = False) -> Trajectory:  # fmt: skip
    """Frames [start:stop:stride] of a spec H5MD trajectory. Velocities are optional in the spec
    profile and are read only if asked for; a file without them is then an error."""
    import h5py

    with h5py.File(path, "r") as fh:
        doc = json.loads(_text(fh["parameters/ucmd/system_json"][()]))
        pos = fh["particles/all/position"]
        sel = slice(start, stop, stride)
        steps = np.asarray(pos["step"][sel])
        positions = np.asarray(pos["value"][sel], dtype=np.float64)
        vel = _velocities(fh, path, sel, steps) if velocities else None
        box_g = fh["particles/all/box/edges"]
        box_steps = np.asarray(box_g["step"][()])
        boxes = np.asarray(box_g["value"][()], dtype=np.float64)
        idx = np.searchsorted(box_steps, steps)
        params = fh["parameters/ucmd"]
        prov = {k: _text(v) for k, v in params.attrs.items()}
        if "backend" in params:
            prov["backend"] = {k: _text(v) for k, v in params["backend"].attrs.items()}
        return Trajectory(
            system_from_json(doc),
            steps,
            np.asarray(pos["time"][sel], dtype=np.float64),
            positions,
            boxes[np.clip(idx, 0, len(boxes) - 1)],
            prov,
            vel,
        )


def _velocities(fh, path, sel: slice, steps: np.ndarray) -> np.ndarray:
    if "particles/all/velocity" not in fh:
        raise ValueError(
            f"{path}: no velocities; run with [output.trajectory] velocities = true and a stride "
            "that resolves the velocity correlation"
        )
    group = fh["particles/all/velocity"]
    if not np.array_equal(np.asarray(group["step"][sel]), steps):
        raise ValueError(f"{path}: velocities and positions are not stored at the same steps")
    return np.asarray(group["value"][sel], dtype=np.float64)


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _text(value.item())
    return str(value) if not isinstance(value, str) else value
