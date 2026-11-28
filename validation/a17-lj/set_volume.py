"""Scale a cubic package to the mean volume of an NPT run (start of the NVT comparison).

python set_volume.py IN.ucsys NPT_THERMO.tsv OUT.ucsys
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def thermo_column(path: Path, name: str) -> np.ndarray:
    lines = path.read_text().splitlines()
    index = lines[0].lstrip("# ").split("\t").index(name)
    return np.array([float(line.split("\t")[index]) for line in lines[1:]])


def main() -> None:
    source, thermo, out = (Path(a) for a in sys.argv[1:4])
    doc = json.loads((source / "system.json").read_text())
    cell = np.array(doc["cell_nm"])
    if not np.allclose(cell, np.eye(3) * cell[0, 0]):
        raise SystemExit("the cell is not cubic")
    volume = float(thermo_column(thermo, "volume_nm3").mean())
    factor = (volume / np.linalg.det(cell)) ** (1 / 3)
    doc["cell_nm"] = (cell * factor).tolist()
    doc["description"] = f"{doc.get('description', '')} Scaled to the mean NPT volume."
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "positions.npy", np.load(source / "positions.npy") * factor)
    (out / "system.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"mean volume {volume:.4f} nm3, scale factor {factor:.6f}")


if __name__ == "__main__":
    main()
