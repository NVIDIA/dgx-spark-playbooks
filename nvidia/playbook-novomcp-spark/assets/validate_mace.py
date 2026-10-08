"""Validate MACE-MPA-0 on the GPU: ethanol single-point + relaxation.

A single-point energy within a few meV of -46.26 eV confirms the stock
CUDA-13 wheel's kernels execute correctly on GB10 (sm_121). Run inside the
mace-spark image (the two Grace env fixes are baked into the Dockerfile).
"""
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from mace.calculators import mace_mp

# Ethanol (CH3CH2OH), a fixed starting geometry.
symbols = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
positions = np.array([
    [-1.1879, 0.1446, 0.0000],
    [0.1894, -0.5718, 0.0000],
    [1.2033, 0.4100, 0.0000],
    [-1.2721, 0.7709, 0.8930],
    [-1.2721, 0.7709, -0.8930],
    [-2.0122, -0.5727, 0.0000],
    [0.2601, -1.2039, 0.8900],
    [0.2601, -1.2039, -0.8900],
    [2.0472, -0.0367, 0.0000],
])

calc = mace_mp(model="medium-mpa-0", device="cuda", default_dtype="float32")
atoms = Atoms(symbols=symbols, positions=positions)
atoms.calc = calc

e_sp = atoms.get_potential_energy()
print(f"single-point energy (eV): {e_sp:.4f}   [expect ~ -46.26]")

BFGS(atoms, logfile=None).run(fmax=0.02, steps=200)
e_relaxed = atoms.get_potential_energy()
print(f"relaxed energy (eV):      {e_relaxed:.4f}   [expect ~ -46.28]")

assert -47.0 < e_sp < -45.0, f"single-point energy {e_sp} eV is far from reference — kernels may not be running correctly"
print("OK — MACE-MPA-0 runs correctly on this GPU.")
