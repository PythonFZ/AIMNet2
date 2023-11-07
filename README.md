[![ZnTrack](https://img.shields.io/badge/Powered%20by-ZnTrack-%23007CB0)](https://zntrack.readthedocs.io/en/latest/)

# AIMNet2: a general-purpose neural netrork potential for organic and element-organic molecules.

The repository contains AIMNet2 models, example Python code and supplementary data for the manuscript

**AIMNet2: A Neural Network Potential to Meet your Neutral, Charged, Organic, and Elemental-Organic Needs**
*Dylan Anstine ,Roman Zubatyuk ,Olexandr Isayev*
[10.26434/chemrxiv-2023-296ch](https://doi.org/10.26434/chemrxiv-2023-296ch)

## Deployment with ZnTrack
To deploy these models using ZnTrack, you need to install this repository `pip install git+https://github.com/PythonFZ/AIMNet2.git`.

Afterwards, you can load the models using
```python
import zntrack
from ase.build import molecule

model = zntrack.from_rev("wb97m_d3_ens", remote="https://github.com/PythonFZ/AIMNet2.git")
atoms = molecule('H2O')

atoms.calc = model.get_calculator()
print(atoms.get_potential_energy())
```
If you want to use an older version of the model, you can pass the `rev=<sha>` argument, to select a different commit.

Alternatively, to download the models permanently you can
```bash
git clone https://github.com/PythonFZ/AIMNet2.git
cd AIMnet2
dvc pull
```
and use `model = zntrack.from_rev("wb97m_d3_ens", remote=".")`.

  
## Models

The models are applicable for systems containing the following set of chemical elements:
{H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I}, both neutral and charged. The models aim to reproduce RKS B97-3c and wB97M-D3 energies.
  
The models are in the form of JIT-compiled PyTorch-2.0 files and could be loaded in Python or C++ code.

Note, that at present models have O(N^2) compute and memory complexity w.r.t. number of atoms. They could be applied to systems up to a few 100's of atoms. Linear scaling models, with the same parametrization, will be released soon.

In Python, the models could be loaded with the `torch.jit.load` function. As an input, they accept single argument of type `Dict[str, Tensor]` with the following data:
```
coords: shape (m, n, 3) - atomic coordinates in Angstrom 
numbers: shape (m, n) - atomic numbers
charge: shape (m, ) - total charge
```
Output is a dictionary with the following keys:
```
energy: shape (m, ) - energy in eV
charges: shape (m, n) - partial atomic charges
```
## Calculators

We provide example code for AIMNet2 calculators for [ASE](https://wiki.fysik.dtu.dk/ase) and [pysisyphus](https://pysisyphus.readthedocs.io/) Python libraries. The code shows an example use of the AIMNet2 models. 

We also provide example geometry optimization scripts with ASE and Pysisyphus, and `pysis_mod` script which is a drop-in replacement for Pysisyphus `pysis` command-line utility, with AIMNet2 calculator enabled.

## Docker image

We provide an example Dockerfile to build a CPU docker image.

The commands for building docker image: 
```bash
cd docker 
docker build --platform linux/amd64 --pull --rm -f "Dockerfile_cpu" -t aimnet-box "."
```

You might skip the `--platform` flag if you are building on Linux.

The image exposes `aimnet2_ase_opt.py` script as entrypoint.

Example command to run geometry optimization with docker image:

```bash
docker run -it --rm -v $(pwd):/app/ aimnet-box models/aimnet2_wb97m-d3_ens.jpt input.sdf output.sdf --charge 0 --traj traj.xyz
```

### Geometry Optimization using IPSuite
You can use these models together with [IPSuite](https://github.com/zincware/IPSuite) (`pip install ipsuite`).
In this example, we show a geometry optimization followed by an MD simulation of a condensed phase system.

```python
import zntrack
import numpy as np
import ipsuite as ips

model = zntrack.from_rev("wb97m_d3_0", remote="https://github.com/PythonFZ/AIMNet2.git", rev="main")

thermostat = ips.calculators.LangevinThermostat(
    temperature=298.15, friction=0.01, time_step=0.5
)

with zntrack.Project(automatic_node_names=True) as proj:
    emc = ips.configuration_generation.SmilesToAtoms(smiles="CCOC(=O)OC")
    ec = ips.configuration_generation.SmilesToAtoms(smiles="C1COC(=O)O1")

    structure = ips.configuration_generation.Packmol(
        data=[emc.atoms, ec.atoms],
        count=[40, 40],
        density=1200,
    )

    geo_opt = ips.calculators.ASEGeoOpt(data=structure.atoms, model=model)

    md = ips.calculators.ASEMD(
            data=geo_opt.atoms,
            data_id=-1,
            model=model,
            thermostat=thermostat,
            steps=10_000,
            sampling_rate=10,
        )

proj.run()

md.load()
print(md.atoms)
```

### Feedback

We would appreciate it if you could share feedback about model accuracy and performance. This would be important not only for us, to guide further developments of the model, but for the broad community as well. 
Please share your thoughts and experience, either positive or negative, by opening an issue in this repo.
