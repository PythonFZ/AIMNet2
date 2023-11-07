import zntrack

import torch
import ase.calculators.calculator
from ase.calculators.singlepoint import SinglePointCalculator
import tqdm


class AIMNet2Calculator(ase.calculators.calculator.Calculator):
    """ASE calculator for AIMNet2 model
    Arguments:
        model (:class:`torch.nn.Module`): AIMNet2 model
        charge (int or float): molecular charge.  Default: 0
    """

    implemented_properties = ["energy", "forces", "free_energy", "charges"]

    def __init__(self, model, charge=0, device=None):
        super().__init__()
        self.model = model
        self.charge = charge
        self.device = device if device is not None else next(model.parameters()).device
        cutoff = max(
            v.item() for k, v in model.state_dict().items() if k.endswith("aev.rc_s")
        )
        self.cutoff = float(cutoff)
        self._t_numbers = None
        self._t_charge = None

    def do_reset(self):
        self._t_numbers = None
        self._t_charge = None
        self.charge = 0.0

    def set_charge(self, charge):
        self.charge = float(charge)

    def _make_input(self):
        coord = (
            torch.as_tensor(self.atoms.positions)
            .to(torch.float)
            .to(self.device)
            .unsqueeze(0)
        )
        if self._t_numbers is None:
            self._t_numbers = (
                torch.as_tensor(self.atoms.numbers)
                .to(torch.long)
                .to(self.device)
                .unsqueeze(0)
            )
            self._t_charge = torch.tensor(
                [self.charge], dtype=torch.float, device=self.device
            )
        d = dict(coord=coord, numbers=self._t_numbers, charge=self._t_charge)
        return d

    def _eval_model(self, d, forces=True):
        prev = torch.is_grad_enabled()
        torch._C._set_grad_enabled(forces)
        if forces:
            d["coord"].requires_grad_(True)
        _out = self.model(d)
        ret = dict(
            energy=_out["energy"].item(),
            charges=_out["charges"].detach()[0].cpu().numpy(),
        )
        if forces:
            if "forces" in _out:
                f = _out["forces"][0]
            else:
                f = -torch.autograd.grad(_out["energy"], d["coord"])[0][0]
            ret["forces"] = f.detach().cpu().numpy()
        torch._C._set_grad_enabled(prev)
        return ret

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=ase.calculators.calculator.all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        _in = self._make_input()
        do_forces = "forces" in properties
        _out = self._eval_model(_in, do_forces)

        self.results["energy"] = _out["energy"]
        self.results["charges"] = _out["charges"]
        if do_forces:
            self.results["forces"] = _out["forces"]


class AIMNet2(zntrack.Node):
    model_path: str = zntrack.deps_path()
    energy_conversion: float = 1.0
    force_conversion: float = 1.0
    AIMNet2_DEVICE: str = zntrack.meta.Environment(None)

    def run(self) -> None:
        pass

    def get_calculator(self, charge=0, **kwargs) -> AIMNet2Calculator:
        with self.state.fs.open(self.model_path, "rb") as f:
            model = torch.load(f)
            model = model.to(self.AIMNet2_DEVICE)
            return AIMNet2Calculator(model, charge=charge, device=self.AIMNet2_DEVICE)

    def predict(self, atoms_list: list[ase.Atoms]) -> list[ase.Atoms]:
        calc = self.get_calculator()
        for atoms in tqdm.tqdm(atoms_list):
            atoms.set_calculator(calc)
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            energy *= self.energy_conversion
            forces *= self.force_conversion
            atoms.set_calculator(
                SinglePointCalculator(atoms, energy=energy, forces=forces)
            )
        return atoms_list
