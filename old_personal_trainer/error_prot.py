import torch
import torchani
from pathlib import Path
from torchani.transforms import AtomicNumbersToIndices, SubtractSAE
from typing import Sequence
from copy import deepcopy
import math
import torch.utils.tensorboard
import os
import shutil
from .mtl_loss import MTLLoss
import tqdm
import configparser


class personal_trainer:
    """
    Kate's Personal training class
    Keep track of species order in elements list, which is fed into setting up network architecture.
    """

    def __init__(self, config_file, device=torch.device("cpu")):
        # Create a configuration object
        config = configparser.ConfigParser(
            allow_no_value=True, inline_comment_prefixes="#"
        )
        config.read(config_file)

        # Create dictionaries for Global and Training variables
        gv = config["Global"]
        tv = config["Trainer"]

        # Assign variables
        self.device = device
        self.netlike1x = gv.getboolean("netlike1x")
        self.netlike2x = gv.getboolean("netlike2x")
        self.functional = gv.get("functional")
        self.basis_set = gv.get("basis_set")
        self.forces = tv.getboolean("forces")
        self.charges = tv.getboolean("charges")
        self.dipole = tv.getboolean("dipole")
        self.gsae_dat = eval(gv.get("gsae_dat"))
        self.batch_size = tv.getint("batch_size")
        self.ds_path = tv.get("ds_path")
        self.h5_path = tv.get("h5_path")
        self.charge_type = tv.get("charge_type")
        self.include_properties = eval(tv.get("include_properties"))
        self.logdir = tv.get("logdir")
        self.projectlabel = tv.get("projectlabel")
        self.now = eval(tv.get("now"))
        self.data_split = eval(tv.get("data_split"))
        self.activation = eval(gv.get("activation"))
        self.bias = gv.getboolean("bias")
        self.classifier_out = gv.getint("classifier_out")
        self.personal = gv.getboolean("personal")
        self.weight_decay = eval(tv.get("weight_decay"))
        self.factor = tv.getfloat("lr_factor")
        self.patience = tv.getint("lr_patience")
        self.threshold = tv.getfloat("lr_threshold")
        self.max_epochs = tv.getint("max_epochs")
        self.earlylr = tv.getfloat("early_stopping_learning_rate")
        self.restarting = tv.getboolean("restarting")
        self.num_tasks = tv.getint("num_tasks")
        self.train_file = tv.get("train_file")
        self.loss_beta = tv.getfloat("loss_beta")
        self.mtl_loss = tv.getboolean("mtl_loss")
        if self.netlike1x:
            self.constants = "/data/khuddzu/torchani_sandbox/torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params"
            self.elements = ["H", "C", "N", "O"]
        elif self.netlike2x:
            self.constants = "/data/khuddzu/torchani_sandbox/torchani/resources/ani-2x_8x/rHCNOSFCl-5.1R_16-3.5A_a8-4.params"
            self.elements = ["H", "C", "N", "O", "S", "F", "Cl"]
        else:
            self.constants = eval(gv.get("constants"))
            self.elements = eval(gv.get("elements"))

    def AEV_Computer(self):
        consts = torchani.neurochem.Constants(self.constants)
        aev_computer = torchani.AEVComputer(**consts)
        return aev_computer

    def Energy_Shifter(self):
        if self.gsae_dat:
            _, energy_shifter = torchani.neurochem.load_sae(
                self.gsae_dat, return_dict=True
            )
        else:
            energy_shifter = torchani.utils.sorted_gsaes(
                self.elements, self.functional, self.basis_set
            )
        assert len(energy_shifter) == len(
            self.elements
        ), "There must be a mistake with what atoms you are trying to use. The length of the EnergyShifter does not match the Elements"
        return energy_shifter

    def datasets_loading(self, energy_shifter):
        # ds_path can either be a path or None
        # if it is a path, it can either exist or not
        # if it is None -> In memory
        # if it is an existing path -> use it
        # if it is a nonoe existing path -> create it, and then use it
        in_memory = self.ds_path is None
        transform = torchani.transforms.Compose(
            [
                AtomicNumbersToIndices(self.elements),
                SubtractSAE(self.elements, energy_shifter),
            ]
        )
        if in_memory:
            learning_sets = torchani.datasets.create_batched_dataset(
                self.h5_path,
                include_properties=self.include_properties,
                batch_size=self.batch_size,
                inplace_transform=transform,
                shuffle_seed=123456789,
                splits=self.data_split,
                direct_cache=True,
            )
            training = torch.utils.data.DataLoader(
                learning_sets["training"],
                shuffle=True,
                num_workers=1,
                prefetch_factor=2,
                pin_memory=True,
                batch_size=None,
            )
            validation = torch.utils.data.DataLoader(
                learning_sets["validation"],
                shuffle=False,
                num_workers=1,
                prefetch_factor=2,
                pin_memory=True,
                batch_size=None,
            )
        else:
            if not Path(self.ds_path).resolve().is_dir():
                try:
                    h5 = torchani.datasets.ANIDataset.from_dir(self.h5_path)
                except BaseException:
                    h5 = torchani.datasets.ANIDataset(self.h5_path)
                torchani.datasets.create_batched_dataset(
                    h5,
                    dest_path=self.ds_path,
                    batch_size=self.batch_size,
                    include_properties=self.include_properties,
                    splits=self.data_split,
                )
            # This below loads the data if dspath exists
            training = torchani.datasets.ANIBatchedDataset(
                self.ds_path, transform=transform, split="training"
            )
            validation = torchani.datasets.ANIBatchedDataset(
                self.ds_path, transform=transform, split="validation"
            )
            training = torch.utils.data.DataLoader(
                training,
                shuffle=True,
                num_workers=1,
                prefetch_factor=2,
                pin_memory=True,
                batch_size=None,
            )
            validation = torch.utils.data.DataLoader(
                validation,
                shuffle=False,
                num_workers=1,
                prefetch_factor=2,
                pin_memory=True,
                batch_size=None,
            )
        return training, validation

    def standard(self, dims: Sequence[int]):
        r"""Makes a standard ANI style atomic network"""
        if self.activation is None:
            torch.nn.GELU()
        else:
            self.activation

        dims = list(deepcopy(dims))
        layers = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.extend(
                [torch.nn.Linear(dim_in, dim_out, bias=self.bias), self.activation]
            )
        # final layer is a linear classifier that is always appended
        layers.append(torch.nn.Linear(dims[-1], self.classifier_out, bias=self.bias))
        assert len(layers) == (len(dims) - 1) * 2 + 1
        return torch.nn.Sequential(*layers)

    def like_1x(self, atom: str = "H", aev_dim=384, **kwargs):
        r"""Makes a sequential atomic network like the one used in the ANI-1x model"""
        dims_for_atoms = {
            "H": (aev_dim, 160, 128, 96),
            "C": (aev_dim, 144, 112, 96),
            "N": (aev_dim, 128, 112, 96),
            "O": (aev_dim, 128, 112, 96),
        }
        return self.standard(dims_for_atoms[atom], **kwargs)

    def like_2x(self, atom: str = "H", aev_dim=1008, **kwargs):
        r"""Makes a sequential atomic network like the one used in the ANI-2x model"""
        dims_for_atoms = {
            "H": (aev_dim, 256, 192, 160),
            "C": (aev_dim, 224, 192, 160),
            "N": (aev_dim, 192, 160, 128),
            "O": (aev_dim, 192, 160, 128),
            "S": (aev_dim, 160, 128, 96),
            "F": (aev_dim, 160, 128, 96),
            "Cl": (aev_dim, 160, 128, 96),
        }
        return self.standard(dims_for_atoms[atom], **kwargs)

    def setup_nets(self, aevsize):
        modules = []
        if self.netlike1x:
            for a in self.elements:
                network = self.like_1x(a, aev_dim=aevsize)
                modules.append(network)
        if self.netlike2x:
            for a in self.elements:
                network = self.like_2x(a, aev_dim=aevsize)
                modules.append(network)
        return modules

    def init_params(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, a=1.0)
            if not self.bias:
                None
            else:
                torch.nn.init.zeros_(m.bias)

    def model_creator(self, aev_computer):
        modules = self.setup_nets(aev_computer.aev_length)
        if self.personal:
            from models.nets import ANIModelCharge

            nn = ANIModelCharge(modules, aev_computer)
            nn.apply(self.init_params)
            model = nn.to(self.device)
        else:
            nn = torchani.ANIModel(modules)
            nn.apply(self.init_params)
            model = torchani.nn.Sequential(aev_computer, nn).to(self.device)
        return nn, model, modules

    def AdamWOpt_build(self, modules, weight_decay):
        params = []
        for mod in modules:
            for i in range(4):
                if weight_decay[i]:
                    params.append(
                        {"params": [mod[i + i].weight], "weight_decay": weight_decay[i]}
                    )
                else:
                    params.append({"params": [mod[i + i].weight]})
        AdamW = torch.optim.AdamW(params)
        return AdamW

    def LR_Plat_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
        )
        return scheduler

    def log_setup(self):
        date = self.now.strftime("%Y%m%d_%H%M")
        log = "{}{}_{}".format(self.logdir, date, self.projectlabel)
        assert (
            os.path.isdir(log) == False
        ), "Oops! This project sub-directory already exists."
        if not os.path.isdir(log):
            print("creating your log sub-directory")
            os.makedirs(log)
        training_writer = torch.utils.tensorboard.SummaryWriter(log_dir=log + "/train")
        latest_checkpoint = "{}/latest.pt".format(log)
        best_checkpoint = "{}/best.pt".format(log)
        # shutil.copy(self.train_file, '{}/trainer.py'.format(log))
        shutil.copyfile(
            "/data/khuddzu/personal_trainer/personal_trainer/protocol.py",
            "{}/protocol.py".format(log),
        )
        if self.personal:
            shutil.copy("models/nets.py", "{}/model.py".format(log))
            shutil.copy("editor.ini", "{}/editor.ini".format(log))
        return training_writer, latest_checkpoint, best_checkpoint

    def save_model(self, nn, optimizer, energy_shifter, checkpoint, lr_scheduler):
        torch.save(
            {
                "model": nn.state_dict(),
                "AdamW": optimizer.state_dict(),
                "self_energies": energy_shifter,
                "AdamW_scehduler": lr_scheduler,
            },
            checkpoint,
        )

    def restart_train(self, latest_checkpoint, nn, optimizer, lr_scheduler):
        if os.path.isfile(latest_checkpoint):
            checkpoint = torch.load(latest_checkpoint)
            nn.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["AdamW"])
            lr_scheduler.load_state_dict(checkpoint["AdamW_scheduler"])

    def eA2debeye(x):
        return x / 0.20819434

    def validate(self, validation, model):
        valdict = {}
        mse_sum = torch.nn.MSELoss(reduction="sum")
        mse = torch.nn.MSELoss(reduction="none")
        total_energy_mse = 0.0
        count = 0
        charge_count = 0
        # Doing dipole code by hand, adding to when charges is true for sake of
        # time
        if self.charges:
            total_charge_mse = 0.0
        #     total_dipole_mse = 0.0
        if self.forces:
            total_force_mse = 0.0
        if self.dipole:
            total_dipole_mse = 0.0
        with torch.no_grad():
            for properties in validation:
                species = properties["species"].to(self.device)
                coordinates = properties["coordinates"].to(self.device).float()
                true_energies = properties["energies"].to(self.device).float()
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                if self.forces:
                    true_forces = properties["forces"].to(self.device).float()
                if self.dipole:
                    properties["dipoles"].to(self.device).float()
                if self.charges:
                    true_charges = properties[self.charge_type].to(self.device).float()
                    # true_dipoles = properties['dipoles'].to(self.device).float()
                    charge_count += true_charges.flatten().shape[0]
                    # dipole_count += true_dipoles.flatten().shape[0]
                if self.personal:
                    if self.charges:
                        (
                            _,
                            predicted_energies,
                            predicted_atomic_energies,
                            predicted_charges,
                            excess_charge,
                            coulomb,
                            correction,
                            predicted_dipoles,
                        ) = model((species, coordinates))
                    if self.dipole:
                        raise NotImplementedError(
                            "Currently there is no setup here for dipole calculation."
                        )
                    if self.forces:
                        raise NotImplementedError(
                            "Currently there is no setup here for force calculation."
                        )
                else:
                    if self.dipole:
                        raise TypeError(
                            "Published ANI does not currently support dipoles."
                        )
                    if self.charges:
                        raise TypeError(
                            "Published ANI does not currently support charge prediction."
                        )
                    _, predicted_energies = model((species, coordinates))
                count += true_energies.shape[0]
                total_energy_mse += mse_sum(predicted_energies, true_energies).item()
                if self.forces:
                    forces = -torch.autograd.grad(
                        predicted_energies.sum(), coordinates
                    )[0]
                    total_force_mse += (
                        mse(true_forces, forces).sum(dim=(1, 2)) / (3 * num_atoms)
                    ).sum()
                if self.dipole:
                    raise NotImplementedError(
                        "Currently there is no setup here for dipole calculation."
                    )
                if self.charges:
                    # total_charge_mse += mse_sum(predicted_charges.sum(dim=1), true_charges.sum(dim=1)).item()
                    total_charge_mse += mse_sum(
                        predicted_charges.flatten(), true_charges.flatten()
                    ).item()
                    # total_dipole_mse += mse_sum(predicted_dipoles.flatten(), true_dipoles.flatten()).item()
        energy_rmse = torchani.units.hartree2kcalmol(
            math.sqrt(total_energy_mse / count)
        )
        valdict["energy_rmse"] = energy_rmse
        if self.forces:
            force_rmse = torchani.units.hartree2kcalmol(
                math.sqrt(total_force_mse / count)
            )
            valdict["force_rmse"] = force_rmse
        if self.dipole:
            raise NotImplementedError("Currently we aren't doing dipoles")
            dipole_rmse = self.eA2debeye(math.sqrt(total_dipole_mse / count))
            valdict["dipole_rmse"] = dipole_rmse
        if self.charges:
            charge_rmse = math.sqrt(total_charge_mse / charge_count)
            # dipole_rmse = torchani.units.ea2debye(math.sqrt(total_dipole_mse/dipole_count))
            valdict["charge_rmse"] = charge_rmse
            # valdict['dipole_rmse']=dipole_rmse
        return valdict

    def trainer(self):
        aev_computer = self.AEV_Computer()
        energy_shifter = self.Energy_Shifter()
        training, validation = self.datasets_loading(energy_shifter)
        nn, model, modules = self.model_creator(aev_computer)
        AdamW = self.AdamWOpt_build(modules, self.weight_decay)
        LRscheduler = self.LR_Plat_scheduler(AdamW)
        training_writer, latest_pt, best_pt = self.log_setup()
        if self.mtl_loss:
            if self.num_tasks > 1:
                mtl = MTLLoss(num_tasks=self.num_tasks).to(self.device)
                AdamW.param_groups[0]["params"].append(
                    mtl.log_sigma
                )  # avoids LRdecay problem
            else:
                raise ValueError("You need to set your number of tasks for MTLLoss")
        mse = torch.nn.MSELoss(reduction="none")
        print("training starting from epoch", LRscheduler.last_epoch + 1)

        for _ in range(LRscheduler.last_epoch + 1, self.max_epochs):
            valrmse = self.validate(validation, model)
            for k, v in valrmse.items():
                training_writer.add_scalar(k, v, LRscheduler.last_epoch)
            learning_rate = AdamW.param_groups[0]["lr"]
            if learning_rate < self.earlylr:
                break
            # best checkpoint
            criteria = valrmse["energy_rmse"]
            if LRscheduler.is_better(criteria, LRscheduler.best):
                print(
                    "Saving the model, epoch={}, RMSE = {}".format(
                        (LRscheduler.last_epoch + 1), criteria
                    )
                )
                self.save_model(nn, AdamW, energy_shifter, best_pt, LRscheduler)
                for k, v in valrmse.items():
                    training_writer.add_scalar(
                        "best_{}".format(k), v, LRscheduler.last_epoch
                    )
            LRscheduler.step(criteria)
            for i, properties in tqdm.tqdm(
                enumerate(training),
                total=len(training),
                desc="epoch {}".format(LRscheduler.last_epoch),
            ):
                ## Get Properties##
                species = properties["species"].to(self.device)
                coordinates = (
                    properties["coordinates"]
                    .to(self.device)
                    .float()
                    .requires_grad_(True)
                )
                true_energies = properties["energies"].to(self.device).float()
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                ## Compute predicted ##
                if self.personal:
                    if self.dipole:
                        raise NotImplementedError(
                            "Currently there is no setup here for dipole calculation."
                        )
                    if self.charges:
                        true_charges = properties[self.charge_type].to(self.device)
                        # true_dipoles = properties['dipoles'].to(self.device).float()
                        (
                            _,
                            predicted_energies,
                            predicted_atomic_energies,
                            predicted_charges,
                            excess_charge,
                            coulomb,
                            correction,
                            predicted_dipoles,
                        ) = model((species, coordinates))
                    else:
                        raise AttributeError(
                            "What personal thing are you trying to do here?"
                        )
                else:
                    if self.dipole:
                        raise TypeError(
                            "Published ANI does not currently support dipoles."
                        )
                    if self.charges:
                        raise TypeError(
                            "Published ANI does not currently support charge prediction."
                        )
                    _, predicted_energies = model((species, coordinates))
                if self.forces:
                    true_forces = properties["forces"].to(self.device).float()
                    forces = -torch.autograd.grad(
                        predicted_energies.sum(),
                        coordinates,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                ## Get loss##
                if self.mtl_loss:
                    energy_loss = (
                        mse(predicted_energies, true_energies) / num_atoms.sqrt()
                    ).mean()
                    if self.forces and self.dipole:
                        raise NotImplementedError(
                            "We currently arent acknowledging this combo"
                        )
                        force_loss = (
                            mse(true_forces, forces).sum(dim=(1, 2)) / (3.0 * num_atoms)
                        ).mean()
                        dipole_loss = (
                            torch.sum(
                                (mse(predicted_dipoles, true_dipoles)) / 3.0, dim=1
                            )
                            / num_atoms.sqrt()
                        ).mean()
                        loss = mtl(energy_loss, force_loss, dipole_loss)
                    if self.forces:
                        force_loss = (
                            mse(true_forces, forces).sum(dim=(1, 2)) / (3.0 * num_atoms)
                        ).mean()
                        loss = mtl(energy_loss, force_loss)
                    if self.dipole:
                        raise NotImplementedError("We currently arent doing dipoles.")
                        dipole_loss = (
                            torch.sum(
                                (mse(predicted_dipoles, true_dipoles)) / 3.0, dim=1
                            )
                            / num_atoms.sqrt()
                        ).mean()
                        loss = mtl(energy_loss, dipole_loss)
                    if self.charges:
                        charge_loss = (
                            mse(predicted_charges, true_charges).sum(dim=1) / num_atoms
                        ).mean()
                        # dipole_loss = (torch.sum((mse(predicted_dipoles, true_dipoles))/3.0, dim=1) / num_atoms.sqrt()).mean()
                        loss, precisions, loss_terms = mtl(energy_loss, charge_loss)
                        training_writer.add_scalar(
                            "charge_loss", charge_loss, LRscheduler.last_epoch
                        )
                        training_writer.add_scalar(
                            "energy_loss", energy_loss, LRscheduler.last_epoch
                        )
                        # training_writer.add_scalar('dipole_loss', dipole_loss, LRscheduler.last_epoch)
                        training_writer.add_scalar(
                            "energy_term",
                            precisions[0] * loss_terms[0],
                            LRscheduler.last_epoch,
                        )
                        training_writer.add_scalar(
                            "charge_term",
                            precisions[1] * loss_terms[1],
                            LRscheduler.last_epoch,
                        )
                        # training_writer.add_scalar('dipole_term', precisions[2]*loss_terms[2], LRscheduler.last_epoch)
                else:
                    energy_loss = (
                        mse(predicted_energies, true_energies) / num_atoms.sqrt()
                    ).mean()
                    loss = energy_loss
                    training_writer.add_scalar(
                        "energy_loss", energy_loss, LRscheduler.last_epoch
                    )
                    if self.charges:
                        charge_loss = (
                            mse(predicted_charges, true_charges).sum(dim=1) / num_atoms
                        ).mean()
                        loss += self.loss_beta * charge_loss
                        training_writer.add_scalar(
                            "charge_loss", charge_loss, LRscheduler.last_epoch
                        )
                    if self.forces:
                        force_loss = (
                            mse(true_forces, forces).sum(dim=(1, 2)) / (3.0 * num_atoms)
                        ).mean()
                        loss += force_loss
                    if self.dipole:
                        raise NotImplementedError("We currently arent doing dipoles.")
                        dipole_loss = (
                            torch.sum(
                                (mse(predicted_dipoles, true_dipoles)) / 3.0, dim=1
                            )
                            / num_atoms.sqrt()
                        ).mean()
