import torch
import torchani
from torchani.transforms import AtomicNumbersToIndices, SubtractSAE
from pathlib import Path
import configparser


class InputBuild():
    """
    Functions that intialize different aspects of the training process.
    The functions here are standard, general functions that are required in every currently available method.
    """

    def __init__(self, config_file):
        # Create a configuration object
        config = configparser.ConfigParser(
    allow_no_value=True, inline_comment_prefixes="#")
        config.read(config_file)

        # Create dictionaries for Global and Training variables
        gv = config['Global']
        tv = config['Trainer']

        # AEV_Computer
        self.netlike1x = gv.getboolean('netlike1x')
        self.netlike2x = gv.getboolean('netlike2x')
        if self.netlike1x == True:
            self.constants = '/data/khuddzu/torchani_sandbox/torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params'
            self.elements = ['H', 'C', 'N', 'O']
        elif self.netlike2x == True:
            self.constants = '/data/khuddzu/torchani_sandbox/torchani/resources/ani-2x_8x/rHCNOSFCl-5.1R_16-3.5A_a8-4.params'
            self.elements = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl']
        else:
            self.constants = eval(gv.get('constants'))

        # Energy_Shifter
        self.gsae_dat = eval(gv.get('gsae_dat'))
        self.functional = gv.get('functional')
        self.basis_set = gv.get('basis_set')

        # Dataset Loading
        self.ds_path = tv.get('ds_path')
        if self.ds_path == 'None':
            self.ds_path = eval(self.ds_path)
        self.h5_path = tv.get('h5_path')
        self.batch_size = tv.getint('batch_size')
        self.data_split = eval(tv.get('data_split'))
        self.include_properties = eval(tv.get('include_properties'))

    def AEV_Computer(self):
        consts = torchani.neurochem.Constants(self.constants)
        aev_computer = torchani.AEVComputer(**consts)
        return aev_computer

    def Energy_Shifter(self):
        if self.gsae_dat:
            _, energy_shifter = torchani.neurochem.load_sae(
                self.gsae_dat, return_dict=True)
        else:
            energy_shifter = torchani.utils.sorted_gsaes(
                self.elements, self.functional, self.basis_set)
        assert len(energy_shifter) == len(
    self.elements), "There must be a mistake with what atoms you are trying to use. The length of the EnergyShifter does not match the Elements"
        return energy_shifter

    def datasets_loading(self, energy_shifter):
        # ds_path can either be a path or None
        # if it is a path, it can either exist or not
        # if it is None -> In memory
        # if it is an existing path -> use it
        # if it is a nonoe existing path -> create it, and then use it
        in_memory = self.ds_path is None
        transform = torchani.transforms.Compose([AtomicNumbersToIndices(
            self.elements), SubtractSAE(self.elements, energy_shifter)])
        if in_memory:
            learning_sets = torchani.datasets.create_batched_dataset(self.h5_path,
                                        include_properties=self.include_properties,
                                        batch_size=self.batch_size,
                                        inplace_transform=transform,
                                        shuffle_seed=123456789,
                                        splits=self.data_split, direct_cache=True)
            training = torch.utils.data.DataLoader(learning_sets['training'],
                                               shuffle=True,
                                               num_workers=1,
                                               prefetch_factor=2,
                                               pin_memory=True,
                                               batch_size=None)
            validation = torch.utils.data.DataLoader(learning_sets['validation'],
                                                 shuffle=False,
                                                 num_workers=1,
                                                 prefetch_factor=2, pin_memory=True, batch_size=None)
        else:
            if not Path(self.ds_path).resolve().is_dir():
                h5 = torchani.datasets.ANIDataset.from_dir(self.h5_path)
                torchani.datasets.create_batched_dataset(h5,
                                                 dest_path=self.ds_path,
                                                 batch_size=self.batch_size,
                                                 include_properties=self.include_properties,
                                                 splits=self.data_split)
            # This below loads the data if dspath exists
            training = torchani.datasets.ANIBatchedDataset(
    self.ds_path, transform=transform, split='training')
            validation = torchani.datasets.ANIBatchedDataset(
    self.ds_path, transform=transform, split='validation')
            training = torch.utils.data.DataLoader(training,
                                           shuffle=True,
                                           num_workers=1,
                                           prefetch_factor=2,
                                           pin_memory=True,
                                           batch_size=None)
            validation = torch.utils.data.DataLoader(validation,
                                             shuffle=False,
                                             num_workers=1,
                                             prefetch_factor=2,
                                             pin_memory=True,
                                             batch_size=None)
        return training, validation

    def save_model(
    self,
    nn,
    optimizer,
    energy_shifter,
    checkpoint,
     lr_scheduler):
        torch.save({
            'model': nn.state_dict(),
            'AdamW': optimizer.state_dict(),
            'self_energies': energy_shifter,
            'AdamW_scehduler': lr_scheduler
            }, checkpoint)

    def restart_train(self, latest_checkpoint, nn, optimizer, lr_scheduler):
        if os.path.isfile(latest_checkpoint):
            checkpoint = torch.load(latest_checkpoint)
            nn.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['AdamW'])
            lr_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])

    # CALLS MISSING FUNCTION
    """
    def model_loader(self, wkdir, checkpoint):
        aev_computer = self.AEV_Computer()
        energy_shifter = self.Energy_Shifter()
        model, nn = self.model_builder(aev_computer, wkdir, checkpoint)
        return aev_computer, energy_shifter, model, nn

"""


class Skeleton():
   from torchani import atomics

   def __init__(self, config_file):
        # Create a configuration object
        config = configparser.ConfigParser(
    allow_no_value=True, inline_comment_prefixes="#")
        config.read(config_file)

        # Create dictionaries for Global and Training variables
        gv = config['Global']
        tv = config['Trainer']

"""
if like1x or like2x then do standard in torchani atomics
if raise model capacity is true than do a different function, which maybe is also in atomics? this will allow for you to create different model architectures with ease. I think it will input into your model a module that is a module list, so you have nn which is the shared, energy nn and charge nn
also, like 2x and 1x will be necessary to determine which elements we are looking at and what aev 
init params should be here too
change model creator so that you have to input your own nn. this will make it more general for everyone and they can easily call their own model class
log setup could go here too, since it is crucial for model saving
