from trainer import personal_trainer
import torch
import torchani
from nets import ANIModelAIM
import datetime
import os

now = datetime.datetime.now()

#print(torchani.__file__)
#print(torchani.__version__)

#['H'. 'C'. 'N'. 'O', 'S', 'F', 'Cl']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f'... GPU allocated: {torch.cuda.get_device_name(0)}')

pt = personal_trainer(device = device,
            netlike1x = True,
            netlike2x = False,
            functional = 'wb97x',
            basis_set = '631gd',
            forces  = True,
            charges = False,
            dipole = False,
            constants = None,
            elements = ['H', 'C', 'N', 'O'],
            gsae_dat = None,
            batch_size = 2048,
            ds_path = '/data/khuddzu/ANI-Data/batched_data/ani1x/',
            h5_path = '/data/khuddzu/ANI-Data/data/ani1x/',
            include_properties = ['energies', 'species', 'coordinates', 'forces'],
            logdir = 'logs/1x/', 
            projectlabel = 'ani1xGSAE_biasfalse_network',
            train_file  = os.path.abspath(__file__), 
            now = now,
            data_split = {'training': 0.8, 'validation': 0.2},
            activation = torch.nn.GELU(),
            bias = False,
            classifier_out = 1, 
            num_tasks = 2, 
            personal = False,
            weight_decay = [6.1E-5, None, None, None],
            lr_factor = 0.7,
            lr_patience = 14,
            lr_threshold = 0.0006, 
            max_epochs = 2000, 
            early_stopping_learning_rate = 1.0E-7, 
            restarting = False)

pt.trainer()



