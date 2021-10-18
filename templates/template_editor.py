from personal_trainer import protocol
import torch
import torchani
from models.nets import ANIModelAIM
import datetime
import os

now = datetime.datetime.now()

print(torchani.__file__)
print(torchani.__version__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'... GPU allocated: {torch.cuda.get_device_name(0)}')

pt = protocol.personal_trainer(device = device,
            netlike1x = False,                                      #Will generate your networks with architecture like ANI-1x  
            netlike2x = True,                                       #Will generate your networks with architecture like ANI-2x
            functional = 'wb97x',                                   #Functional of the level of theory of your dataset
            basis_set = '631gd',                                    #Basis set of the level of theory of your dataset
            forces  = False,                                        #True if training to forces
            charges = True,                                         #True if training with ANIAIM model
            dipole = False,                                         #True if training to dipoles
            constants = None,                                       #Path to constants file, if getting params from a separate file (netlike1x and netlike2x both need to be false and protocol code currently needs to be modified if you want to use this keyword)
            elements = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl'],        #List of elements in your dataset, this also specifies your species order, should match 1x of 2x allowed atoms
            gsae_dat = None,                                        #Path to a dat file containing GSAE values, if not getting them from torchani
            batch_size = 2048,                                      #Specified batch_size    
            ds_path = '/data/khuddzu/ANI-Data/batched_data/ani2x/', #Path to your batched datasets, if using; if in moria this is required to train 2x
            h5_path = '/data/khuddzu/ANI-Data/data/ani2x/',         #Path to your original h5 file. This is used if you havent batched yet, or if you are planning on not batching your datasets
            include_properties = ['energies', 'species', 'coordinates', 'forces'],  #Properties from your h5 file you want to include in your training and validation datasets
            logdir = 'logs/',                                       #Path to your logs directory, where your project will be saved
            projectlabel = 'ANIAIMtest',                            #Name of your project
            train_file  = os.path.abspath(__file__),                #Keep as is. This is used to copy this file to your logs
            now = now,                                              #Datetime specified above. This is used to properly date your project.
            data_split = {'training': 0.8, 'validation': 0.2},      #Training and validation split desired
            activation = torch.nn.GELU(),                           #Activation function you are using
            bias = False,                                           #Whether or not you want to use biases in your network
            classifier_out = 2,                                     #Number of outputs of your network
            num_tasks = 1,                                          #Number of tasks to use in MTLLoss
            personal = True,                                        #True if using a personal model. Protocol code is currently set up for ANIAIM output. Needs to be modified with changes
            weight_decay = [6.1E-5, None, None, None],              #List corresponding to weight decay by layer
            lr_factor = 0.7,                                        #ReduceLROnPlateau factor
            lr_patience = 14,                                       #ReduceLROnPlateau patience
            lr_threshold = 0.0006,                                  #ReduceLROnPlateau threshold
            max_epochs = 2000,                                      #Max number of epochs you are willing to train to
            early_stopping_learning_rate = 1.0E-7,                  #Stop training at this LR
            restarting = False)                                     #True if you are restarting training from a current pt file

pt.trainer()



