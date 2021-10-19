from personal_trainer import evaluator
import torch
import torchani

pe = evaluator.personal_evaluator(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            wkdir = '/data/khuddzu/GSAEs/training_center/logs/1x/20211013_0051_ani1xGSAE_biasfalse_network',
            checkpoint = 'best',
            netlike1x = True,
            netlike2x = False,
            functional = 'wb97x',
            basis_set = '631gd',
            constants = None,
            elements = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl'],
            gsae_dat = None,
            activation = torch.nn.GELU(),
            bias = False,
            classifier_out = 1,
            personal = False)


aev_computer, energy_shifter, model = pe.model_builder()

print(aev_computer)
print(energy_shifter)
print(model)
