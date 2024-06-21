from personal_trainer import protocol
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#wkdir = 'PATH TO MODEL AND DIRECTORY WITH CHECKPOINT FILE'
wkdir = '/data/khuddzu/networks/ANI-2x_mbis/training_center/logs/20230821_1223_mbis_ANI2x_screened_-4_2_mtl_trial1/'
pe = protocol.personal_trainer(f'{wkdir}/editor.ini', device=device)
aev_computer, self_energies, model, nn = pe.model_loader(wkdir=wkdir,
                                                        checkpoint = 'best.pt')
print(model)
