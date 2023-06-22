import personal_trainer
from personal_trainer import protocol
pe = protocol.personal_trainer('editor.ini')
ac, es, model = pe.model_loader(wkdir = '/data/khuddzu/networks/ANI-2x_mbis/training_center/logs/20230620_1231_mbis_ANI2x_coulomb_CE_mtl_trial4/',
                                checkpoint = 'best.pt')
print(model)
