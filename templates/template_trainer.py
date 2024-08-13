from personal_trainer import protocol
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pt = protocol.personal_trainer("editor.ini", device=device)
pt.trainer()
