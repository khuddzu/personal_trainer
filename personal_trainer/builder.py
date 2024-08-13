import torch
import torchani
from typing import List, Optional, Union


class AEVBuild:

    def __init__(self, config_handler):
        self.config = config_handler

    def _term_builder(self):
        

class ModelBuilder:
    def __init__(self, config_handler):
        self.config = config_handler
        self.activation_mapping = {
            'GELU': torch.nn.GELU(), 
            'CELU': torch.nn.CELU(alpha=0.1)
        }
   
    def architecture(self, 
            elements: List[str] = ['H', 'C', 'N', 'O'],
            bias: bool = True, 
            activation: Optional[str] = None,
            classifiers: int = 1) -> List[torch.nn.Module]:
        
        if activation:
            activation = self.activation_mapping[activation.upper()]
        
        if netlike1x:
            modules = [torchani.atomics.like_1x(atom, activation=activation, bias=bias, classifier_out=class_out) for atom in elements]
        
        if netlike2x:
            modules = [torchani.atomics.like_2x(atom, activation=activation, bias=bias, classifier_out=class_out) for atom in elements]
        return modules

    def init_params(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, a=1.0)
            if self.inputs['bias']:
                None
            else:
                torch.nn.init.zeros_(m.bias)

    def standard_model(self):
        nn = torchani.ANIModel(self.modules)
        nn.apply(self.init_params)
        model = torchani.nn.Sequential(self.aev_computer, nn).to(self.device)
        return nn, model


