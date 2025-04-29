import torch
import torch.nn as nn
from model.modules import *
from core.utils import norm
import numpy as np

class DA_MIST(nn.Module):
    def __init__(self, transformer_config=None, event_mem=None, scene_mem=None, classifier=None, stage="None"):
        super().__init__()
        self.stage = stage  
        self.embedding = Temporal(1024, 512) 
        self.selfatt = transformer_config if transformer_config else SATAformer(512, 2, 4, 128, 512, dropout=0.5)
        self.classifier = classifier if classifier else Classifier(1024, 1)
        self.event_mem = event_mem if event_mem else Event_Memory_Unit(a_nums=60, n_nums=60, flag=self.stage)

        if self.stage == "stage2":
            self.scene_mem = scene_mem if scene_mem else Scene_Memory_Unit(s_nums=60, t_nums=60, flag=self.stage)
        else:
            self.scene_mem = None

    def _compute_orthogonality_loss(self, F_NB, F_AB, F_SB, F_TB):  
        def normalize(f):
            return f / (torch.norm(f, dim=-1, keepdim=True) + 1e-6)
        event_features = normalize(F_NB + F_AB)  # [batch, seq_len, dim]
        scene_features = normalize(F_SB + F_TB)  # [batch, seq_len, dim]
        dot_product = torch.bmm(event_features.transpose(1, 2), scene_features)  
        num_elements = scene_features.size(2) * scene_features.size(2)  
        return torch.norm(dot_product, p="fro") ** 2 / num_elements

    def forward(self, x, mode="train"):
        x = self.embedding(x)
        x = self.selfatt(x)
        event_output = self.event_mem(x)
        x_aug = event_output["F_M"]
        pred = self.classifier(x_aug)
        outputs = {"pred": pred}

        if mode == "test":
            return outputs

        if self.stage == "stage1":
            outputs["losses"] = {
                'triplet_loss': event_output.get("triplet_margin", 0),
                'kl_loss': event_output.get("kl_loss", 0),
                'distance_loss': event_output.get("distance", 0)
            }
            outputs["mem_result"] = {
                'A_att': event_output.get("A_att", 0),
                "N_att": event_output.get("N_att", 0),
                "A_Natt": event_output.get("A_Natt", 0),
                "N_Aatt": event_output.get("N_Aatt", 0),
            }
        elif self.stage == "stage2":
            scene_output = self.scene_mem(x)
            outputs["losses"] = {
                'scene_loss': scene_output["scene_loss"],
                'orth_loss': self._compute_orthogonality_loss(
                    event_output["F_NB"],
                    event_output["F_AB"],
                    scene_output["F_SB"],
                    scene_output["F_TB"]
                )
            }

        return outputs

    def get_params(self, component_name): 
        if component_name == 'event_mem':
            return self.event_mem.parameters()
        elif component_name == 'scene_mem':
            return self.scene_mem.parameters()
        elif component_name == 'classifier':
            return self.classifier.parameters()
        elif component_name == 'embedding': 
            return self.embedding.parameters()
        elif component_name == 'selfatt':  
            return self.selfatt.parameters()
        else:
            raise ValueError(f"Invalid component name: {component_name}.")



