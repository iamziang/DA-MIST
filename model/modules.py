import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.layers import *
from core.utils import norm

class Temporal(nn.Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            nn.ReLU(),
        ) 
    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x

class SATAformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, State_Aware_Dual_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Scene_Memory_Unit(nn.Module):
    """
    Scene_Memory_Unit
    """
    def __init__(self, s_nums, t_nums, flag):
        super(Scene_Memory_Unit, self).__init__()
        self.flag = flag
        self.Smemory = Memory_Bank(nums=s_nums, dim=512)  # Source style memory bank
        self.Tmemory = Memory_Bank(nums=t_nums, dim=512)  # Target style memory bank
        self.BCE = nn.BCELoss()

    def forward(self, x):
        b, t, d = x.size()
        mid_point = b // 2
        S_x = x[:mid_point]   # Source domain samples
        T_x = x[mid_point:]   # Target domain samples

        # Retrieval scores from memory banks
        S_Satt, S_Saug = self.Smemory(S_x)  # Source similarity and augmented features from source bank
        T_Satt, T_Saug = self.Tmemory(S_x)  # Source similarity and augmented features from target bank
        S_Tatt, S_Taug = self.Smemory(T_x)  # Target similarity and augmented features from source bank
        T_Tatt, T_Taug = self.Tmemory(T_x)  # Target similarity and augmented features from target bank
        
        # Scene memory loss calculation
        scene_loss = (self.BCE(S_Satt, torch.ones_like(S_Satt)) +
                    self.BCE(T_Satt, torch.zeros_like(T_Satt)) +
                    self.BCE(S_Tatt, torch.zeros_like(S_Tatt)) +
                    self.BCE(T_Tatt, torch.ones_like(T_Tatt))) 

        # Concatenate augmented features for orthogonality loss computation
        f_sb = torch.cat([S_Saug, T_Saug], dim=0)
        f_tb = torch.cat([S_Taug, T_Taug], dim=0)

        return {
            "scene_loss": scene_loss,
            "F_SB": f_sb,
            "F_TB": f_tb
        }
    

class Event_Memory_Unit(nn.Module):
    """
    Event_Memory_Unit
    """
    def __init__(self, a_nums, n_nums, flag):
        super(Event_Memory_Unit, self).__init__()
        # Dual State Memory Banks
        self.flag = flag
        self.Amemory = Memory_Bank(nums=a_nums, dim=512)
        self.Nmemory = Memory_Bank(nums=n_nums, dim=512)
        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))   
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))  
        self.triplet = nn.TripletMarginLoss(margin=1)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()  
        epsilon = torch.randn_like(std)
        return mu + epsilon * std
    
    def latent_loss(self, mu, var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1))  
        return kl_loss

    def forward(self, x):
        if self.flag == "stage1":
            b, t, d = x.size()
            mid_point = b // 2
            N_x = x[:mid_point]  
            A_x = x[mid_point:]

            A_att, A_aug = self.Amemory(A_x)
            N_Aatt, N_Aaug = self.Nmemory(A_x)
            A_Natt, A_Naug = self.Amemory(N_x)
            N_att, N_aug = self.Nmemory(N_x)

            _, A_index = torch.topk(A_att, t // 16 + 1, dim=-1)
            negative_ax = torch.gather(A_x, 1, A_index.unsqueeze(2).expand(-1, -1, d)).mean(1)
            _, N_index = torch.topk(N_att, t // 16 + 1, dim=-1)
            anchor_nx = torch.gather(N_x, 1, N_index.unsqueeze(2).expand(-1, -1, d)).mean(1)
            _, P_index = torch.topk(N_Aatt, t // 16 + 1, dim=-1)
            positive_nx = torch.gather(A_x, 1, P_index.unsqueeze(2).expand(-1, -1, d)).mean(1)

            triplet_margin_loss = self.triplet(norm(anchor_nx), norm(positive_nx), norm(negative_ax)) 
            N_aug_mu = self.encoder_mu(N_aug)  
            N_aug_var = self.encoder_var(N_aug) 
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_var) 
            
            anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,1,-1).mean(1)  # 50 * 512 

            A_aug_new = self.encoder_mu(A_aug)
            negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b//2,1,-1).mean(1)  # 50 * 512 
            
            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)

            A_Naug = self.encoder_mu(A_Naug)  
            N_Aaug = self.encoder_mu(N_Aaug) 
          
            distance = torch.relu(100 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()
            x = torch.cat((x, (torch.cat([N_aug_new + A_Naug, A_aug_new + N_Aaug], dim=0))), dim=-1)

            return {
                "F_M": x,
                'triplet_margin': triplet_margin_loss,
                'kl_loss': kl_loss, 
                'distance': distance,
                'A_att': A_att,
                "N_att": N_att,
                "A_Natt": A_Natt,
                "N_Aatt": N_Aatt
            }
        
        elif self.flag == "stage2":
            _, A_aug = self.Amemory(x) 
            _, N_aug = self.Nmemory(x)  

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)
            
            return {"F_M": torch.cat([x, A_aug + N_aug], dim=-1),
                    "F_NB": N_aug,
                    "F_AB": A_aug
            }

        else:
            _, A_aug = self.Amemory(x)  # [64, 16, 512]
            _, N_aug = self.Nmemory(x)  # [64, 16, 512]

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)
            
            return {"F_M": torch.cat([x, A_aug + N_aug], dim=-1)}
        
class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(), nn.Linear(128,out_dim), nn.Sigmoid())
    def forward(self, x):
        return self.mlp(x).squeeze()


    
    
    
