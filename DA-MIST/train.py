import torch
import torch.nn as nn

class stage1_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()  

    def forward(self, outputs, _label):
        loss = {}
        _label = _label.float()

        mem_result = outputs['mem_result']
        cls_result = outputs['pred']
        triplet = outputs['losses']['triplet_loss']

        b = _label.size(0)//2
        t = cls_result.size(1)

        anomaly = torch.topk(cls_result, t//16 + 1, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)

        panomaly = torch.topk(1 - mem_result["N_Aatt"], t//16 + 1, dim=-1)[0].mean(-1)
        panomaly_loss = self.bce(panomaly, torch.ones((b)).cuda())

        A_att = torch.topk(mem_result["A_att"], t//16 + 1, dim=-1)[0].mean(-1)
        A_loss = self.bce(A_att, torch.ones((b)).cuda())

        N_loss = self.bce(mem_result["N_att"], torch.ones_like(mem_result["N_att"]).cuda())
        A_Nloss = self.bce(mem_result["A_Natt"], torch.zeros_like(mem_result["A_Natt"]).cuda())

        cost = anomaly_loss + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss) + 0.1 * triplet

        loss['total_loss'] = cost
        loss['att_loss'] = anomaly_loss
        loss['N_Aatt'] = panomaly_loss
        loss['A_loss'] = A_loss
        loss['N_loss'] = N_loss
        loss['A_Nloss'] = A_Nloss
        loss["triplet"] = triplet
        return cost, loss

class stage2_Loss(nn.Module):
    def __init__(self, lambda_scene=0.1, lambda_orth=0.1):
        super(stage2_Loss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.lambda_scene = lambda_scene
        self.lambda_orth = lambda_orth

    def forward(self, outputs, pseudo_labels):
        loss = {}
        y_hat_mix = outputs['pred']
        inst_loss = self.bce_loss(y_hat_mix, pseudo_labels)

        scene_loss = outputs['losses']['scene_loss']  
        orth_loss = outputs['losses']['orth_loss']  

        cost = inst_loss + self.lambda_scene * scene_loss + self.lambda_orth * orth_loss
       
        loss['total_loss'] = cost
        loss['inst_loss'] = inst_loss 
        loss['scene_loss'] = scene_loss  
        loss['orth_loss'] = orth_loss 

        return cost, loss


def train_stage1(net, normal_loader, abnormal_loader, optimizer, criterion, index, writer):
    net.train()
    net.stage = net.event_mem.flag = "stage1"
    ninput, nlabel = next(normal_loader)   
    ainput, alabel = next(abnormal_loader)  
    _data = torch.cat((ninput, ainput), 0)   
    _label = torch.cat((nlabel, alabel), 0)
    _data = _data.cuda()  
    _label = _label.cuda() 
    outputs = net(_data, mode="train")
    cost, loss = criterion(outputs, _label)
    writer.add_scalar('Loss/Train', cost.item(), index)
    for key, value in loss.items():
        writer.add_scalar(f'Loss/{key}', value.item(), index)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


def train_stage2(net, src_loader, tgt_loader, optimizer, criterion, index, writer):
    net.train()
    net.stage = net.event_mem.flag = net.scene_mem.flag = "stage2"
    sinput, slabel = next(src_loader)  
    tinput, tlabel = next(tgt_loader)  
    _data = torch.cat((sinput, tinput), 0)   
    _label = torch.cat((slabel, tlabel), 0)
    _data = _data.cuda()   
    _label = _label.cuda() 
    outputs = net(_data, mode="train")
    cost, loss = criterion(outputs, _label)
    writer.add_scalar('Loss/Train', cost.item(), index)
    for key, value in loss.items():
        writer.add_scalar(f'Loss/{key}', value.item(), index)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()