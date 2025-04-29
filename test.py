import torch
from configs.options import *
from configs.config import *
from model.networks import *
import numpy as np
from core.dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def test_src(net, test_loader, test_info, step, writer, model_file = None):
    with torch.no_grad():
        net.eval()
        net.stage = net.event_mem.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/Cholec_gt.npy")
        frame_predict = None
        cls_label = []  
        cls_pre = []   
        for i in range(len(test_loader.dataset)//3):

            _data, _label = next(load_iter)
            
            _data = _data.cuda()  
            _label = _label.cuda()  
            cls_label.append(int(_label[0])) 
            outputs = net(_data, mode="test")
            a_predict = outputs['pred'].cpu().numpy().mean(0)
            cls_pre.append(1 if a_predict.max()>0.5 else 0)  
            fpre_ = np.repeat(a_predict, 16)   
            if frame_predict is None:         
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])  
        
        binary_predictions = (frame_predict > 0.5).astype(int)

        false_positives = np.sum((binary_predictions == 1) & (frame_gt == 0))
        true_negatives = np.sum(frame_gt == 0)
        far_score = false_positives / true_negatives if true_negatives > 0 else 0
            
        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)  
        auc_score = auc(fpr, tpr)
      
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
       
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)
      
        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)
        
        
        writer.add_scalar('Metrics/AUC', auc_score, step)
        writer.add_scalar('Metrics/AP', ap_score, step)
        writer.add_scalar('Metrics/Accuracy', accuracy, step)

def test_full(net, test_loader, test_info, step, writer, model_file = None):
    with torch.no_grad():
        net.eval()
        net.stage = net.event_mem.flag = net.scene_mem.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/Full_gt.npy")
        frame_predict = None

        for i in range(len(test_loader.dataset)//3):
            _data, _ = next(load_iter)      
            _data = _data.cuda()  
            outputs = net(_data, mode="test")
            a_predict = outputs['pred'].cpu().numpy().mean(0)
            fpre_ = np.repeat(a_predict, 16)   
            if frame_predict is None:         
                frame_predict = fpre_
            else:
                frame_predict = np.concatenate([frame_predict, fpre_])  
                   
        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)  
        auc_score = auc(fpr, tpr)
       
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)

        accuracy = 1
      
        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)
        
        
        writer.add_scalar('Metrics/AUC', auc_score, step)
        writer.add_scalar('Metrics/AP', ap_score, step)
        writer.add_scalar('Metrics/Accuracy', accuracy, step)
       