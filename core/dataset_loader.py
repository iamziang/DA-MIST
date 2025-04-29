import torch
import torch.utils.data as data
import os
import numpy as np
from core import utils
import random

class Stage1Video(data.DataLoader):
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path=root_dir
        self.mode=mode
        self.modal=modal
        self.num_segments = num_segments         
        self.len_feature = len_feature
        if self.modal == 'all':
            self.feature_path = []
            if self.mode == "Train":
                for _modal in ['RGB', 'Flow']:  
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features",_modal))
            else:
                for _modal in ['RGBTest', 'FlowTest']:
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features",_modal))
        else:
            self.feature_path = os.path.join(self.data_path, modal)
        split_path = os.path.join("list",'Cholec_Endo_{}.list'.format(self.mode))   
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[483:]  
            elif is_normal is False:
                self.vid_list = self.vid_list[:483]  
            else:
                assert (is_normal == None)
                print("Please sure is_normal = [True/False]")
                self.vid_list=[]
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data,label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        label=0
        if "_A" not in vid_name:  
            label=1     
        video_feature = np.load(os.path.join(self.feature_path,
                                vid_name )).astype(np.float32)
        if self.mode == "Train":
            new_feature = np.zeros((self.num_segments,self.len_feature)).astype(np.float32)
            sample_index = utils.random_perturb(video_feature.shape[0],self.num_segments)   
            for i in range(len(sample_index)-1):
                if sample_index[i] == sample_index[i+1]:
                    new_feature[i,:] = video_feature[sample_index[i],:]
                else:
                    new_feature[i,:] = video_feature[sample_index[i]:sample_index[i+1],:].mean(0)
                    
            video_feature = new_feature
        return video_feature, label    
    
class Stage2Video(data.Dataset):
    def __init__(self, root_dir, label_dir, mode, modal, num_segments, len_feature, seed=-1, domain_name=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path = root_dir
        self.label_path = label_dir
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.domain_name = domain_name
        self.feature_path = os.path.join(self.data_path, modal)

        if self.mode == "Train":
            if label_dir is None:
                raise ValueError("label_dir must be provided in Train mode")
            self.label_path = label_dir

        split_path = os.path.join("list", f'{self.domain_name}_Endo_{self.mode}.list')
        with open(split_path, 'r', encoding="utf-8") as split_file:
            self.vid_list = [line.split() for line in split_file]

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, label = self.get_data(index)
        return data, label

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        video_feature = np.load(os.path.join(self.feature_path, vid_name)).astype(np.float32)
        frames_label = torch.zeros(1, dtype=torch.float32)
        
        if self.mode == "Train":
            frames_label = np.load(os.path.join(self.label_path, vid_name[:-10] + '_pseudo.npy')).astype(np.float32)
            assert len(video_feature) == len(frames_label), "Feature and label length mismatch"
            start_index = random.randint(0, len(video_feature) - 1)
            clips = []
            labels = []

            for i in range(self.num_segments):  
                index = (start_index + i) % len(video_feature)
                clips.append(video_feature[index])
                labels.append(frames_label[index])

            video_feature = np.array(clips)
            frames_label = np.array(labels)
        
        return video_feature, frames_label

class PseudoVideo(data.DataLoader):
    def __init__(self, root_dir, mode, modal, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.data_path=root_dir
        self.mode=mode
        self.modal=modal
        self.num_segments = num_segments       
        self.len_feature = len_feature
        if self.modal == 'all':
            self.feature_path = []
            if self.mode == "Train":
                for _modal in ['RGB', 'Flow']:  
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features",_modal))
            else:
                for _modal in ['RGBTest', 'FlowTest']:
                    self.feature_path.append(os.path.join(self.data_path, "i3d-features",_modal))
        else:
            self.feature_path = os.path.join(self.data_path, modal)
        split_path = os.path.join("list",'Full_Endo_{}.list'.format(self.mode))  
        split_file = open(split_path, 'r',encoding="utf-8")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()
        
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, label, vid_name = self.get_data(index)
        return data, label, vid_name

    def get_data(self, index):
        vid_name = self.vid_list[index][0]
        label=0
        if "_A" not in vid_name:  
            label=1     
        video_feature = np.load(os.path.join(self.feature_path,
                                vid_name )).astype(np.float32)
        return video_feature, label , vid_name   
    
