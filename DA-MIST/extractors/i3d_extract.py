from multiprocessing import Pool
import os
from tqdm import tqdm 
import torch
import argparse
import cv2
import numpy as np
from I3dpt import I3D, Unit3Dpy
import math

def load_frame(frame_file, resize_size=(224, 224)):
    frame = cv2.imread(frame_file)
    frame = cv2.resize(frame, (340, 256), interpolation=cv2.INTER_LANCZOS4)
    frame = cv2.resize(frame, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LANCZOS4)
    frame = ((frame.astype(float) / 255) * 2) - 1
    return frame

def augment_data(data):
    # data b, t, h, w, c
    data_flip_h = np.array(data[:,:,:,::-1,:])  
    data_flip_v= np.array(data[:,:,::-1,:,:])  
    return [data, data_flip_h, data_flip_v]

def load_rgb_batch(frames_dir, rgb_files, frame_indices):
    batch_data = []
    for i in range(frame_indices.shape[0]):  # batchnum
        batch = []
        for j in range(frame_indices.shape[1]):  # chunksize
            img_path = os.path.join(frames_dir, rgb_files[frame_indices[i][j]])
            img_data = load_frame(img_path)
            batch.append(img_data)
        batch_data.append(np.stack(batch))
    return np.array(batch_data)     #  b, t, h, w, c

###---------------I3D model to extract snippet feature---------------------
# Input:  bx3x16x224x224
# Output: bx1024
def forward_batch(b_data,net):
    b_data = b_data.transpose([0, 4, 1, 2, 3])
    b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 18x3x16x224x224 
    with torch.no_grad():
        b_data = b_data.cuda().float()   # torch.Size([18, 3, 16, 224, 224])
        b_features,_ = net(b_data,feature_layer=5)   # torch.Size([18, 1024, 1, 1, 1])
    b_features = b_features.data.cpu().numpy()[:,:,0,0,0]  # (18, 1024)
    return b_features


def run(args_item):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model,video_dir, output_dir, batch_size, task_id=args_item
    mode='rgb'
    chunk_size = 16   
    frequency = 16
    sample_mode='resize'
    video_name=video_dir.split("/")[-1]
    assert(mode in ['rgb', 'flow'])
    assert(sample_mode in ['oversample', 'center_crop', 'resize'])
    save_file = '{}_{}.npy'.format(video_name, "i3d")
    if save_file in os.listdir(os.path.join(output_dir)):
        print("{} has been extracted".format(save_file))
        pass

    else:  
    # setup the model  
        i3d = I3D(400, modality='rgb', dropout_prob=0, name='inception')
        new_conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=7,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False
        )
        i3d.conv3d_0c_1x1 = new_conv3d_0c_1x1
        i3d.eval()
        i3d.load_state_dict(torch.load(load_model))
        i3d.to(device)

        
        rgb_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')], key=lambda x: int(x.split('_')[1].rstrip('.jpg')))
        frame_cnt = len(rgb_files) 
        clipped_length = frame_cnt // chunk_size 

        frame_indices = [list(range(i * frequency, min(i * frequency + chunk_size, frame_cnt)))
                 for i in range(clipped_length)]
        frame_indices = np.array(frame_indices)

        chunk_num = frame_indices.shape[0]   #  chunk_num = clipped_length

        batch_num = int(np.ceil(chunk_num / batch_size))  
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)

        full_features = [[] for i in range(3)]  # 3 crop

        for batch_id in tqdm(range(batch_num)):    
            batch_data = load_rgb_batch(video_dir, rgb_files, frame_indices[batch_id])    
            batch_data_three_aug = augment_data(batch_data)  
            for i in range(3):                    
                assert(batch_data_three_aug[i].shape[-2]==224)
                assert(batch_data_three_aug[i].shape[-3]==224)
                full_features[i].append(forward_batch(batch_data_three_aug[i],i3d)) 
           
        full_features = [np.concatenate(i, axis=0) for i in full_features] 
        full_features = [np.expand_dims(i, axis=0) for i in full_features]  
        full_features = np.concatenate(full_features, axis=0)      
        np.save(os.path.join(output_dir,save_file), full_features)

        print('{} done: {} / {}, {}'.format(
            video_name, frame_cnt, clipped_length, full_features.shape))  


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="rgb",type=str)
    parser.add_argument('--load_model',default="weights/fti3d.pth", type=str)
    parser.add_argument('--input_dir', default="../dataset/frames",type=str)
    parser.add_argument('--output_dir',default="../tgt/three", type=str)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--sample_mode', default="resize",type=str)
    parser.add_argument('--frequency', type=int, default=16)
    args = parser.parse_args()

    vid_list=[]
    for videos in os.listdir(args.input_dir):
        for video in os.listdir(os.path.join(args.input_dir,videos)):
            save_file = '{}_{}.npy'.format(video, "i3d")
            if save_file in os.listdir(os.path.join(args.output_dir)):
                print("{} has been extracted".format(save_file))
            else:
                vid_list.append(os.path.join(args.input_dir,videos,video))
    
    nums=len(vid_list)
    print("leave {} videos".format(nums))
    # pool = Pool(4)
    # pool.map(run, zip([args.load_model]*nums, vid_list, [args.output_dir]*nums,[args.batch_size]*nums,range(nums)))
    for i, vid in enumerate(vid_list):
        args_item = (args.load_model, vid, args.output_dir, args.batch_size, i)
        run(args_item)
