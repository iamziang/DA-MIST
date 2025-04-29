import numpy as np
import os
import cv2

clip_len = 16

# the dir of testing images
video_root = '/iAESDataset/frames/'   ## the path of test videos
subfolders = ['anomaly', 'normal']
feature_list = '/DA-MIST/list/dViAEs_Endo_Test.list'

all_subfolder_contents = {}
for subfolder in subfolders:
    subfolder_path = os.path.join(video_root, subfolder)
    all_subfolder_contents[subfolder] = {name: os.path.join(subfolder_path, name) 
                                         for name in os.listdir(subfolder_path) 
                                         if os.path.isdir(os.path.join(subfolder_path, name))}

# the ground truth txt
gt_txt = '/DA-MIST/list/dViAEs_Endo_Annotation.txt'     ## the path of test annotations
gt_lines = list(open(gt_txt))
gt = []
lists = list(open(feature_list))
tlens = 0
vlens = 0
for idx in range(len(lists)):
    # name = lists[idx].strip('\n').split('/')[-1]
    name = lists[idx].strip('\n').split()
    if '_0.npy' not in name[0]:
        continue
    vname = name[0][:-10]

    found = False
    for subfolder, contents in all_subfolder_contents.items():
        if vname in contents:
            found = True
            vname_path = contents[vname]
            jpg_files = [f for f in os.listdir(vname_path) if f.endswith('.jpg') and os.path.isfile(os.path.join(vname_path, f))]
            lens = len(jpg_files)
            break
    if not found:
        print(f"Folder '{vname}' was not found in any of the subfolders.")
        continue  


    # the number of testing images in this sub-dir

    gt_vec = np.zeros(lens).astype(np.float32)
    if '_A' not in vname:
        for gt_line in gt_lines:
            if vname in gt_line:
                gt_content = gt_line.strip('\n').split()
                abnormal_fragment = [[int(gt_content[i]),int(gt_content[j])] for i in range(1,len(gt_content),2) \
                                        for j in range(2,len(gt_content),2) if j==i+1]
                if len(abnormal_fragment) != 0:
                    abnormal_fragment = np.array(abnormal_fragment)
                    for frag in abnormal_fragment:
                        gt_vec[frag[0]:frag[1]]=1.0
                break
    # mod = (lens-1) % clip_len # minusing 1 is to align flow  rgb: minusing 1 when extracting features
    # gt_vec = gt_vec[:-1]
    mod = (lens) % clip_len 
    gt_vec = gt_vec[:]
    if mod:
        gt_vec = gt_vec[:-mod]
    gt.extend(gt_vec)
    # if sum(gt_vec)/len(gt_vec):
    tlens += len(gt_vec)
    vlens += sum(gt_vec)



if len(gt) == tlens and sum(gt) == vlens:
    # If consistency check passes, save gt to file
    np.save('dViAEs_gt.npy', gt)
    print("File saved: Consistency check passed.")
    print(f"Length of gt: {len(gt)}, expected: {tlens}")
    print(f"Number of 1s in gt: {sum(gt)}, expected: {vlens}")
else:
    # If consistency check fails, issue a warning
    print("Warning: Data inconsistency detected!")
    print(f"Length of gt: {len(gt)}, expected: {tlens}")
    print(f"Number of 1s in gt: {sum(gt)}, expected: {vlens}")

