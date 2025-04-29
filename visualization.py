import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
import numpy as np
import torch
import cv2
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from model.networks import *
from extractors.I3dpt import I3D, Unit3Dpy

def forward_batch(b_data,net):
    b_data = b_data.transpose([0, 4, 1, 2, 3])
    b_data = torch.from_numpy(b_data).cuda().float()
    with torch.no_grad():
        b_features, _ = net(b_data,feature_layer=5)
    return b_features[:,:,0,0,0]

def load_video(path):
    frames = []
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame(frame)
        frames.append(frame)
    cap.release()
    return frames

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (340, 256), interpolation=cv2.INTER_LANCZOS4)
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    frame = (np.array(frame).astype(float) * 2 / 255) - 1
    return frame

def batch_split(clipped_length,batch_size,chunk_size):
    frame_indices = [] 
    for i in range(clipped_length):
        frame_indices.append(
            [j for j in range(i * 16, i * 16 + chunk_size)])

    frame_indices = np.array(frame_indices)
    chunk_num = frame_indices.shape[0]
    batch_num = int(np.ceil(chunk_num / batch_size))   
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)
    return frame_indices,batch_num

def plot_scores(scores, width, height):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.plot(scores, color='blue')
    ax.set(xlim=[0, len(scores)], ylim=[0, 1])
    ax.set_xticks([0, len(scores)//2, len(scores)])
    ax.set_yticks([0, 0.5, 1])
    ax.axhline(0.5, color='red', linestyle='--')
    plt.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plt.close(fig)
    return image.reshape(fig.canvas.get_width_height()[::-1] + (3,))[:, :, ::-1]

def video_processing(input_path, batch_size=10, extractor_weights=None, detector_weights=None):
    start_time = time()
    extractor = setup_extract_model(extractor_weights).eval()
    detector = setup_detect_model(detector_weights).eval()

    frames = load_video(input_path)
    frames_cnt = len(frames)
    clipped_length = math.ceil(frames_cnt /16)
    copy_length = (clipped_length *16)-frames_cnt
    frames += [frames[-1]] * copy_length

    frame_indices, batch_num = batch_split(clipped_length, batch_size = batch_size, chunk_size = 16)

    full_features = torch.Tensor().cuda()
    for batch_id in tqdm(range(batch_num)):
        batch_data = np.zeros(frame_indices[batch_id].shape + (224,224,3))      
        for i in range(frame_indices[batch_id].shape[0]):
            for j in range(frame_indices[batch_id].shape[1]):
                
                batch_data[i,j] = frames[frame_indices[batch_id][i][j]]
        full_features = torch.cat([full_features,forward_batch(batch_data,extractor)], dim = 0)

    print("{} has been extracted. Its shape:{}".format(input_path,full_features.size()))
    print("---------------------start detecting---------------")
    full_features = full_features.unsqueeze(0)
    out = detector(full_features, mode="test") 
    scores = out['pred'].cpu().detach().numpy()  # (1, 301)
    # res = detector(full_features)
    # scores = res["frame"].cpu().detach().numpy()  # (1, 301)
    scores = np.repeat(scores,16)[:-5]  # 4811
    end_time = time()
    cost_time = end_time - start_time
    print("Processing completed in {:.2f} seconds.".format(cost_time))
    print("fps:{}".format(frames_cnt/cost_time))
    return scores 
    
def setup_extract_model(weights_path):
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
    i3d.load_state_dict(torch.load(weights_path))
    i3d.cuda()
    return i3d


def setup_detect_model(weights_path):
    # damist = DA_MIST(transformer_config = SATAformer(512, 2, 4, 128, 512, dropout=0.5), event_mem = Event_Memory_Unit(a_nums=60, n_nums=60, flag="Test")).cuda()
    damist = DA_MIST(event_mem = Event_Memory_Unit(a_nums=60, n_nums=60, flag="test"), scene_mem = Scene_Memory_Unit(s_nums=60, t_nums=60, flag="test"), stage="test").cuda()
    state_dict = torch.load(weights_path)
    damist.load_state_dict(state_dict, strict=False)
    return damist


def plot_scores(scores, width, height, frame_num):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.plot(scores[:frame_num], color='blue')  
    ax.set_xlim([0, len(scores)])
    ax.set_xticks([0, len(scores)//2, len(scores)])  
    ax.set_xticklabels(['1', f'{len(scores)//2}', f'{len(scores)}'], fontsize=8)  
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1])  
    ax.set_yticklabels(['0', '0.5', '1'], fontsize=8)  
    ax.axhline(0.5, color='red', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Frame Num', fontsize=8)  
    ax.set_ylabel('Score', fontsize=8) 
    plt.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = image[:, :, ::-1]
    plt.close(fig)
    return image

def cv2write(video_path, score_list, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("video capture open fail")
        return

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    output_width, output_height = 455, 256
    graph_width = 455  
    total_width = output_width + graph_width 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (total_width, output_height))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (output_width, output_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        score_image = plot_scores(score_list, graph_width, output_height, frame_num)
        extended_frame = np.ones((output_height, total_width, 3), dtype=np.uint8) * 255
        extended_frame[:, :output_width] = frame
        extended_frame[:, output_width:] = score_image[:output_height, :graph_width]
        score = score_list[frame_num - 1] if frame_num <= len(score_list) else 0
        left_x_up = 10
        left_y_up = 10
        right_x_down = int(left_x_up + 200)
        right_y_down = int(left_y_up + 40)
        word_x = left_x_up + 5
        word_y = left_y_up + 15
        cv2.rectangle(extended_frame, (left_x_up, left_y_up), (right_x_down, right_y_down), (55,255,155), 2)
        cv2.putText(extended_frame, 'frame_num:{}'.format(frame_num), (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55,255,155), 1)
        cv2.putText(extended_frame, 'frame_score:{:.2f}'.format(score), (word_x, word_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255 if score > 0.5 else 55), 1)

        out.write(extended_frame) 
        frame_num += 1

    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print("Video saved to {}".format(output_path))

if __name__ == "__main__":
    input_path = "demo/videos/BLEED&BLUR.mp4"
    output_path = "demo/results/VIS_BLEED&BLUR.mp4"
    extractor_weights_path = "extractors/weights/fti3d.pth"
    detector_weights_path = "weights/stage2_trans_3407.pkl"
    scores = video_processing(input_path, extractor_weights=extractor_weights_path, detector_weights=detector_weights_path)
    cv2write(input_path, scores, output_path)
    