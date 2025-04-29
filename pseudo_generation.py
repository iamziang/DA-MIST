import os
import torch
import torch.utils.data as data
import numpy as np
from model.networks import *
from core.dataset_loader import *
from configs.options import *
from configs.config import *


def sharpen(scores, tau=0.5):
    """
    Sharpening function for binary classification to enhance label confidence based on anomaly scores.
    
    Arguments:
    scores -- numpy array of shape (n_samples,), where each element is the anomaly score for a frame.
    tau -- temperature parameter controlling the sharpness.

    Returns:
    sharpened_probabilities -- numpy array of shape (n_samples,) with sharpened probabilities for the anomaly class.
    """
    predicted_probabilities = np.vstack((scores, 1-scores)).T
    probabilities_raised = np.power(predicted_probabilities, 1/tau)
    denominator = probabilities_raised[:, 0] + probabilities_raised[:, 1]
    sharpened_probabilities = probabilities_raised[:, 0] / denominator
    
    return sharpened_probabilities

def setup_detect_model(weights_path):
    damist = DA_MIST(transformer_config = SATAformer(512, 2, 4, 128, 512, dropout=0.5), event_mem = Event_Memory_Unit(a_nums=60, n_nums=60, flag="Test")).cuda()
    state_dict = torch.load(weights_path)
    damist.load_state_dict(state_dict, strict=False)
    return damist

if __name__ == "__main__":
    args = parse_args()  
    config = Config(args)
    worker_init_fn = None
   
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)
    config.len_feature = 1024

    detector_weights_path = "./weights/stage1_trans_3407.pkl"
    detector = setup_detect_model(detector_weights_path).eval()

    test_loader = data.DataLoader(
            PseudoVideo(root_dir = 'features/full/', mode = 'Train', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
                batch_size = 3,
                shuffle = False, num_workers = 0,
                worker_init_fn = worker_init_fn)
    load_iter = iter(test_loader)
    l = len(test_loader.dataset)
    
    for i in range(len(test_loader.dataset)//3):

        _data, _label, _name = next(load_iter)
        base_name = os.path.splitext(_name[0])[0]

        _data = _data.cuda() 
        _label = _label.cuda()  
        out = detector(_data, mode="test") 
        a_predict = out['pred'].cpu().detach().numpy().mean(0)
        pseudo_labels = np.zeros_like(a_predict)

        is_negative_bag = "_A" in base_name
        
        if not is_negative_bag:
            pseudo_labels = sharpen(a_predict)
        
        # np.savetxt(
        #     os.path.join("pseudo_label/test_train", f"{base_name[:-6]}_pseudo.txt"),  #
        #     pseudo_labels,  
        #     fmt="%.6f"  
        # )
        np.save(os.path.join("pseudo_label/tgt", f"{base_name[:-6]}_pseudo.npy"), pseudo_labels)
    
    print('finished!')
