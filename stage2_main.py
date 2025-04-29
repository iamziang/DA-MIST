import os
import numpy as np
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from core import utils
from configs.options import *
from configs.config import *
from train import *
from test import test_full
from model.networks import *
from core.dataset_loader import *
from tqdm import tqdm

if __name__ == "__main__":
    args = parse_args()  
    config = Config(args)
    worker_init_fn = None
   
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)
    
    config.len_feature = 1024
    
    damist = DA_MIST(event_mem = Event_Memory_Unit(a_nums=60, n_nums=60, flag="stage2"), scene_mem = Scene_Memory_Unit(s_nums=60, t_nums=60, flag="stage2"), stage="stage2").cuda()
    stage1_weights = torch.load("weights/stage1_trans_3407.pkl")
    selected_weights = {}
    for name, param in stage1_weights.items():
        if name.startswith('embedding.') or name.startswith('selfatt.') or name.startswith('event_mem.'):
            selected_weights[name] = param
    
    damist.load_state_dict(selected_weights, strict=False)

    src_train_loader = data.DataLoader(
        Stage2Video(root_dir = config.root_full, label_dir='pseudo_label/src/', mode = 'Train',modal = config.modal, num_segments = 200, len_feature = config.len_feature, domain_name="Cholec"),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    tgt_train_loader = data.DataLoader(
        Stage2Video(root_dir = config.root_full, label_dir='pseudo_label/tgt/', mode='Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, domain_name="dViAEs"),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = data.DataLoader(
        Stage2Video(root_dir = config.root_full, label_dir=None , mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature, domain_name="Full"),
            batch_size = 3,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)

    test_info = {"step": [], "auc": [],"ap":[],"ac":[]}
    
    best_auc = 0

    criterion = stage2_Loss()
    optimizer = torch.optim.Adam(damist.parameters(), lr = config.lr, betas = (0.9, 0.999), weight_decay = 0.00005)
    writer = SummaryWriter(log_dir=os.path.join(config.output_path, 'stage2_logs'))


    test_full(damist, test_loader, test_info, 0, writer)
    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,   
            dynamic_ncols = True     
        ):   

        if (step - 1) % len(src_train_loader) == 0:
            src_loader_iter = iter(src_train_loader)

        if (step - 1) % len(tgt_train_loader) == 0:
            tgt_loader_iter = iter(tgt_train_loader)
        train_stage2(damist, src_loader_iter, tgt_loader_iter, optimizer, criterion, step, writer)
        if step % 10 == 0 and step > 10:
            test_full(damist, test_loader, test_info, step, writer)
            # scheduler.step(test_info["ap"][-1])
            # current_lr = optimizer.param_groups[0]['lr']
            # writer.add_scalar('Learning Rate', current_lr, step)
            if test_info["ap"][-1] > best_auc:
                best_auc = test_info["ap"][-1]
                utils.save_best_record(test_info, 
                    os.path.join(config.output_path, "stage2_best_record_{}.txt".format(config.seed)))
                torch.save(damist.state_dict(), os.path.join(args.weight_path, \
                    "stage2_trans_{}.pkl".format(config.seed)))
            if step == config.num_iters:
                torch.save(damist.state_dict(), os.path.join(args.weight_path, \
                    "stage2_trans_{}.pkl".format(step)))


