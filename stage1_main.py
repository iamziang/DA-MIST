import os
import numpy as np
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from core import utils
from configs.options import *
from configs.config import *
from train import *
from test import test_src
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
    damist = DA_MIST(transformer_config = SATAformer(512, 2, 4, 128, 512, dropout=0.5), event_mem = Event_Memory_Unit(a_nums=60, n_nums=60, flag="stage1")).cuda()

    normal_train_loader = data.DataLoader(
        Stage1Video(root_dir = config.root_src, mode = 'Train',modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = True),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    abnormal_train_loader = data.DataLoader(
        Stage1Video(root_dir = config.root_src, mode='Train', modal = config.modal, num_segments = 200, len_feature = config.len_feature, is_normal = False),
            batch_size = 64,
            shuffle = True, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn, drop_last = True)
    test_loader = data.DataLoader(
        Stage1Video(root_dir = config.root_src, mode = 'Test', modal = config.modal, num_segments = config.num_segments, len_feature = config.len_feature),
            batch_size = 3,
            shuffle = False, num_workers = config.num_workers,
            worker_init_fn = worker_init_fn)

    test_info = {"step": [], "auc": [],"ap":[],"ac":[]}
    
    best_auc = 0

    criterion = stage1_Loss()
    optimizer = torch.optim.Adam(damist.parameters(), lr = config.lr, betas = (0.9, 0.999), weight_decay = 0.00005)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50, verbose=False)
    writer = SummaryWriter(log_dir=os.path.join(config.output_path, 'stage1_logs'))
    
    test_src(damist, test_loader, test_info, 0, writer)
    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,   
            dynamic_ncols = True     
        ):   
        # if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = config.lr[step - 1]
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)

        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)
        train_stage1(damist, normal_loader_iter,abnormal_loader_iter, optimizer, criterion, step, writer)
        if step % 10 == 0 and step > 10:
            test_src(damist, test_loader, test_info, step, writer)
            # scheduler.step(test_info["ap"][-1])
            # current_lr = optimizer.param_groups[0]['lr']
            # writer.add_scalar('Learning Rate', current_lr, step)
            if test_info["ap"][-1] > best_auc:
                best_auc = test_info["ap"][-1]
                utils.save_best_record(test_info, 
                    os.path.join(config.output_path, "stage1_best_record_{}.txt".format(config.seed)))
                torch.save(damist.state_dict(), os.path.join(args.weight_path, \
                    "stage1_trans_{}.pkl".format(config.seed)))
            if step == config.num_iters:
                torch.save(damist.state_dict(), os.path.join(args.weight_path, \
                    "stage1_trans_{}.pkl".format(step)))


