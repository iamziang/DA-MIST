import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

import videotransforms
from charades_dataset import Charades as Dataset
from charades_dataset import Val_Charades as Val_Dataset
from I3dpt import I3D, Unit3Dpy

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')  
parser.add_argument('-save_model', type=str)  
parser.add_argument('-root', type=str)  

def set_parameter_requires_grad(model, layer_names):
    for name, param in model.named_parameters():
        param.requires_grad = False
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True

def run(init_lr=0.1, max_steps=128e2, mode='rgb', root='data/cholec80/frame',
        train_split='data/cholec80/cholec80.json', batch_size=32, save_model='save_model'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loc_losses = []
    val_cls_losses = []
    val_tot_losses = []

    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("Number of iterations in this phase:", len(dataloader))

    val_dataset = Val_Dataset(train_split, 'validation', root, mode, test_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("Number of iterations in this phase:", len(val_dataloader))

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    i3d = I3D(400, modality='rgb', dropout_prob=0, name='inception')
    i3d.load_state_dict(torch.load('checkpoints/model_rgb.pth'))
    new_conv3d_0c_1x1 = Unit3Dpy(
        in_channels=1024,
        out_channels=7,
        kernel_size=(1, 1, 1),
        activation=None,
        use_bias=True,
        use_bn=False
    )
    i3d.conv3d_0c_1x1 = new_conv3d_0c_1x1
    unfrozen_layers = ['mixed_5b', 'mixed_5c', 'conv3d_0c_1x1']
    set_parameter_requires_grad(i3d, unfrozen_layers)
    i3d.to(device)
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, i3d.parameters()), lr=init_lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = 1  
    steps = 0

    while steps < max_steps:  # for epoch in range(num_epochs):
        print(f'Step {steps}/{max_steps}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train()
            else:
                i3d.eval()

            tot_loss = 0.0  
            tot_loc_loss = 0.0  
            tot_cls_loss = 0.0  
            num_iter = 0  
            optimizer.zero_grad()

            for data in dataloaders[phase]:
                num_iter += 1
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # print(inputs.shape)    8, 3, 64, 224, 224
                # print(labels.shape)    8, 7, 64
                per_frame_score, per_frame_logits = i3d(inputs)  # 计算每帧的logit  # 8, 7, 7
                # print(per_frame_logits.shape)

                t = inputs.size(2)  
                # per_frame_logits = F.interpolate(per_frame_logits, size=t, mode='bilinear')
                per_frame_logits = F.interpolate(per_frame_logits, size=t, mode='linear')  # 8, 7, 64
                # print(per_frame_logits.shape)

                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.item()

                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                              torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.item()

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.item()
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 400 == 0:
                            avg_train_loc_loss = tot_loc_loss / (400 * num_steps_per_update)
                            avg_train_cls_loss = tot_cls_loss / (400 * num_steps_per_update)
                            avg_train_tot_loss = tot_loss / 400
                            print(
                                f'{phase} Loc Loss: {avg_train_loc_loss:.4f}, Cls Loss: {avg_train_cls_loss:.4f}, Tot Loss: {avg_train_tot_loss:.4f}')
                            model_path = save_model + str(steps).zfill(6) + '.pth'
                            torch.save(i3d.module.state_dict(), model_path)

                            tot_loss = tot_loc_loss = tot_cls_loss = 0.
                            with open('training_validation_losses.txt', 'a') as f:
                                f.write(f'Step {steps}: Training Loc Loss: {avg_train_loc_loss:.4f}, Training Cls Loss: {avg_train_cls_loss:.4f}, Training Tot Loss: {avg_train_tot_loss:.4f}\n')
            if phase == 'val':
                avg_val_loc_loss = tot_loc_loss / num_iter
                avg_val_cls_loss = tot_cls_loss / num_iter
                avg_val_tot_loss = (tot_loss * num_steps_per_update) / num_iter
                print(
                    f'{phase} Loc Loss: {avg_val_loc_loss:.4f}, Cls Loss: {avg_val_cls_loss:.4f}, Tot Loss: {avg_val_tot_loss:.4f}')

                if steps % 400 == 0:
                    with open('train_valid_i3d_losses.txt', 'a') as f:
                        f.write(f'Step {steps}: Validation Loc Loss: {avg_val_loc_loss:.4f}, Validation Cls Loss: {avg_val_cls_loss:.4f}, Validation Tot Loss: {avg_val_tot_loss:.4f}\n')



if __name__ == '__main__':
    #run(mode=args.mode, root=args.root, save_model=args.save_model)
    run(mode='rgb', root='data/cholec80/frame', save_model='save_i3d/')


                        