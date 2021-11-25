import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import scipy.io as scio
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
import h5py
import re
from test import test


from math import exp
from data_loader import ICDAR2015, Synth80k, ICDAR2013, VietSynth, VietSB

###import file#######
from mseloss import Maploss



from collections import OrderedDict
from eval.script import getresult



from PIL import Image
from torchvision.transforms import transforms
from craft import CRAFT
from torch.autograd import Variable
from multiprocessing import Pool
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
#3.2768e-5
random.seed(42)

parser = argparse.ArgumentParser(description='CRAFT reimplementation')

##### Checkpoint #####
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')


##### Config #####
parser.add_argument('--batch_size', default=128, type = int,
                    help='batch size of training')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')


##### Test #####
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--mag_ratio', default=2., type=float, help='image magnification ratio')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')

##### Data #####
parser.add_argument('--synth_data', default="Synthtext", type=str,
                        choices=['Synthtext','VietST'], help='Synthesis data')
parser.add_argument('--synth_path',type=str,  help='Synthesis data path')
parser.add_argument('--real_data', default="ICDAR2015", type=str,
                        choices=['ICDAR2013','ICDAR2015','VietSB'], help='Real data')
parser.add_argument('--real_path',type=str,  help='Real data path')

##### Output ####
parser.add_argument('--out_folder',default='./checkpoints/',type=str,
                    help='Ouput folder checkpoints')
args = parser.parse_args()


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def step(
    net,
    syn_images, syn_gh_label, syn_gah_label, syn_mask,
    real_images, real_gh_label, real_gah_label, real_mask,
    index,
    len_data_loader,
    out_folder,
    st,
    loss_value
    ):
    
    images = torch.cat((syn_images,real_images), 0)
    gh_label = torch.cat((syn_gh_label, real_gh_label), 0)
    gah_label = torch.cat((syn_gah_label, real_gah_label), 0)
    mask = torch.cat((syn_mask, real_mask), 0)

    images = Variable(images.type(torch.FloatTensor)).cuda()
    gh_label = gh_label.type(torch.FloatTensor)
    gah_label = gah_label.type(torch.FloatTensor)
    gh_label = Variable(gh_label).cuda()
    gah_label = Variable(gah_label).cuda()
    mask = mask.type(torch.FloatTensor)
    mask = Variable(mask).cuda()

    out, _ = net(images)

    optimizer.zero_grad()

    out1 = out[:, :, :, 0].cuda()
    out2 = out[:, :, :, 1].cuda()
    loss = criterion(gh_label, gah_label, out1, out2, mask)

    loss.backward()
    optimizer.step()
    loss_value[0] += loss.item()
    if index % 2 == 0 and index > 0:
        et = time.time()
        print('epoch {}:({}/{}) batch || training time for 2 batch {} || training loss {} ||'.format(epoch, index, len_data_loader, round(et-st[0],4), round(loss_value[0]/2,4)))
        loss_time = 0
        loss_value[0] = 0
        st[0] = time.time()
    # if loss < compare_loss:
    #     print('save the lower loss iter, loss:',loss)
    #     compare_loss = loss
    #     torch.save(net.module.state_dict(),
    #                '/data/CRAFT-pytorch/real_weights/lower_loss.pth'

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    if args.synth_data == 'VietST':
        dataloader = VietSynth(args.synth_path,target_size=768, viz=False, debug=True)
    elif args.synth_data == 'Synthtext':
        dataloader = Synth80k(args.synth_path,target_size=768, viz=False, debug=True)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
    batch_syn = iter(train_loader)
    net = CRAFT()
    if isinstance(args.resume,str) and args.resume.endswith('.pth'):
        net.load_state_dict(copyStateDict(torch.load(args.resume)))

    # realdata = realdata(net)
    if args.real_data == 'VietSB':
        realdata = VietSB(net, args.real_path, target_size=768)
    elif args.real_data == 'ICDAR2015':
        realdata = ICDAR2015(net, args.real_path, target_size=768)
    elif args.real_data == 'ICDAR2013':
        realdata = ICDAR2013(net, args.real_path, target_size=768)
    real_data_loader = torch.utils.data.DataLoader(
        realdata,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
    net = net.cuda()
    #net = CRAFT_net

    if args.cuda:
        net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Maploss()
    #criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    net.train()


    step_index = 0
    os.makedirs(args.out_folder,exist_ok = True)

    loss_time = 0
    loss_value = 0
    compare_loss = 1
    for epoch in range(10):
        loss_value = [0]
        # if epoch % 50 == 0 and epoch != 0:
        #     step_index += 1
        #     adjust_learning_rate(optimizer, args.gamma, step_index)

        st = [time.time()]
        for index, (real_images, real_gh_label, real_gah_label, real_mask, _) in enumerate(real_data_loader):
            syn_images, syn_gh_label, syn_gah_label, syn_mask, __ = next(batch_syn)
            step(
                net,
                syn_images, syn_gh_label, syn_gah_label, syn_mask,
                real_images, real_gh_label, real_gah_label, real_mask,
                index,
                len(real_data_loader),
                args.out_folder,
                st,
                loss_value
            )
            if index % 1000 == 0:
                net.eval()
                print('Saving state, index:', index)
                out_path = os.path.join(args.out_folder,
                    'synweights_epoch_{}_iter_{}_.pth'.format(epoch,repr(index))
                    )
                torch.save(net.module.state_dict(),
                            out_path)
                test(net,args)
                #test('/data/CRAFT-pytorch/craft_mlt_25k.pth')
                if index != 0:
                    getresult()
                net.train()
        final_iter_path = os.path.join(
            args.out_folder,
            'synweights_epoch_{}_iter_final_.pth'.format(epoch)
        )
        torch.save(net.module.state_dict(),
            final_iter_path)








