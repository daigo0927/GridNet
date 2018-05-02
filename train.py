import sys, os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from model import GridNet

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *

import pdb

def train(args):

    data_aug = Compose([RandomRotate(10),
                        RandomHorizontallyFlip()])

    # Setup dataloader
    data_loader = get_loader(args.dataset)
    # data_path = get_data_path(args.dataset)
    data_path = args.dataset_dir
    t_loader = data_loader(data_path, is_transform = True,
                           img_size = (args.img_rows, args.img_cols),
                           augmentations = data_aug, img_norm = args.img_norm)
    v_loader = data_loader(data_path, is_transform = True,
                           split = 'validation', img_size = (args.img_rows, args.img_cols),
                           img_norm = args.img_norm)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size = args.batch_size,
                                  num_workers = args.num_workers, shuffle = True)
    valloader = data.DataLoader(v_loader, batch_size = args.batch_size,
                                num_workers = args.num_workers)

    # Setup metrics
    running_metrics = runningScore(n_classes)

    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()
        loss_window = vis.line(X = torch.zeros((1,)).cpu(),
                               Y = torch.zeros((1)).cpu(),
                               opts = dict(xlabel = 'minibatches',
                                           ylabel = 'Loss',
                                           title = 'Training loss',
                                           legend = ['Loss']))

    # Setup model
    model = GridNet(in_chs = 3, out_chs = n_classes)
    # model = torch.nn.DataParallel(model,
    #                               device_ids = range(torch.cuda.device_count()))
    if torch.cuda.is_available():
        model.to(args.device)
        
    if hasattr(model.modules, 'optimizer'):
        optimizer = model.modules.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = args.l_rate,
                                    momentum = 0.99, weight_decay = 5e-4)

    if hasattr(model.modules, 'loss'):
        print('Using custom loss')
        loss_fn = model.modules.loss
    else:
        loss_fn = cross_entropy2d

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f'Loading model and optimizer from checkpoint {args.resume}')
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"Loaded checkpoint {args.resume}, epoch {checkpoint['epoch']}")
        else:
            print(f'No checkpoint found at {args.resume}')

    best_iou = -100.
    for epoch in range(args.n_epoch):
        
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            # if torch.cuda.is_available():
            #     images = Variable(images.cuda())
            #     labels = Variable(labels.cuda())
            images = images.to(args.device)
            # labels = labels.type(torch.FloatTensor).to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)

            # pdb.set_trace()

            loss = loss_fn(input = outputs, target = labels)

            loss.backward()
            optimizer.step()

            if args.visdom:
                vis.line(X = torch.ones((1, 1)).cpu() * i,
                         Y = orch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                         win = loss_window,
                         update = 'append')

            if (i+1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{args.n_epoch}] Loss: {loss.data[0]}")

            model.eval()
            for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                images_val = images_val.to(args.device)
                labels_val = labels_val.to(args.device)

                outputs = model(images_val)
                pred = outputs.data.max(1)[1].cpu().numpy()
                gt = labels_val.data.cpu().numpy()
                running_metrics.update(gt, pred)

                score, class_iou = running_metrics.get_score()
                for k, v in score.item():
                    print(k, v)
                running_metrics.reset()

                if score['Mean IoU : \t'] >= best_iou:
                    best_iou = score['Mean IoU : \t']
                    state = {'epoch': epoch+1,
                             'model_state': model.state_dict(),
                             'optimizer_state': optimizer.state_dict()}
                    torch.save(state, f'{args.dataset}_best_model.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='mit_sceneparsing_benchmark',
                        help = 'Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--dataset_dir', required = True, type = str,
                        help = 'Directory containing target dataset')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help = 'Height of the input')
    parser.add_argument('-img_cols', nargs='?', type=int, default=256,
                        help = 'Width of input')

    parser.add_argument('--device', type = str, default = 'cuda',
                        help = 'Utilize device cuda (default) or cpu')
    parser.add_argument('--num_workers', type = int, default = 1,
                        help = '# of worker used for data loading')

    parser.add_argument('--img_norm', dest = 'img_norm', action = 'store_true',
                        help = 'Enable input images scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest = 'img_norm', action = 'store_false',
                        help = 'Disable input images scales normalization [0, 1] | True by Default')
    parser.set_defaults(img_norm = True)
    
    parser.add_argument('--n_epoch', nargs = '?', type = int, default = 100,
                        help = '# of epochs')
    parser.add_argument('--batch_size', nargs = '?', type = int, default = 8,
                        help = 'Batch size')
    parser.add_argument('--l_rate', nargs = '?', type = float, default = 1e-5,
                        help = 'Learning rate [1-e5]')
    parser.add_argument('--resume', nargs = '?', type = str, default = None,
                        help = 'Path to previous saved model to restart from')

    parser.add_argument('--visdom', dest = 'visdom', action = 'store_true',
                        help = 'Enable visualizaion(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest = 'visdom', action = 'store_false',
                        help = 'Disable visualization(s) in visdom | False by default')
    parser.set_defaults(visdom = False)

    args = parser.parse_args()
    train(args)

























                    
