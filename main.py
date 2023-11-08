from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets

from models import TSN

from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    
def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = 5e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def train(train_loader, model, criterion, optimizer, epoch, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
   
    
    # temperature
    increase = pow(1.05, epoch)
    temperature = 100 # * increase
    print (temperature)    
    

    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(True)

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # discard final batch
        if i == len(train_loader)-1:
            break
        # measure data loading time
        data_time.update(time.time() - end)

        # target size: [batch_size]
        #target = target.cuda(async=True)
        print(input.shape)
        target = target.cuda()
        input_var = input
        target_var = target

      
        output = model(input_var, temperature)
        
        loss = criterion(output, target_var)          
           
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        if i % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            if args.iter_size != 1:
                for g in optimizer.param_groups:
                    for p in g['params']:
                        p.grad /= args.iter_size

            if args.clip_gradient is not None:
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
                if total_norm > args.clip_gradient:
                    pass 
                    #print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
            else:
                total_norm = 0

            optimizer.step()
            optimizer.zero_grad()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'                   
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-2]['lr'])))
#             print(('Flow_Con_Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=flow_con_losses)))    
    
    # Write logs on Tensorboard 
    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-2]['lr'], epoch)
    
    return temperature

def validate(val_loader, model, criterion, iter, temperature, epoch, tf_writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # another losses
    flow_con_losses = AverageMeter()       
    
    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(False)
#     torch.no_grad()
    # switch to evaluate mode
    model.eval()
#     model.train()

    output_list = []
    pred_arr = []
    target_arr = []
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # discard final batch
        if i == len(val_loader)-1:
            break
        target = target.cuda()
#         target = target.cuda(async=False)
        input_var = input
        target_var = target
        # compute output
        output= model(input_var, temperature) 

#         output = model(input_var, temperature)
        #loss = criterion(output, target_var)          
        
        # class acc
        pred = torch.argmax(output.data, dim=1)
        pred_arr.extend(pred)
        target_arr.extend(target)
        

#         print ('Accuracy {:.02f}%'.format(np.mean(cls_acc)*100))
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        
        #losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))    
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        output_list.append(output)        
        
        print(('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.4f})\t'
                #'Loss {loss.val:.4f} ({loss.avg:.4f})\t'                  
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, top1=top1, top5=top5)))
#             print(('Flow_Con_Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=flow_con_losses))) 

    output_tensor = torch.cat(output_list, dim=0)

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}  Time {batch_time.avg:.4f}'
          .format(top1=top1, top5=top5, batch_time=batch_time)))  
    
    if tf_writer is not None:
        #tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
    
    return top1.avg, output_tensor , pred_arr, target_arr 

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
  
    checkpoint_folder= 'checkpoints/RAFT_TSM_something/training'
    os.makedirs(checkpoint_folder, exist_ok= True)
    epoch = str(state['epoch'])
    filename = f'RAFT_TSM_something_epoch{epoch}_checkpoint.pth.tar'
    torch.save(state, checkpoint_folder + '/' + filename) 

    if is_best:
        #best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        best_name ='model_best.pth.tar'
        shutil.copyfile(checkpoint_folder+ '/' + filename, checkpoint_folder + '/'+ best_name)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def freeze_network(model):
    for name, p in model.named_modules():
        if 'raft_motion_estimator' in name:
            p.requires_grad = False 

    return model 


def main(args):
   
    # Load Model (RAFT_TSM) 

    model = TSN(
        args.num_class,
        args.num_segments, 
        args.pretrained_parts, 
        args.modality,
        base_model = args.arch, 
        consensus_type=args.consensus_type, 
        dropout=args.dropout, 
        partial_bn=not args.no_partialbn, 
        args= args
        ) 
    
    model = freeze_network(model)
    model = model.cuda()
    #tmp_input = 255*torch.rand(8, 3, 224, 224).contiguous().cuda()
    
    #output = model(tmp_input, 100) 
    
    
    # Fetch a optimizer  
    policies = model.get_optim_policies(args.dataset) 
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    optimizer = torch.optim.SGD(policies, 
                                args.lr,
                                momentum=0.9,
                                weight_decay=5e-4, nesterov= True) 

    model = nn.DataParallel(model, device_ids=args.gpus)   
    print("Parameter Count: %d" % count_parameters(model))
    
    # resume checkpoint 
    if args.restore_ckpt is not None:
        
        checkpoint = torch.load(args.restore_ckpt)
        
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']

        model.load_state_dict(checkpoint['state_dict'])    
        print(f'checkpoint is loaded --> {args.restore_ckpt}')

        optimizer = optimizer.load_state_dict(checkpoint['optimizer']) 

    # raft pretrained is loaded 
    if args.raft_pretrained is not None :
        model.load_state_dict(torch.load(args.raft_pretrained), strict=False)
        print(f'RAFT pretrained weight is loaded --> {args.raft_pretrained}')

  
    '''
    # Do not need with the RAFT_TSM Model
    #if args.stage != 'chairs':
    #    model.module.freeze_bn()
    '''

    # Fetch 2 dataloaders 
    train_loader, validation_loader = datasets.fetch_dataloader(args)

    tf_writer= SummaryWriter(log_dir=args.log_dir)


    # define loss function and optimizer 
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")
    
    # swtich to a train mode 
    model.cuda() 
    model.train()
    
    # Evaluation 
    if args.evaluate:
        prec1, score_tensor, pred_arr, target_arr = validate(validation_loader, model, criterion, 0, temperature=100, epoch=None, tf_writer=None)
    
        return
    
    # TRAIN
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        # train for one epoch
        temperature = train(train_loader, model, criterion, optimizer, epoch, tf_writer) 

        # evaluate on validation set
        prec1, _, _, _ = validate(validation_loader, model, criterion, (epoch + 1) * len(train_loader), temperature=temperature, epoch=epoch, tf_writer= tf_writer)
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 >= best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict() 
        }, is_best)
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    #parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--lr', type=float, default=0.00004)
    #parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, nargs='+', default=[224, 224])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    
    #parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    #parser.add_argument('--clip', type=float, default=1.0)
   
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    #parser.add_argument('--add_noise', action='store_true')

    parser.add_argument('--dataset', type=str, default='somethingv2')
    parser.add_argument('--modality', type=str, default='RGB')
    parser.add_argument('--num_segments', type=int, default=8)

    parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_class', type=int, default=174)
    
    parser.add_argument('--arch', type=str, default='RAFT')
    parser.add_argument('--consensus_type', type=str, default='avg') 

    parser.add_argument('--dropout', '--do', default=0.5, type=float, metavar='DO', help='dropout ratio (default: 0.5)')
        
    parser.add_argument('--no_partialbn', '--npb', default=False , action="store_true") 

    parser.add_argument('--pretrained_parts', type=str, default='finetune', choices=['scratch','finetune'])

    parser.add_argument('--log_dir', type=str, default ='checkpoints' )
    
    parser.add_argument('--loss_type', type=str, default ='nll' )

    parser.add_argument('--raft_pretrained', type=str, default ='models/raft-kitti.pth')   
    parser.add_argument('--evaluate', default=False, action="store_true")  


    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    
    parser.add_argument('--epochs', default=45, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('-i', '--iter-size', default=1, type=int, metavar='N', help='number of iterations before on update')

    parser.add_argument('--clip-gradient', '--gd', default=1, type=float, metavar='W', help='gradient norm clipping (default: disabled)')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    main(args)
 
