
import argparse
import sys
sys.path.append('core')
from models import TSN 

def freeze_network(model):
    for name, p in model.named_modules():
        if 'raft_motion_estimator' in name:
            p.requires_grad = False 
            
    return model 


if __name__ == "__main__": 
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

    parser.add_argument('--clip-gradient', '--gd', default=200, type=float, metavar='W', help='gradient norm clipping (default: disabled)')
    args = parser.parse_args()


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
    '''

    for name, p in model.named_modules(): 
        if 'raft' in name:
            print(p.requires_grad)

    '''
