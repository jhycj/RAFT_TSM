import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from update import BasicUpdateBlock, SmallUpdateBlock 
from extractor import BasicEncoder, SmallEncoder 
from corr import CorrBlock, AlternateCorrBlock 

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class RAFT(nn.Module): 
    def __init__(self,args): 
        super(RAFT, self).__init__() 
        self.args = args 

        if args.small:
            self.hidden_dim = hdim = 96 
            self.context_dim = cdim = 64 

            args.corr_levels = 4 
            args.corr_radius = 3

        else: 
            self.hidden_dim = hdim = 128 
            self.context_dim = cdim = 128 

            args.corr_levels = 4 
            args.corr_radius = 4 

        
        if 'dropout' not in self.args: 
            self.args.dropout = 0 

        if 'alternate_corr' not in self.args : 
            self.args.alternate_corr = False 

        # Feature Network, Context Network, and Update Block 

        if args.small : 
            
            self.fnet = SmallEncoder() 
            self.cnet = SmallEncoder() 
            self.update_block = SmallUpdateBlock() 

        else: 

            self.fnet = BasicEncoder() 
            self.cnet = BasicEncoder() 
            self.update_block = BasicUpdateBlock()  

    
    def forward(self,image1, image2, iters= 12, flow_init=None, upsample=True, test_mode = False) : 

        # Estimate Optical flow between pair of frames 

        image1 = 2*(image1 / 255.0) - 1.0 
        image2 = 2*(image2 / 255.0) - 1.0

        image1 = image1.contiguous() 
        image2 = image2.contiguous() 

        hdim = self.hidden_dim 
        cdim = self.context_dim 


        # run the feature network 

        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2]) 

        fmap1 = fmap1.float() 
        fmap2 = fmap2.float() 
        
        # pass the AlternateCorrBlock  
        corr_fn = CorrBlock(fmap1, fmap2, radius = self.args.corr_radius) 
        
         















