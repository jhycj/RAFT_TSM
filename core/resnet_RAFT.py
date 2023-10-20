"""
An example combining `Temporal Shift Module` with `ResNet`. This implementation
is based on `Temporal Segment Networks`, which merges temporal dimension into
batch, i.e. inputs [N*T, C, H, W]. Here we show the case with residual connections
and zero padding with 8 frames as input.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch as tr
from tsm_util import tsm
from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8


#__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']

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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,num_segments, stride=1, downsample=None, remainder=0):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.remainder= remainder        
        self.num_segments = num_segments        
    

    def forward(self, x):
        identity = x
        out = tsm(x, self.num_segments)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class RAFT(nn.Module):
    def __init__(self, args):
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

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        
        # get 2 input images 
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
     
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])      


        print(f'feature map1 shape: {fmap1.shape}') #[1, 256, 55, 128]
        print(f'feature map2 shape: {fmap2.shape}') #[1, 256, 55, 128]

        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions


class RAFT_MultipleFrames(nn.Module):
    def __init__(self, args):
        super(RAFT_MultipleFrames, self).__init__()
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

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    # def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
    def forward(self, input_tensor, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between multiple(more than 2) frames """
        
        # get 2 input images 
        #image1 = 2 * (image1 / 255.0) - 1.0
        #image2 = 2 * (image2 / 255.0) - 1.0
     
        # get multple input images ==> input tensor 
        #input_tensor = input_tensor.contiguous 
        print('input_tensor')
        print(input_tensor.shape) # torch.Size([16, 3, 224, 224] 

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            #fmap1, fmap2 = self.fnet([image1, image2])      
            fmap = self.fnet(input_tensor) 

        #print(f'feature map1 shape: {fmap1.shape}') #[1, 256, 55, 128]
        #print(f'feature map2 shape: {fmap2.shape}') #[1, 256, 55, 128]
        
        print(f'feature map shape: {fmap.shape}') # shape: torch.Size([16, 256, 28, 28]) 

        size = fmap.shape
        _, c, h, w = size

        fmap = fmap.view((-1, self.args.num_segments) + size[1:])

        fmap = fmap.permute(0,2,1,3,4) 
        print(f're shaped: {fmap.shape}') # re shpaed: torch.Size([2, 256, 8, 28, 28]) # B, C,T,H,W

        '''
        x_pre = x[:,:,:-1].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
        x_post = x[:,:,1:].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
        '''

        fmap_pre = fmap[:, :, :-1].permute(0,2,1,3,4).contiguous().view(-1,c,h,w) #[b*(t-1), C, H, W] 
        fmap_post = fmap[:,:,1:].permute(0,2,1,3,4).contiguous().view(-1,c,h,w) #[b*(t-1), C, H, W] 

        print(f'fmap_pre: {fmap_pre.shape}')
        print(f'fmap_post: {fmap_post.shape}')

        fmap_pre = fmap_pre.float()
        fmap_post = fmap_post.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap_pre, fmap_post, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap_pre, fmap_post, radius=self.args.corr_radius) # return a instance 

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions



class ResNet(nn.Module):

    def __init__(self, block, block2, layers, num_segments, flow_estimation, num_classes=1000, zero_init_residual=False, args=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()          
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax = nn.Softmax(dim=1)        
        self.num_segments = num_segments     
        self.flow_estimation = flow_estimation
        self.patch = 15 

        ## MotionSqueeze
        '''
        if flow_estimation:
            #self.patch= 15 # 7*2 + 1 
            #self.patch = 29 # 14*2 + 1   
            self.patch_dilation =1
            self.matching_layer = Matching_layer_scs(ks=1, patch=self.patch, stride=1, pad=0, patch_dilation=self.patch_dilation)
#             self.matching_layer = Matching_layer_mm(patch=self.patch)
            
            self.flow_refinement = Flow_refinement(num_segments=num_segments, expansion=block.expansion,pos=2)      
            self.soft_argmax = nn.Softmax(dim=1)
        
            self.chnl_reduction = nn.Sequential(
                nn.Conv2d(128*block.expansion, 64, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        '''
        ## RAFT 
        if flow_estimation: 
            self.raft_motion_estimator = RAFT_MultipleFrames(args) 
       
        self.layer1 = self._make_layer(block, 64, layers[0], num_segments=num_segments)
        self.layer2 = self._make_layer(block, 128, layers[1],  num_segments=num_segments, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],  num_segments=num_segments, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],  num_segments=num_segments, stride=2)       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc1 = nn.Linear(512 * block.expansion, num_classes)                   
        self.fc1 = nn.Conv1d(512*block.expansion, num_classes, kernel_size=1, stride=1, padding=0,bias=True)         
        
        for m in self.modules():       
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x / norm) 

    
    def apply_binary_kernel(self, match, h, w, region):
        # binary kernel
        x_line = tr.arange(w, dtype=tr.float).to('cuda').detach()
        y_line = tr.arange(h, dtype=tr.float).to('cuda').detach()
        x_kernel_1 = x_line.view(1,1,1,1,w).expand(1,1,w,h,w).to('cuda').detach()
        y_kernel_1 = y_line.view(1,1,1,h,1).expand(1,h,1,h,w).to('cuda').detach()
        x_kernel_2 = x_line.view(1,1,w,1,1).expand(1,1,w,h,w).to('cuda').detach()
        y_kernel_2 = y_line.view(1,h,1,1,1).expand(1,h,1,h,w).to('cuda').detach()

        ones = tr.ones(1).to('cuda').detach()
        zeros = tr.zeros(1).to('cuda').detach()

        eps = 1e-6
        kx = tr.where(tr.abs(x_kernel_1 - x_kernel_2)<=region, ones, zeros).to('cuda').detach()
        ky = tr.where(tr.abs(y_kernel_1 - y_kernel_2)<=region, ones, zeros).to('cuda').detach()
        kernel = kx * ky + eps
        kernel = kernel.view(1,h*w,h*w).to('cuda').detach()                
        return match* kernel


    def apply_gaussian_kernel(self, corr, h,w,p, sigma=5):
        b, c, s = corr.size()

        x = tr.arange(p, dtype=tr.float).to('cuda').detach()
        y = tr.arange(p, dtype=tr.float).to('cuda').detach()

        idx = corr.max(dim=1)[1] # b x hw    get maximum value along channel
        idx_y = (idx // p).view(b, 1, 1, h, w).float()
        idx_x = (idx % p).view(b, 1, 1, h, w).float()

        x = x.view(1,1,p,1,1).expand(1, 1, p, h, w).to('cuda').detach()
        y = y.view(1,p,1,1,1).expand(1, p, 1, h, w).to('cuda').detach()

        gauss_kernel = tr.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * sigma**2))
        gauss_kernel = gauss_kernel.view(b, p*p, h*w)#.permute(0,2,1).contiguous()

        return gauss_kernel * corr

    def match_to_flow_soft(self, match, k, h,w, temperature=1, mode='softmax'):        
        b, c , s = match.size()     
        idx = tr.arange(h*w, dtype=tr.float32).to('cuda')
        idx_x = idx % w
        idx_x = idx_x.repeat(b,k,1).to('cuda')
        idx_y = tr.floor(idx / w)   
        idx_y = idx_y.repeat(b,k,1).to('cuda')

        soft_idx_x = idx_x[:,:1]
        soft_idx_y = idx_y[:,:1]
        displacement = (self.patch-1)/2
        
        topk_value, topk_idx = tr.topk(match, k, dim=1)    # (B*T-1, k, H*W)
        topk_value = topk_value.view(-1,k,h,w)
        
        match = self.apply_gaussian_kernel(match, h, w, self.patch, sigma=5)
        match = match*temperature
        match_pre = self.soft_argmax(match)
        smax = match_pre           
        smax = smax.view(b,self.patch,self.patch,h,w)
        x_kernel = tr.arange(-displacement*self.patch_dilation, displacement*self.patch_dilation+1, step=self.patch_dilation, dtype=tr.float).to('cuda')
        y_kernel = tr.arange(-displacement*self.patch_dilation, displacement*self.patch_dilation+1, step=self.patch_dilation, dtype=tr.float).to('cuda')
        x_mult = x_kernel.expand(b,self.patch).view(b,self.patch,1,1)
        y_mult = y_kernel.expand(b,self.patch).view(b,self.patch,1,1)
            
        smax_x = smax.sum(dim=1, keepdim=False) #(b,w=k,h,w)
        smax_y = smax.sum(dim=2, keepdim=False) #(b,h=k,h,w)
        flow_x = (smax_x*x_mult).sum(dim=1, keepdim=True).view(-1,1,h*w) # (b,1,h,w)
        flow_y = (smax_y*y_mult).sum(dim=1, keepdim=True).view(-1,1,h*w) # (b,1,h,w)    

        flow_x = (flow_x / (self.patch_dilation * displacement))
        flow_y = (flow_y / (self.patch_dilation * displacement))
            
        return flow_x, flow_y, topk_value          
        
    def _make_layer(self, block, planes, blocks, num_segments, stride=1):   

        # (for example) self.layer1 = self._make_layer(block, 64, layers[0], num_segments=num_segments)

        downsample = None        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        layers = []
        layers.append(block(self.inplanes, planes, num_segments, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            remainder =int( i % 3)
            layers.append(block(self.inplanes, planes, num_segments, remainder=remainder))
            
        return nn.Sequential(*layers)            
    
    def flow_computation(self, x, pos=2, temperature=100):
        
        x = self.chnl_reduction(x)
        
        size = x.size()               
        x = x.view((-1, self.num_segments) + size[1:])        # N T C H W
        x = x.permute(0,2,1,3,4).contiguous() # B C T H W   
                        
        # match to flow            
        k = 1            
        temperature = temperature                    
        b,c,t,h,w = x.size()            
        t = t-1         

        x_pre = x[:,:,:-1].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
        x_post = x[:,:,1:].permute(0,2,1,3,4).contiguous().view(-1,c,h,w)
            
        match = self.matching_layer(x_pre, x_post)    # (B*T-1*group, H*W, H*W)   
        
        #print(f'match_shape')
        #print(match.shape) 
        #print('------')       
        u, v, confidence = self.match_to_flow_soft(match, k, h, w, temperature)
        flow = tr.cat([u,v], dim=1).view(-1, 2*k, h, w)  #  (b, 2, h, w)            
        #print(f'flow_shape') 
        #print(flow.shape) 
        #print('-----')
                        
        # backward flow
#             match2 = self.matching_layer(x_post, x_pre)
#             u_2, v_2, confidence_2 = self.match_to_flow_soft(match2, k, h, w,temperature)       
#             flow_2 = tr.cat([u_2,v_2],dim=1).view(-1,2, h, w)   
    
        return flow, confidence     
        
    def forward(self, x, temperature):
        
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
       
        x = self.layer1(x)                                                 
        x = self.layer2(x)

        # MotionSqueeze Estimation 
        if (self.flow_estimation == 1):  
            flow_1, match_v = self.flow_computation(x, temperature=temperature)
            x = self.flow_refinement(flow_1,x, match_v)
        '''
        
        # RAFT Flow Estimation 
        if (self.flow_estimation == 1): 

            # run the raft motion estimator 
            self.raft_motion_estimator(x)


        x = self.layer3(x)                                    
        x = self.layer4(x)
        x = self.avgpool(x)    
        x = x.view(x.size(0), -1,1)    
                    
        x = self.fc1(x)      
        return x


def resnet50(pretrained=False, shift='TSM',num_segments = 8, flow_estimation=0, args=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, Bottleneck, [3, 4, 6, 3],num_segments=num_segments , flow_estimation=flow_estimation, args= args, **kwargs)          
    if pretrained:
        #pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        import torch 
        pretrained_dict = torch.load('pretrained/resnet50-19c8e357.pth')
        new_state_dict =  model.state_dict()
    
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
        
    return model


