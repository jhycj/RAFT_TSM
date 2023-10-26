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
import torch 


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
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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
        print('RESNET50 Freeze_bn is applied!!!!!')
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
    def forward(self, input_tensor, iters=12, flow_init=None, upsample=False, test_mode=False):
        """ Estimate optical flow between multiple(more than 2) frames """
        
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            #fmap1, fmap2 = self.fnet([image1, image2])      
            fmap = self.fnet(input_tensor) 
        

        size = fmap.shape
        _, c, h, w = size # B*T, C, H, W 

        fmap = fmap.view((-1, self.args.num_segments) + size[1:]) # [2, 8, 256, 28, 28] 
        fmap = fmap.permute(0,2,1,3,4) # [2, 256, 8, 28, 28] 


        fmap_pre = fmap[:, :, :-1].permute(0,2,1,3,4).contiguous().view(-1,c,h,w) #[b*(t-1), C, H, W] 
        fmap_post = fmap[:,:,1:].permute(0,2,1,3,4).contiguous().view(-1,c,h,w) #[b*(t-1), C, H, W] 


        fmap_pre = fmap_pre.float()
        fmap_post = fmap_post.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap_pre, fmap_post, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap_pre, fmap_post, radius=self.args.corr_radius) # return a instance 


        # run the context network (Multiple Frame version) 
        with autocast(enabled=self.args.mixed_precision):
            
            input_size = input_tensor.shape # torch.Size([8, 3, 224, 224]   
            
            cnet_input = input_tensor.view((-1, self.args.num_segments) + input_size[1:]) # # torch.Size([1, 8, 3, 224, 224])  
            cnet_input = cnet_input[:, :-1]
            b, t, c, h, w = cnet_input.shape   
            
            cnet_input = cnet_input.contiguous().view(-1, c, h, w) 
        
            cnet = self.cnet(cnet_input)

            net, inp = torch.split(cnet, [hdim, cdim], dim=1) # [hdim, cdim] = [128, 128]
            
            net = torch.tanh(net) # torch.Size([7, 128, 28, 28]) 
            inp = torch.relu(inp) # torch.Size([7, 128, 28, 28]) 
        
        # input tensor shape:  torch.Size([16, 3, 224, 224] 
        # sample_image1 = torch.randn([7, 3, 224, 224])   # sample_image1 shape must be [14, 3, 224, 224] if input tensor's shape is [16, 3, 224, 224]  
                
        coords0, coords1 = self.initialize_flow(cnet_input)  # # torch.Size([7, 2, 28, 28])  # # torch.Size([7, 2, 28, 28]): 

        
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        
        for itr in range(iters):
            #coords1 = coords1.detach()
        
            corr = corr_fn(coords1) # index correlation volume :# corr shape: torch.Size([7, 324, 28, 28])  
    
            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                net = net.cuda()
                inp = inp.cuda()
                corr = corr.cuda()
                flow = flow.cuda() 

                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            
            coords0 = coords0.cuda()
            coords1 = coords1.cuda() 
            delta_flow = delta_flow.cuda()

            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                #print((coords1-coords0).shape) # torch.Size([3, 2, 28, 28]) 
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                
                # Do not apply upsaple_flow 
            
            #flow_predictions.append(flow_up)
            flow_predictions.append(coords1-coords0) 

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions, fmap 


class RAFT_Flow_refinement(nn.Module):
    def __init__(self, num_segments, expansion = 1, pos=2):
        super(RAFT_Flow_refinement, self).__init__()
        self.num_segments = num_segments
        self.expansion = 1
        self.pos = pos
        self.out_channel = 64*(2**(self.pos-1))*self.expansion * 2
        #print(f'self.out_channel: {self.out_channel}') # 256 

        self.c1 = 16
        self.c2 = 32
        self.c3 = 64

        
        self.conv1 = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=3, groups=3, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(3, self.c1, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.c1),
        nn.ReLU()
        )
        self.conv2 = nn.Sequential(
        nn.Conv2d(self.c1, self.c1, kernel_size=3, stride=1, padding=1, groups=self.c1, bias=False),
        nn.BatchNorm2d(self.c1),
        nn.ReLU(),
        nn.Conv2d(self.c1, self.c2, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.c2),
        nn.ReLU()
        )
        self.conv3 = nn.Sequential(
        nn.Conv2d(self.c2, self.c2, kernel_size=3, stride=1, padding=1, groups=self.c2, bias=False),
        nn.BatchNorm2d(self.c2),
        nn.ReLU(),
        nn.Conv2d(self.c2, self.c3, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.c3),
        nn.ReLU()
        )
        self.conv4 = nn.Sequential(
        nn.Conv2d(self.c3, self.c3, kernel_size=3, stride=1, padding=1, groups=self.c3, bias=False),
        nn.BatchNorm2d(self.c3),
        nn.ReLU(),
        nn.Conv2d(self.c3, self.out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(self.out_channel),
        nn.ReLU()
        )

        self.relu = nn.ReLU(inplace=True)
    
    def get_match_v(self, x) : 
        
        B, c, h, w = x.shape
        match_v = tr.ones([B,1, h,w]).cuda()

        return match_v

    def forward(self, x, res, match_v = None): # x<-flow (u,v), res<-x, match_v : max_value에 해당  
        
        x = x[-1] # use only last prediction 
        x = x.cuda() 

        match_v = self.get_match_v(x) 
        
        if match_v is not None:
            x = tr.cat([x, match_v], dim=1) 
        
        _, c, h, w = x.size() # torch.Size([b*(t-1), 2, 224, 224])   # ([7, 3, 224, 224]) 

        x = x.view(-1, self.num_segments-1, c,h,w)# (b, t-1, c, h, w) #([1, 7, 3, 224, 224]) 
        
        x = tr.cat([x,x[:,-1:,:,:,:]], dim=1) ## (b,t,3,h,w) <----why???????
        
        x = x.view(-1,c,h,w) # torch.Size([8, 3, 224, 224]) 

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x) 
        
        res = res.permute(0,2,1,3,4)
        b, t, c, h, w= res.shape
        res = res.view(-1,c,h,w) 
        
        x = x + res 

        return x


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
        self.flow_refinement = RAFT_Flow_refinement(num_segments= self.num_segments, expansion= block.expansion, pos=2)

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
    
   
        
    def forward(self, x, temperature):
        
        # RAFT Flow Estimation 
        if (self.flow_estimation == 1): 
            # run the raft motion estimator 
            flow_predictions, fmap = self.raft_motion_estimator(x)
            res = fmap  # res must be [4, 512, 28, 28]) 
            x = self.flow_refinement(flow_predictions, res) 

        x = self.layer2(x) 
        x = self.layer3(x)                                    
        x = self.layer4(x)
        x = self.avgpool(x)    
        x = x.view(x.size(0), -1,1)    
                    
        x = self.fc1(x)      
        return x


def resnet50(pretrained=True, shift='TSM',num_segments = 8, flow_estimation=0, args=None, **kwargs):
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


