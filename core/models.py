from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import xavier_uniform_, constant_

#from resnet_TSM import resnet50
from resnet_RAFT import resnet50

# pytorch 0.3.1
# from torch.nn.init import normal, constant

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, pretrained_parts, modality,
                 base_model='RAFT', dataset='something', new_length=1,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,fc_lr5=True,
                 crop_num=1, partial_bn=True, patch_size = None, tsm_mode = None, args=None) :
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.pretrained_parts = pretrained_parts
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.base_model_name = base_model
        self.dataset = dataset
        self.fc_lr5 = fc_lr5        
        self.new_length = new_length
        self.consensus = ConsensusModule(consensus_type)

    
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        print(("""
               Initializing TSN with base model: {}
                TSN Configurations:
                input_modality:     {}
                num_segments:       {}
                new_length:         {}
                consensus_module:   {}
                dropout_ratio:      {}
                    """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))
      
        if (base_model == 'RAFT') : 
            
            self.base_model = resnet50(True, shift='TSM', num_segments = num_segments, flow_estimation = 1, args= args)
            self.base_model.last_layer_name = 'fc1'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]          
            feature_dim = self._prepare_tsn(num_class)   

        
        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)
    
    def _prepare_tsn(self, num_class):    
        
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_channels
        #feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        
        if self.dropout == 0:
            #setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            setattr(self.base_model, self.base_model.last_layer_name, nn.Conv1d(feature_dim, num_class, kernel_size=1, stride=1, padding=0,bias=True))  
            self.new_fc = None
        
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            # self.new_fc = nn.Linear(feature_dim, num_class)
            self.new_fc = nn.Conv1d(feature_dim, num_class, kernel_size=1, stride=1, padding=0,bias=True)

        std = 0.001
            
        # pytorch 0.4.1            
        if self.new_fc is None:
            xavier_uniform_(getattr(self.base_model, self.base_model.last_layer_name).weight)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        
        else:
            xavier_uniform_(self.new_fc.weight)
            constant_(self.new_fc.bias, 0)   
            
        return feature_dim

    
    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():                
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        else:
            print("No BN layer Freezing.")

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self, dataset):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult':  5 if self.dataset == 'kinetics' else 1, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult':  10 if self.dataset == 'kinetics' else 2, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]
        

    def get_optim_policies_BN2to1D(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        last_conv_weight = []
        last_conv_bias = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            # (conv1d or conv2d) 1st layer's params will be append to list: first_conv_weight & first_conv_bias, total num 1 respectively(1 conv2d)
            # (conv1d or conv2d or Linear) from 2nd layers' params will be append to list: normal_weight & normal_bias, total num 69 respectively(68 Conv2d + 1 Linear)
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                last_conv_weight.append(ps[0])
                if len(ps) == 2:
                    last_conv_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            # (BatchNorm1d or BatchNorm2d) params will be append to list: bn, total num 2 (enabled pbn, so only: 1st BN layer's weight + 1st BN layer's bias)
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # 4
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
             {'params': last_conv_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "last_conv_weight"},
            {'params': last_conv_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "last_conv_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input, temperature):

        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        # input.size(): [32, 9, 224, 224]
        # after view() func: [96, 3, 224, 224]

        
        input_var = input.view((-1, sample_len) + input.size()[-2:])
            
        # self.base_model : renet50 
        base_out = self.base_model(input_var, temperature) # RAFT version. 
        # zc comments
        if self.dropout > 0:
           
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        # zc comments end
        
        if self.reshape:
            if "flow" in self.base_model_name:
                base_out = base_out.view((-1, (self.num_segments)) + base_out.size()[1:])
            else:
                base_out = base_out.view((-1, (self.num_segments)) + base_out.size()[1:])                       
            output = self.consensus(base_out)
            return output.squeeze(3).squeeze(1)


    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data



    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),GroupRandomHorizontalFlip(selective_flip=True, is_flow=False)])
#             return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875]),
#                                                    GroupRandomHorizontalFlip(is_flow=False)])        
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])