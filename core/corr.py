import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape # batch -> b*T 
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)

        for i in range(self.num_levels-1): # Pyramid 

            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):

        r = self.radius
        coords = coords.permute(0, 2, 3, 1) 
        batch, h1, w1, _ = coords.shape  
    
        out_pyramid = []


        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            #print(f'{i}th pyramid corr: {corr.shape}') 

            #print(f'corr inspective: {corr[365, 0].shape}')
            # print(corr[165, 0]) 
            
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1) 

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i 
            #print(f'centroid_lvl: {centroid_lvl.shape}') # [7*28*28, 1, 1, 2] 
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2) 
            #print(f'delta: {delta_lvl.shape}')             
            #print(f'delta_lvl: {delta_lvl.shape}') # delta_lvl: torch.Size([1, 9, 9, 2]) 
            coords_lvl = centroid_lvl + delta_lvl
            #print(f'coords_lvl: {coords_lvl.shape}') # [p, 9, 9, 2]
        
            #print(f'corr: {corr.shape}') # corr: torch.Size([8548, 1, 28, 28]) ->([5488, 1, 14, 14]) -> [5488, 1, 7, 7]) ->([5488, 1, 3, 3]) 
            
            corr = bilinear_sampler(corr, coords_lvl) # [p, 1, 9, 9]  , [p, 9, 9, 2]
            #print(f'after bilinear corr shape:{corr.shape}') # ([p, 1, 9, 9]) (All level has same corr shape)
            
            corr = corr.view(batch, h1, w1, -1)
            #print(f'after corr shape:{corr.shape}') # ([7, 28, 28, 81]) (All level has same corr shape)
            
            out_pyramid.append(corr)
 
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        # input : 2 feature  maps  of image1 and image2 
        batch, dim, ht, wd = fmap1.shape # [b*(t-1), C, H, W] 
        
        fmap1 = fmap1.view(batch, dim, ht*wd)   
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
