# Authors: Arthur Wang
# Date: 2025-4-20
# Description: 
#   This script loads dataset from all locations, checking values automatically. Only Requires max value found in video samples (I.e. output_X)
#

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# import torch_tensorrt

import math
import time, gc
# from torch.nn.modules.utils import _triple
# from torch.profiler import profile, record_function, ProfilerActivity



class CrossViewAttention(nn.Module):
    def __init__(self, channels, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        #self.B = fm_shape[0]
        #self.C = fm_shape[1]
        #self.H = fm_shape[2]
        #self.W = fm_shape[3]
        #self.area = self.H * self.W
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout)
		    
        self.lin = nn.Sequential(nn.Linear(channels, ff_dim), nn.GELU(), nn.Linear(ff_dim, channels), nn.Dropout(dropout))
        self.pos_emb = nn.Parameter(torch.randn(1, 20*20, channels))
		
        self.lnorm1 = nn.LayerNorm(channels)
        self.lnorm2 = nn.LayerNorm(channels)
		
    def forward(self, pov1, pov2):
		
        B, C, H, W = pov1.shape
        
        #print(pov1.shape)
        Area = H * W
		
        #print(pov1.view(-1, C, Area).shape, self.pos_emb.shape)
        p1 = pov1.view(-1, C, Area).permute(2, 0, 1) + self.pos_emb.permute(1, 0, 2)         # B x HW x 256
        p2 = pov1.view(-1, C, Area).permute(2, 0, 1) + self.pos_emb.permute(1, 0, 2)

        attnA, _ = self.cross_attention(query=p1, key=p2, value=p2)
        p1x = self.lnorm1(attnA + p1)
        
        p1xfc = self.lnorm2(self.lin(p1x) + p1x)
        
        attnB, _ = self.cross_attention(query=p2, key=p1, value=p1)
        p2x = self.lnorm1(attnB + p2)
        
        p2xfc = self.lnorm2(self.lin(p2x) + p2x)
        
        fusedA = p1xfc.permute(1, 2, 0).view(B, C, H, W)
        fusedB = p2xfc.permute(1, 2, 0).view(B, C, H, W)
        
        return fusedA, fusedB

class MultiViewCrossAttention(nn.Module):
    def __init__(self, channels, num_views):
        super().__init__()

        self.channels = channels

        self.num_views = num_views
        
        self.adaavgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        
        self.mha = nn.MultiheadAttention(256, 8, dropout=0.1, bias=True)
    
        # Conv2d layers for Q, K, V for all heads
        self.query = nn.Conv2d(self.channels, self.channels//2, 1)
        self.key = nn.Conv2d(self.channels, self.channels//2, 1)
        self.value = nn.Conv2d(self.channels, self.channels//2, 1)
        self.zo =  nn.Conv2d(self.channels//2, self.channels, 1)
        
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(self.channels, self.channels)

    
    def attention(self, Q, K, V, dims, mask=None):
        d_k = Q.size(-1)
#         print("dk", d_k)
        scores = torch.bmm(Q, K / (d_k ** 0.5))
#         print("sc", scores.shape)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)

        Z = torch.bmm(attention_weights, V)              # B x HW x 256
#         print("Z_0", Z.shape)
        Z = Z.permute(0, 2, 1).view(-1, self.channels//2, dims[0], dims[1])             # B x 256 x H x W
#         print("Z_p", Z.shape)
        return Z, attention_weights

    def forward(self, base_view_id, povs, dims, mask=None):
        H = dims[0]
        W = dims[1]
        area = H * W
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            X_base = povs[base_view_id]
            Q_base = self.query(X_base)
            Q_base_flat = Q_base.view(-1, self.channels//2, area).permute(0, 2, 1)         # B x HW x 256
    #             print("Q1", Q1.shape)

            for i in range(0, self.num_views):
#                 print(i, base_view_id, self.num_views)
                if i != base_view_id:
#                     print(i)
                    X_n = povs[i]
                    K_n = self.key(X_n)
                    V_n = self.value(X_n)

                    K_n_flat = K_n.view(-1, self.channels//2, area)                          # B x 256 x HW
    #                 print("K2", K2_flat.shape)
                    V_n_flat = V_n.view(-1, self.channels//2, area).permute(0, 2, 1)         # B x HW x 256
    #                 print("K2", V2_flat.shape)


                    # Perform scaled dot-product attention and concatenate heads
#                     Q_base_flat = Q_base_flat.half()
#                     K_n_flat = K_n_flat.half()
#                     V_n_flat = V_n_flat.half()
                    Z, _ = self.attention(Q_base_flat, K_n_flat, V_n_flat, dims, mask)
                    #Z = self.mha(
#                     print(Z.dtype)
                    out = self.zo(Z)
                    X_base = X_base + out

        # Final linear transformation
        X_base /= self.num_views
        pooled = self.adaavgpool2d(X_base)
        forward = self.fc1(self.flat(pooled))
        return forward

class FeetNet(nn.Module):
    def __init__(self, num_views):
        super().__init__()
        
        self.num_views = num_views
        #self.backbone1_swin3d_conv = nn.Sequential(*list(models.video.swin3d_t(pretrained=True, progress=True).children()))
        #self.backbone2_swin3d_conv = nn.Sequential(*list(models.video.swin3d_t(pretrained=True, progress=True).children()))
        
        #print(self.backbone1_swin3d_conv)
        self.backbone_lin = nn.Linear(512, 11, bias=True)
        self.backbone1_r2p1d_conv = nn.Sequential(*list(models.video.r2plus1d_18(pretrained=True, progress=True).children())[:-2])
        self.backbone2_r2p1d_conv = nn.Sequential(*list(models.video.r2plus1d_18(pretrained=True, progress=True).children())[:-2])
        # for params in self.backbone1_r2p1d_conv.parameters():
        #     params.requires_grad = False
        # for params in self.backbone2_r2p1d_conv.parameters():
        #     params.requires_grad = False
        #self.backbone1_r2p1d_conv = nn.Sequential(*list(models.video.r2plus1d_18(pretrained=True, progress=True).children())[:-3])
        #self.backbone2_r2p1d_conv = nn.Sequential(*list(models.video.r2plus1d_18(pretrained=True, progress=True).children())[:-3])
        # self.pov3_r2p1d_conv = nn.Sequential(*list(models.video.r2plus1d_18(pretrained=True, progress=True).children())[:-3])
        # self.pov4_r2p1d_conv = nn.Sequential(*list(models.video.r2plus1d_18(pretrained=True, progress=True).children())[:-3])
        #print(self.backbone1_r2p1d_conv)
        
        # self.own_backbone = nn.Sequential(
		# nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1),
		# nn.BatchNorm3d(64),
		# nn.ReLU(),
		
		# nn.Conv3d(64, 256, kernel_size=3, stride=2, padding=1),
		# nn.BatchNorm3d(256),
		# nn.ReLU(),
		
		# nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
		# nn.BatchNorm3d(512),
		# nn.ReLU())
		
        self.own_backbone_2d = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
		
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU())
        
        
        self.flat = nn.Flatten()
        
        
        self.multiview_cross_attention = CrossViewAttention(256, 8, 512, 0.1)

        #self.multiview_cross_attention1 = MultiViewCrossAttention(256, self.num_views)
        #self.multiview_cross_attention2 = MultiViewCrossAttention(256, self.num_views)
        # self.multiview_cross_attention3 = MultiViewCrossAttention(256, self.num_views)
        # self.multiview_cross_attention4 = MultiViewCrossAttention(256, self.num_views)
        # self.pool = nn.MaxPool2d(2)
        self.activation = nn.ReLU()

        
        #self.fcn = nn.Sequential(nn.Linear(50176, 8192),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(8192, 4096),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(4096, 2048),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Dropout()
        # )
            
            
        #self.fcn1x1 = nn.Conv2d(256, 16, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))		
            
        self.fc1 = nn.Linear(25088, 16384)
        # self.fc1 = nn.Linear(50176, 20840)
        #self.fc2 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(16384, 8192)
        #self.fc3 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(8192, 2048)
        self.fc4 = nn.Linear(2048, 512)
        self.fc5 = nn.Linear(512, 5)
        
        self.fc_bn1 = nn.LayerNorm(16384)
        self.fc_bn2 = nn.LayerNorm(8192)
        
        self.dropout = nn.Dropout(0.3)
        
        self.l_fc1 = nn.Linear(2048, 512)
        self.l_fc2 = nn.Linear(512, 64)
        self.l_fc3 = nn.Linear(64, 5)
        
        self.r_fc1 = nn.Linear(2048, 512)
        self.r_fc2 = nn.Linear(512, 64)
        self.r_fc3 = nn.Linear(64, 5)
        

      

    def forward(self, v1):#, v3, v4):
        
        # pov1_maps = torch.reshape(self.pov1_r2p1d_conv(v1), (-1, 512, 16, 16))
        # pov2_maps = torch.reshape(self.pov2_r2p1d_conv(v2), (-1, 512, 16, 16))
        # pov3_maps = torch.reshape(self.pov3_r2p1d_conv(v3), (-1, 512, 16, 16))
        # pov4_maps = torch.reshape(self.pov4_r2p1d_conv(v4), (-1, 512, 16, 16))
        # pov1_maps = self.pool(torch.reshape(self.pov1_r2p1d_conv(v1), (-1, 256, 32, 32)))
        # pov2_maps = self.pool(torch.reshape(self.pov2_r2p1d_conv(v2), (-1, 256, 32, 32)))
        # pov3_maps = self.pool(torch.reshape(self.pov3_r2p1d_conv(v3), (-1, 256, 32, 32)))
        # pov4_maps = self.pool(torch.reshape(self.pov4_r2p1d_conv(v4), (-1, 256, 32, 32)))
        # print("before", v1.shape)
        # v1 = v1.permute(0, 1, 4, 2, 3)
        # v2 = v2.permute(0, 1, 4, 2, 3)
        # print("after", v1.shape)


        pov1_maps = self.backbone1_r2p1d_conv(v1)
        #print(pov1_maps.shape)
        #out = self.backbone_lin(pov1_maps.squeeze((2, 3, 4)))
        #return out
        #pov2_maps = self.backbone2_r2p1d_conv(v2)
        # pov3_maps = self.pov3_r2p1d_conv(v3)
        # pov4_maps = self.pov4_r2p1d_conv(v4)
        #dims = pov1_maps.shape
        #print(pov2_maps.shape)

        #pov1_maps = torch.reshape((pov1_maps), (-1, dims[1], dims[3], dims[4]))
        #pov2_maps = torch.reshape((pov2_maps), (-1, dims[1], dims[3], dims[4]))
        
        #povs = torch.stack((pov1_maps, pov2_maps))#, pov3_maps, pov4_maps]

        #hw_dims = dims[3:]
        #attention_out1, attention_out2 = self.multiview_cross_attention(pov1_maps, pov2_maps)
    
        #attention_out1 = self.multiview_cross_attention1(0, povs, hw_dims)
        #attention_out2 = self.multiview_cross_attention2(1, povs, hw_dims)
        # attention_out3 = self.multiview_cross_attention3(2, povs)
        # attention_out4 = self.multiview_cross_attention4(3, povs)
        
        #combined = torch.stack((attention_out1, attention_out2))#, attention_out3, attention_out4))
		
        
        #print(attention_out1.shape)
        #print(combined.shape)
        
        #print(crunched.shape)
        #meaned = torch.mean(combined, 0, True).squeeze(0)
        #crunched1 = self.global_pool(meaned).squeeze((0, 2, 3))
        #crunched1 = self.global_pool(meaned).squeeze((0, 2, 3))


        #print(crunched.shape)
        p1 = self.flat(pov1_maps)#.squeeze(2)
        #p2 = self.flat(pov2_maps)
        #comb = torch.cat((p1, p2), dim=1)
        
        #comb = p1 + p2
        
        #print(meaned.shape)
        
        #print(meaned.shape)
        x = self.activation(self.fc_bn1(self.fc1(p1)))
        x = self.dropout(x)
        x = self.activation(self.fc_bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.activation(self.fc4(x))
        x = self.dropout(x)
        x = self.activation(self.fc5(x))
        
        #x = self.activation(self.fc4(x))
        
        #x = self.fcn(meaned)
        
        #lx = self.activation(self.l_fc1(x))
        #lx = self.activation(self.l_fc2(lx))
        #lx = self.l_fc3(lx)
        
        #rx = self.activation(self.r_fc1(x))
        #rx = self.activation(self.r_fc2(rx))
        #rx = self.r_fc3(rx)
        
        #l_scores = F.softmax(lx, dim=-1)
        #r_scores = F.softmax(rx, dim=-1)
        
        return x #lx, rx
