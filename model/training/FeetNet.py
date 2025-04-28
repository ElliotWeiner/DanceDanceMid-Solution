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





class MultiViewCrossAttention(nn.Module):
    def __init__(self, channels, num_views):
        super().__init__()

        self.channels = channels

        self.num_views = num_views
        
        self.adaavgpool2d = nn.AdaptiveAvgPool2d((1, 1))
    
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
        self.backbone1_r2p1d_conv = nn.Sequential(*list(models.video.r2plus1d_18(pretrained=False, progress=True).children())[:-3])
        self.backbone2_r2p1d_conv = nn.Sequential(*list(models.video.r2plus1d_18(pretrained=False, progress=True).children())[:-3])
        # self.pov3_r2p1d_conv = nn.Sequential(*list(models.video.r2plus1d_18(pretrained=True, progress=True).children())[:-3])
        # self.pov4_r2p1d_conv = nn.Sequential(*list(models.video.r2plus1d_18(pretrained=True, progress=True).children())[:-3])
        
        self.flat = nn.Flatten()
        
        self.multiview_cross_attention1 = MultiViewCrossAttention(256, self.num_views)
        self.multiview_cross_attention2 = MultiViewCrossAttention(256, self.num_views)
        # self.multiview_cross_attention3 = MultiViewCrossAttention(256, self.num_views)
        # self.multiview_cross_attention4 = MultiViewCrossAttention(256, self.num_views)
        # self.pool = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(256, 10420)
        #self.fc1 = nn.Linear(204800, 10420)
        #self.fc2 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(10420, 2048)
        #self.fc3 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(2048, 512)
        
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc_bn2 = nn.BatchNorm1d(512)
        
        self.dropout = nn.Dropout()
        
        self.l_fc1 = nn.Linear(512, 256)
        self.l_fc2 = nn.Linear(256, 32)
        self.l_fc3 = nn.Linear(32, 5)
        
        self.r_fc1 = nn.Linear(512, 256)
        self.r_fc2 = nn.Linear(256, 32)
        self.r_fc3 = nn.Linear(32, 5)
        
        self.activation = nn.ReLU()

      

    def forward(self, v1, v2):#, v3, v4):
        
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
        pov2_maps = self.backbone2_r2p1d_conv(v2)
        # pov3_maps = self.pov3_r2p1d_conv(v3)
        # pov4_maps = self.pov4_r2p1d_conv(v4)
        dims = pov2_maps.shape
        # print(pov2_maps.shape)

        pov1_maps = torch.reshape((pov1_maps), (-1, dims[1], dims[3], dims[4]))
        pov2_maps = torch.reshape((pov2_maps), (-1, dims[1], dims[3], dims[4]))
        
        povs = torch.stack((pov1_maps, pov2_maps))#, pov3_maps, pov4_maps]

        hw_dims = dims[3:]
    
        attention_out1 = self.multiview_cross_attention1(0, povs, hw_dims)
        attention_out2 = self.multiview_cross_attention2(1, povs, hw_dims)
        # attention_out3 = self.multiview_cross_attention3(2, povs)
        # attention_out4 = self.multiview_cross_attention4(3, povs)
        
        combined = torch.stack((attention_out1, attention_out2))#, attention_out3, attention_out4))
        
#         print(attention_out1.shape)
#         print(combined.shape)
        
        meaned = torch.mean(combined, 0, True)
        
#         print(meaned.shape)
        #p1 = self.flat(pov1_maps)
        #p2 = self.flat(pov2_maps)
        #meaned = torch.cat((p1, p2), dim=1)
        #print(p1.shape)
        x = self.activation(self.fc1(meaned))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        #x = self.activation(self.fc4(x))
        
        lx = self.activation(self.l_fc1(x))
        lx = self.activation(self.l_fc2(lx))
        lx = self.l_fc3(lx)
        
        rx = self.activation(self.r_fc1(x))
        rx = self.activation(self.r_fc2(rx))
        rx = self.r_fc3(rx)
        
        # l_scores = F.softmax(lx, dim=-1)
        # r_scores = F.softmax(rx, dim=-1)
        
        return lx, rx
