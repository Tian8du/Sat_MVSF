
"""
This is a script for running the Sat-MVSF.
Copyright (C) <2023> <Jian Gao & GPCV>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import torch
import torch.nn.functional as F

# For RPC Warping
# For QCF implement, see https://github.com/WHU-GPCV/SatMVS
def RPC_PLH_COEF(P, L, H, coef):
    # P: (batch, n_num)

    # import time
    # start = time.time()
    with torch.no_grad():
        coef[:, :, 1] = L
        coef[:, :, 2] = P
        coef[:, :, 3] = H
        coef[:, :, 4] = L * P
        coef[:, :, 5] = L * H
        coef[:, :, 6] = P * H
        coef[:, :, 7] = L * L
        coef[:, :, 8] = P * P
        coef[:, :, 9] = H * H
        coef[:, :, 10] = P * coef[:, :, 5]
        coef[:, :, 11] = L * coef[:, :, 7]
        coef[:, :, 12] = L * coef[:, :, 8]
        coef[:, :, 13] = L * coef[:, :, 9]
        coef[:, :, 14] = L * coef[:, :, 4]
        coef[:, :, 15] = P * coef[:, :, 8]
        coef[:, :, 16] = P * coef[:, :, 9]
        coef[:, :, 17] = L * coef[:, :, 5]
        coef[:, :, 18] = P * coef[:, :, 6]
        coef[:, :, 19] = H * coef[:, :, 9]
        # torch.cuda.synchronize()
        # end = time.time()

        # print(P.shape, L.shape, H.shape)
        # print((H*H*H).shape)
    # if P.shape[1] == 7426048:
        # print(P.shape, end-start, "s")
    # return coef


def RPC_Obj2Photo(inlat, inlon, inhei, rpc, coef):
    # inlat: (B, ndepth*H* W)
    # inlon:  (B, ndepth*H* W)
    # inhei:  (B, ndepth*H*W)
    # rpc: (B, 170)

    with torch.no_grad():
        lat = inlat.clone()
        lon = inlon.clone()
        hei = inhei.clone()

        lat -= rpc[:, 2].view(-1, 1) # self.LAT_OFF
        lat /= rpc[:, 7].view(-1, 1) # self.LAT_SCALE

        lon -= rpc[:, 3].view(-1, 1) # self.LONG_OFF
        lon /= rpc[:, 8].view(-1, 1) # self.LONG_SCALE

        hei -= rpc[:, 4].view(-1, 1) # self.HEIGHT_OFF
        hei /= rpc[:, 9].view(-1, 1) # self.HEIGHT_SCALE

        RPC_PLH_COEF(lat, lon, hei, coef)

        # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
        samp = torch.sum(coef * rpc[:, 50: 70].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[:, 70:90].view(-1, 1, 20), dim=-1)
        line = torch.sum(coef * rpc[:, 10: 30].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[:, 30:50].view(-1, 1, 20), dim=-1)

        samp *= rpc[:, 6].view(-1, 1) # self.SAMP_SCALE
        samp += rpc[:, 1].view(-1, 1) # self.SAMP_OFF

        line *= rpc[:, 5].view(-1, 1) # self.LINE_SCALE
        line += rpc[:, 0].view(-1, 1) # self.LINE_OFF

    return samp, line


def RPC_Photo2Obj(insamp, inline, inhei, rpc, coef):
    # insamp: (B, ndepth*H* W)
    # inline:  (B, ndepth*H* W)
    # inhei:  (B, ndepth*H* W)
    # rpc: (B, 170)

    # import time

    with torch.no_grad():
        # torch.cuda.synchronize()
        # t0 = time.time()
        samp = insamp.clone()
        line = inline.clone()
        hei = inhei.clone()

        samp -= rpc[:, 1].view(-1, 1) # self.SAMP_OFF
        samp /= rpc[:, 6].view(-1, 1) # self.SAMP_SCALE

        line -= rpc[:, 0].view(-1, 1) # self.LINE_OFF
        line /= rpc[:, 5].view(-1, 1) # self.LINE_SCALE

        hei -= rpc[:, 4].view(-1, 1) # self.HEIGHT_OFF
        hei /= rpc[:, 9].view(-1, 1) # self.HEIGHT_SCALE
        # t1 = time.time()
        RPC_PLH_COEF(samp, line, hei, coef)
        # torch.cuda.synchronize()
        # t2 = time.time()

        # coef: (B, ndepth*H*W, 20) rpc[:, 90:110] (B, 20)
        lat = torch.sum(coef * rpc[:, 90:110].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[:, 110:130].view(-1, 1, 20), dim=-1)
        lon = torch.sum(coef * rpc[:, 130:150].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[:, 150:170].view(-1, 1, 20), dim=-1)
        # torch.cuda.synchronize()
        # t3 = time.time()

        lat *= rpc[:, 7].view(-1, 1)
        lat += rpc[:, 2].view(-1, 1)

        lon *= rpc[:, 8].view(-1, 1)
        lon += rpc[:, 3].view(-1, 1)
        # torch.cuda.synchronize()
        # t4 = time.time()
    # if (insamp.shape[1]==7426048):
        # print(t1 - t0, "s")
        # print(t2 - t1, "s")
        # print(t3 - t2, "s")
        # print(t4 - t3, "s")
        # print()
    return lat, lon


def rpc_warping(src_fea, src_rpc, ref_rpc, depth_values, coef):
    # src_fea: [B, C, H, W]
    # src_rpc: [B, 170]
    # ref_rpc: [B, 170]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]

    # import time
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.double, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.double, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y = y.view(1, 1, height, width).repeat(batch, num_depth, 1, 1) # (B, ndepth, H, W)
        x = x.view(1, 1, height, width).repeat(batch, num_depth, 1, 1)

        if len(depth_values.shape) == 2:
            h = depth_values.view(batch, num_depth, 1, 1).double().repeat(1, 1, height, width) # (B, ndepth, H, W)
        else:
            h = depth_values # (B, ndepth, H, W)

        x = x.view(batch, -1)
        y = y.view(batch, -1)
        h = h.view(batch, -1)
        h = h.double()

        # start = time.time()
        lat, lon = RPC_Photo2Obj(x, y, h, ref_rpc, coef)
        samp, line = RPC_Obj2Photo(lat, lon, h, src_rpc, coef) # (B, ndepth*H*W)
        # end = time.time()

        # print(torch.mean(samp - x), torch.var(samp - x))
        # print(torch.mean(line - y), torch.var(line - y))

        samp = samp.float()
        line = line.float()

        proj_x_normalized = samp / ((width - 1) / 2) - 1
        proj_y_normalized = line / ((height - 1) / 2) - 1
        proj_x_normalized = proj_x_normalized.view(batch, num_depth, height*width)
        proj_y_normalized = proj_y_normalized.view(batch, num_depth, height * width)

        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    # if height == 592*4:
        # print(end - start, "s")

    return warped_src_fea

