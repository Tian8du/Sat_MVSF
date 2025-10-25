"""
This is a script for running the Sat-MVSF.
Copyright (C) <2023> <Jin Liu, Jian Gao, GPCV>

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

from satmvs.modules.module import *
from satmvs.modules.warping import rpc_warping
from satmvs.modules.depth_range import get_depth_range_samples


###########################################################
# Compute depth during training
###########################################################
def compute_depth_when_train(features, proj_matrices, depth_values, num_depth, cost_regularization):
    proj_matrices = torch.unbind(proj_matrices, 1)
    assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
    assert depth_values.shape[1] == num_depth, f"depth_values.shape[1]:{depth_values.shape[1]}  num_depth:{num_depth}"

    num_views = len(features)
    ref_feature, src_features = features[0], features[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    b_num, f_num, img_h, img_w = ref_feature.shape
    device = ref_feature.device

    ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
    volume_sum = ref_volume
    volume_sq_sum = ref_volume ** 2
    del ref_volume

    coef = torch.ones((b_num, img_h * img_w * num_depth, 20), dtype=torch.double, device=device)

    for src_fea, src_proj in zip(src_features, src_projs):
        warped_volume = rpc_warping(src_fea, src_proj, ref_proj, depth_values, coef)
        volume_sum = volume_sum + warped_volume
        volume_sq_sum = volume_sq_sum + warped_volume ** 2
        del warped_volume

    volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
    prob_volume = cost_regularization(volume_variance)
    prob_volume = F.softmax(prob_volume, dim=1)

    depth = depth_regression(prob_volume, depth_values=depth_values)
    photometric_confidence, _ = prob_volume.max(1)

    return {"depth": depth, "photometric_confidence": photometric_confidence}


###########################################################
# CascadeREDNet (train)
###########################################################
class CascadeREDNet(nn.Module):
    def __init__(self, min_interval=2.5, ndepths=[48, 32, 8],
                 depth_interals_ratio=[4, 2, 1], share_cr=False, cr_base_chs=[8, 8, 8]):
        super(CascadeREDNet, self).__init__()
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.min_interval = min_interval

        print(f"********** ndepths:{ndepths}, depth_intervals_ratio:{depth_interals_ratio}, chs:{self.cr_base_chs} ************")
        assert len(ndepths) == len(depth_interals_ratio)

        if self.num_stage == 3:
            self.stage_infos = {
                "stage1": {"scale": 4.0},
                "stage2": {"scale": 2.0},
                "stage3": {"scale": 1.0},
            }
        elif self.num_stage == 2:
            self.stage_infos = {
                "stage1": {"scale": 4.0},
                "stage2": {"scale": 1.0},
            }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode='unet')

        if self.share_cr:
            self.cost_regularization = RED_Regularization(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([
                RED_Regularization(in_channels=self.feature.out_channels[i], base_channels=self.cr_base_chs[i])
                for i in range(self.num_stage)
            ])

    def forward(self, imgs, proj_matrices, depth_values):
        features = []
        for nview_idx in range(imgs.size(1)):  # imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        img_h, img_w = imgs.shape[-2:]
        outputs = {}
        depth, cur_depth = None, None

        for stage_idx in range(self.num_stage):
            features_stage = [feat[f"stage{stage_idx + 1}"] for feat in features]
            proj_matrices_stage = proj_matrices[f"stage{stage_idx + 1}"]
            stage_scale = self.stage_infos[f"stage{stage_idx + 1}"]["scale"]

            if depth is not None:
                cur_depth = F.interpolate(depth.unsqueeze(1), [img_h, img_w], mode='bilinear', align_corners=False).squeeze(1)
            else:
                cur_depth = depth_values

            depth_range_samples = get_depth_range_samples(
                cur_depth=cur_depth,
                ndepth=self.ndepths[stage_idx],
                depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * self.min_interval,
                dtype=imgs.dtype,
                device=imgs.device,
                shape=[imgs.shape[0], img_h, img_w]
            )

            dv = F.interpolate(depth_range_samples.unsqueeze(1),
                               [self.ndepths[stage_idx], img_h // int(stage_scale), img_w // int(stage_scale)],
                               mode='trilinear', align_corners=False)

            outputs_stage = compute_depth_when_train(
                features_stage, proj_matrices_stage,
                depth_values=dv.squeeze(1),
                num_depth=self.ndepths[stage_idx],
                cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx],
            )

            depth = outputs_stage['depth']
            outputs[f"stage{stage_idx + 1}"] = outputs_stage
            outputs.update(outputs_stage)

        return outputs


###########################################################
# Inference version
###########################################################
def compute_depth_when_pred(features, proj_matrices, depth_values, num_depth, cost_regularization):
    proj_matrices = torch.unbind(proj_matrices, 1)
    assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
    assert depth_values.shape[1] == num_depth, f"depth_values.shape[1]:{depth_values.shape[1]}  num_depth:{num_depth}"

    num_views = len(features)
    ref_feature, src_features = features[0], features[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    b_num, f_num, img_h, img_w = ref_feature.shape
    device = ref_feature.device

    # 初始化状态张量
    state1 = torch.zeros((b_num, 8,  img_h, img_w), device=device)
    state2 = torch.zeros((b_num, 16, img_h // 2, img_w // 2), device=device)
    state3 = torch.zeros((b_num, 32, img_h // 4, img_w // 4), device=device)
    state4 = torch.zeros((b_num, 64, img_h // 8, img_w // 8), device=device)

    exp_sum        = torch.zeros((b_num, 1, img_h, img_w), dtype=torch.double, device=device)
    depth_image    = torch.zeros((b_num, 1, img_h, img_w), dtype=torch.double, device=device)
    max_prob_image = torch.zeros((b_num, 1, img_h, img_w), dtype=torch.double, device=device)

    coef = torch.ones((b_num, img_h * img_w, 20), dtype=torch.double, device=device)

    for d in range(num_depth):
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, 1, 1, 1)
        depth_value = depth_values[:, d:d + 1]

        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume

        for src_fea, src_proj in zip(src_features, src_projs):
            warped_volume = rpc_warping(src_fea, src_proj, ref_proj, depth_value, coef)
            volume_sum += warped_volume
            volume_sq_sum += warped_volume ** 2
            del warped_volume

        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2)).squeeze(2)
        reg_cost, state1, state2, state3, state4 = cost_regularization(volume_variance, state1, state2, state3, state4)

        reg_cost = reg_cost.double()
        prob = reg_cost.exp()
        update_flag_image = (max_prob_image < prob).double()
        new_max_prob_image = update_flag_image * prob + (1 - update_flag_image) * max_prob_image
        new_depth_image = depth_value.double() * prob + depth_image

        max_prob_image = new_max_prob_image
        depth_image = new_depth_image
        exp_sum = exp_sum + prob

    forward_exp_sum = exp_sum + 1e-10
    forward_depth_map = (depth_image / forward_exp_sum).squeeze(1).float()
    forward_prob_map = (max_prob_image / forward_exp_sum).squeeze(1).float()

    return {"depth": forward_depth_map, "photometric_confidence": forward_prob_map}


###########################################################
# Inference CascadeREDNet
###########################################################
class Infer_CascadeREDNet(nn.Module):
    def __init__(self, min_interval=2.5, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1],
                 share_cr=False, cr_base_chs=[8, 8, 8]):
        super(Infer_CascadeREDNet, self).__init__()
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.min_interval = min_interval

        assert len(ndepths) == len(depth_interals_ratio)

        if self.num_stage == 3:
            self.stage_infos = {
                "stage1": {"scale": 4.0},
                "stage2": {"scale": 2.0},
                "stage3": {"scale": 1.0},
            }
        elif self.num_stage == 2:
            self.stage_infos = {
                "stage1": {"scale": 4.0},
                "stage2": {"scale": 1.0},
            }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode='unet')

        if self.share_cr:
            self.cost_regularization = slice_RED_Regularization(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([
                slice_RED_Regularization(in_channels=self.feature.out_channels[i], base_channels=self.cr_base_chs[i])
                for i in range(self.num_stage)
            ])

    def forward(self, imgs, proj_matrices, depth_values):
        features = []
        for nview_idx in range(imgs.size(1)):
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        img_h, img_w = imgs.shape[-2:]
        outputs = {}
        depth, cur_depth = None, None

        for stage_idx in range(self.num_stage):
            features_stage = [feat[f"stage{stage_idx + 1}"] for feat in features]
            proj_matrices_stage = proj_matrices[f"stage{stage_idx + 1}"]
            stage_scale = self.stage_infos[f"stage{stage_idx + 1}"]["scale"]

            if depth is not None:
                cur_depth = F.interpolate(depth.unsqueeze(1), [img_h, img_w], mode='bilinear', align_corners=False).squeeze(1)
            else:
                cur_depth = depth_values

            depth_range_samples = get_depth_range_samples(
                cur_depth=cur_depth,
                ndepth=self.ndepths[stage_idx],
                depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * self.min_interval,
                dtype=imgs.dtype,
                device=imgs.device,
                shape=[imgs.shape[0], img_h, img_w]
            )

            dv = F.interpolate(depth_range_samples.unsqueeze(1),
                               [self.ndepths[stage_idx], img_h // int(stage_scale), img_w // int(stage_scale)],
                               mode='trilinear', align_corners=False)

            outputs_stage = compute_depth_when_pred(
                features_stage, proj_matrices_stage,
                depth_values=dv.squeeze(1),
                num_depth=self.ndepths[stage_idx],
                cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx],
            )

            depth = outputs_stage['depth']
            outputs[f"stage{stage_idx + 1}"] = outputs_stage
            outputs.update(outputs_stage)

        return outputs
