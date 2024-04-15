"""Model Define."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022-2024, All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 22日 星期四 03:54:39 CST
# ***
# ************************************************************************************/
#
import os
import math

import torch
from torch import nn
from torch.nn import BatchNorm2d
import torch.nn.functional as F

from typing import List

import todos
import pdb

class AnimeFace(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.keypt_detector = KeypointDetector()
        self.face_generator = OcclusionAwareGenerator()
        self.load_weights()
        self.eval()

    def load_weights(self, model_path="models/anime_face.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)
        self.face_generator.load_state_dict(sd['generator'])
        self.keypt_detector.load_state_dict(sd['kp_detector'])

    def forward(self, source_kp, offset_kp, source_tensor):
        return source_kp


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, 
            block_expansion=64, 
            num_blocks=5, 
            max_features=1024, 
            num_kp=10, 
            num_channels=3, 
            scale_factor=0.25, 
            kp_variance=0.01, 
        ):
        super().__init__()

        infeatures = num_kp + 1
        # self.infeatures = infeatures
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=infeatures * (num_channels + 1),
                                max_features=max_features, num_blocks=num_blocks)
        self.mask = nn.Conv2d(self.hourglass.out_filters, infeatures, kernel_size=(7, 7), padding=(3, 3))
        self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        else:
            pdb.set_trace()

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        if 'jacobian' in kp_driving:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)
            # n * 10 * h * w * 2

        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 2)

        # #adding background feature
        # identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        # adding background feature
        bg_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        sparse_motions = torch.cat([bg_grid, driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper
        """
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1: # True
            source_image = self.down(source_image)
        else:
            pdb.set_trace()

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        # n * 11 * h * w * 2
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        # n * 11 * 3 * h * w

        out_dict['sparse_deformed'] = deformed_source

        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        # n * 11 * 2 * h * w
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)
        # n * h * w * 2

        out_dict['deformation'] = deformation

        if self.occlusion: # True
            out_dict['occlusion_map'] = torch.sigmoid(self.occlusion(prediction))

        return out_dict


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, 
                 num_channels=3, 
                 num_kp=14, 
                 block_expansion=64, 
                 max_features=512, 
                 num_down_blocks=2,
                 num_bottleneck_blocks=6, 
                 dense_motion_params={'block_expansion': 64, 'max_features': 1024, 'num_blocks': 5, 'scale_factor': 0.25}, 
                 num_root_kp=4, 
                ):
        super().__init__()

        dense_motion_kp_num = num_kp - num_root_kp # 10
        self.dense_motion_network = DenseMotionNetwork(num_kp=dense_motion_kp_num, num_channels=num_channels,
                                                       **dense_motion_params)

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks): # 2
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))



    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def apply_optical(self, input_previous, input_skip, motion_params):
        # tensor [source_image] size: [1, 3, 256, 256], min: 0.003138, max: 1.0, mean: 0.484256
        # kp_driving is dict:
        #     tensor [value] size: [1, 10, 2], min: -0.403658, max: 0.757789, mean: 0.047853
        #     tensor [jacobian] size: [1, 10, 2, 2], min: -0.868667, max: 1.904356, mean: 0.797541
        # kp_source is dict:
        #     tensor [value] size: [1, 10, 2], min: -0.443909, max: 0.9228, mean: 0.100751
        #     tensor [jacobian] size: [1, 10, 2, 2], min: -0.661923, max: 2.661613, mean: 0.976099
        # [driving_image] type: <class 'NoneType'>
        # [input_previous] type: <class 'NoneType'>
        # tensor [input_skip] size: [1, 256, 64, 64], min: 0.0, max: 15.954874, mean: 0.531201
        # motion_params is dict:
        #     tensor [sparse_deformed] size: [1, 11, 3, 64, 64], min: 0.0, max: 0.993511, mean: 0.344482
        #     tensor [mask] size: [1, 11, 64, 64], min: 0.0, max: 0.999667, mean: 0.090909
        #     tensor [deformation] size: [1, 64, 64, 2], min: -1.000909, max: 1.065437, mean: 0.029458
        #     tensor [occlusion_map] size: [1, 1, 64, 64], min: 0.024466, max: 0.575934, mean: 0.256564

        occlusion_map = motion_params['occlusion_map']
        deformation = motion_params['deformation']
        input_skip = self.deform_input(input_skip, deformation)

        if occlusion_map is not None: # True
            if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear')
            if input_previous is not None:
                input_skip = input_skip * occlusion_map + input_previous * (1 - occlusion_map)
            else:
                # ==> pdb.set_trace()
                input_skip = input_skip * occlusion_map
        out = input_skip

        # tensor [out] size: [1, 256, 64, 64], min: 0.0, max: 4.157534, mean: 0.128219
        return out

    def forward(self, source_image, kp_driving, kp_source, driving_image=None):
        # tensor [source_image] size: [1, 3, 256, 256], min: 0.003138, max: 1.0, mean: 0.484256
        # kp_driving is dict:
        #     tensor [value] size: [1, 10, 2], min: -0.403658, max: 0.757789, mean: 0.047853
        #     tensor [jacobian] size: [1, 10, 2, 2], min: -0.868667, max: 1.904356, mean: 0.797541
        # kp_source is dict:
        #     tensor [value] size: [1, 10, 2], min: -0.443909, max: 0.9228, mean: 0.100751
        #     tensor [jacobian] size: [1, 10, 2, 2], min: -0.661923, max: 2.661613, mean: 0.976099
        # [driving_image] type: <class 'NoneType'>

        # Encoding (downsampling) part
        out = self.first(source_image)
        skips = [out]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            skips.append(out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                kp_source=kp_source)
        deformation = dense_motion['deformation']
        output_dict['mask'] = dense_motion['mask']
        output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

        occlusion_map = dense_motion['occlusion_map']
        output_dict['occlusion_map'] = occlusion_map
        out_backflow = self.apply_optical(input_previous=None, input_skip=out, motion_params=dense_motion)

        # if occlusion_map is not None:
            # if out_backflow.shape[2] != occlusion_map.shape[2] or out_backflow.shape[3] != occlusion_map.shape[3]:
                # occlusion_map = F.interpolate(occlusion_map, size=out_backflow.shape[2:], mode='bilinear')
            # out_backflow = out_backflow * occlusion_map
        output_dict["deformed"] = self.deform_input(source_image, deformation)

        # Decoding part
        out_backflow = self.bottleneck(out_backflow)
        for i in range(len(self.up_blocks)):
            out_backflow = self.apply_optical(input_skip=skips[-(i + 1)], input_previous=out_backflow, motion_params=dense_motion)
            out_backflow = self.up_blocks[i](out_backflow)
        out_backflow = self.apply_optical(input_skip=skips[0], input_previous=out_backflow, motion_params=dense_motion)

        out_backflow = self.final(out_backflow)
        out_backflow = F.sigmoid(out_backflow)
        out_backflow = self.apply_optical(input_skip=source_image, input_previous=out_backflow, motion_params=dense_motion)

        output_dict["prediction"] = out_backflow

        # output_dict is dict:
        #     tensor [mask] size: [1, 11, 64, 64], min: 0.0, max: 0.999667, mean: 0.090909
        #     tensor [sparse_deformed] size: [1, 11, 3, 64, 64], min: 0.0, max: 0.993511, mean: 0.344482
        #     tensor [occlusion_map] size: [1, 1, 64, 64], min: 0.024466, max: 0.575934, mean: 0.256564
        #     tensor [deformed] size: [1, 3, 256, 256], min: 0.0, max: 1.0, mean: 0.385487
        #     tensor [prediction] size: [1, 3, 256, 256], min: 0.045993, max: 0.981559, mean: 0.388611

        return output_dict


class KeypointDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, 
                 block_expansion=32, 
                 num_kp=14, 
                 num_channels=3, 
                 max_features=1024,
                 num_blocks=5, 
                 temperature=0.1, 
                 estimate_jacobian=True, 
                 scale_factor=0.25, 
                 single_jacobian_map=True, 
                 pad=0,
                 subroot_leaf_attention=True, 
                 attention_channel=64, 
                ):
        super().__init__()
        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if subroot_leaf_attention: # True
            self.subroot_attention_block = nn.Sequential(
                nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=attention_channel, kernel_size=(1, 1), padding=0),
                BatchNorm2d(64),
                nn.ReLU()
            )
            self.leaf_attention_block = nn.Sequential(
                nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=attention_channel, kernel_size=(1, 1), padding=0),
                BatchNorm2d(64),
                nn.ReLU()
            )

        if estimate_jacobian: # True
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            pdb.set_trace()
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1: # True
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        else:
            pdb.set_trace()

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3)) # N * 10 * 2
        kp = {'value': value}

        return kp

    def forward(self, x):
        # tensor [x] size: [1, 3, 256, 256], min: 0.003138, max: 1.0, mean: 0.484256

        if self.scale_factor != 1: # True for self.scale_factor === 0.25
            x = self.down(x)
        else:
            pdb.set_trace()

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)
        out['heatmap'] = heatmap
        out['feature_map'] = feature_map

        if self.jacobian is not None: # True
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])

            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            # N * 10 *2 *2
            out['jacobian'] = jacobian
        else:
            pdb.set_trace()

        # out is dict:
        #     tensor [value] size: [1, 14, 2], min: -0.443909, max: 0.9228, mean: 0.080227
        #     tensor [heatmap] size: [1, 14, 58, 58], min: 0.0, max: 0.160103, mean: 0.000297
        #     tensor [feature_map] size: [1, 35, 64, 64], min: 0.0, max: 10.35937, mean: 0.139735
        #     tensor [jacobian] size: [1, 14, 2, 2], min: -0.795599, max: 2.661613, mean: 0.928748

        return out


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']
    # N * 10 * 2

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    # 1,1,h,w,2
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    # N * 10 * 1 *1 *1
    coordinate_grid = coordinate_grid.repeat(*repeats)
    # N,10,h,w,2

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    # N * 10 * 1 * 1 * 2
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)
    # N * 10 * h * w

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    # xx [[-1,-1/3,1/3,1],[-1,-1/3,1/3,1],[-1,-1/3,1/3,1],[-1,-1/3,1/3,1]]
    # yy [[-1,-1,-1,-1],[-1/3,-1/3,-1/3,-1/3],[1/3,1/3,1/3,1/3],[1,1,1,1]]

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2.0)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class DecoderNoRes(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 1) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            #out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))



class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super().__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            pdb.set_trace()
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out
