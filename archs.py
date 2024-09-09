from functools import partial
import torch
import torchvision.models
from torch import nn
import numpy as np
from torchvision.models.convnext import LayerNorm2d
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = ['UNet', 'NestedUNet', 'NestedUNetDeconv']

from torchvision.ops.misc import ConvNormActivation


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = np.array([32, 64, 128, 256, 512]) * 2
        nb_filter = [int(x) for x in nb_filter]
        # nb_filter = [8, 16, 32, 64, 128]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        # if self.deep_supervision:
        self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        # else:
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class Encoder(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()

        self.encoder_model = encoder_model

        # DenseNet Arguments
        if encoder_model == "densenet":
            self.captured_feature_channel_counts = [64, 128, 256, 896]
            model = torchvision.models.densenet201(pretrained=True)
            capture_layers = ["features.relu0", "features.transition1.conv", "features.transition2.conv",
                              "features.transition3.conv"]

        # ConvNext Base Arguments
        elif encoder_model == "convnext4x_base":
            self.captured_feature_channel_counts = [128, 256, 512, 1024]
            capture_layers = ["features.0", "features.2", "features.4", "features.6"]
            model = torchvision.models.convnext_base(pretrained=True)

        # ConvNext Small Arguments
        elif encoder_model == "convnext4x_small":
            self.captured_feature_channel_counts = [96, 192, 384, 768]
            capture_layers = ["features.0", "features.2", "features.4", "features.6"]
            model = torchvision.models.convnext_small(pretrained=True)

        elif encoder_model == "convnext2x":
            self.captured_feature_channel_counts = [128, 256, 512, 1024]
            capture_layers = ["features.0", "features.2", "features.4", "features.6"]
            model = torchvision.models.convnext_base(pretrained=True)

        elif encoder_model == "cswin_base_384":
            from cswin import CSWin
            model = CSWin(img_size=384, patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 12, 12], num_heads=[4, 8, 16, 32], mlp_ratio=4.,)
            model_state_dict = torch.load("cswin_base_384.pth")
            model.load_state_dict(model_state_dict["state_dict_ema"], strict=False)

            self.captured_feature_channel_counts = [96, 192, 384, 768]
            capture_layers = ["merge1", "merge2", "merge3"]

            self.convnext_2x2_feature_extractor = nn.Sequential(
                ConvNormActivation(
                    in_channels=3,
                    out_channels=self.captured_feature_channel_counts[0],
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    norm_layer=partial(LayerNorm2d, eps=1e-6)
                )
            )

        if self.encoder_model.split("_")[0] == "cswin":
            feature_extractor = model
        else:
            feature_extractor = torchvision.models.feature_extraction.create_feature_extractor(model, capture_layers)
        self.feature_extractor = feature_extractor

    def forward(self, x):
        if self.encoder_model.split("_")[0] == "cswin":
            features_list = self.feature_extractor(x)
        else:
            features = self.feature_extractor(x)
            features_list = list(features.values())

        if self.encoder_model == "convnext2x":
            first_feature = self.convnext_2x2_feature_extractor(x)
            features_list = [first_feature] + features_list

        return features_list


class NestedUNetDeconv(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        n_filter = 128
        nb_filter = [32, 64, 128, 256, 512]
        use_deconv = False
        if use_deconv:
            self.up = nn.ModuleList([
                nn.ConvTranspose2d(in_channels=n_filter * 2, out_channels=n_filter * 2, kernel_size=2, stride=2),
                nn.ConvTranspose2d(in_channels=n_filter * 4, out_channels=n_filter * 4, kernel_size=2, stride=2),
                nn.ConvTranspose2d(in_channels=n_filter * 2, out_channels=n_filter * 2, kernel_size=2, stride=2),
                nn.ConvTranspose2d(in_channels=n_filter * 8, out_channels=n_filter * 8, kernel_size=2, stride=2),
                nn.ConvTranspose2d(in_channels=n_filter * 4, out_channels=n_filter * 4, kernel_size=2, stride=2),
                nn.ConvTranspose2d(in_channels=n_filter * 2, out_channels=n_filter * 2, kernel_size=2, stride=2),
                nn.ConvTranspose2d(in_channels=n_filter * 16, out_channels=n_filter * 16, kernel_size=2, stride=2),
                nn.ConvTranspose2d(in_channels=n_filter * 8, out_channels=n_filter * 8, kernel_size=2, stride=2),
                nn.ConvTranspose2d(in_channels=n_filter * 4, out_channels=n_filter * 4, kernel_size=2, stride=2),
                nn.ConvTranspose2d(in_channels=n_filter * 2, out_channels=n_filter * 2, kernel_size=2, stride=2)
            ])
        else:
            self.up = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) for _ in range(10)]
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = VGGBlock(3, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up[0](x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up[1](x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up[2](x1_1)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up[3](x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up[4](x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up[5](x1_2)], 1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up[6](x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up[7](x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up[8](x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up[9](x1_3)], 1))
        output = self.final(x0_4)
        return output
