import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary

import torchxrayvision as xrv
import skimage, torch, torchvision

class CustomModel(nn.Module):
    def __init__(self, model_name, label_numbers):
        super().__init__()
        self.label_numbers = label_numbers
        # ResNet系列
        if model_name == 'resnet18':
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'resnet34':
            self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'resnet101':
            self.base_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'resnet152':
            self.base_model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'resnext50_32x4d':
            self.base_model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'resnext101_32x8d':
            self.base_model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'resnext101_64x4d':
            self.base_model = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'wide_resnet50_2':
            self.base_model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'wide_resnet101_2':
            self.base_model = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()

        # DenseNet系列
        elif model_name == 'densenet121-res224-chex':
            #self.base_model = xrv.models.DenseNet(weights="densenet121-res224-chex")
            self.base_model = xrv.models.DenseNet(weights="densenet121-res224-nih")
            #self.base_model = xrv.models.DenseNet(weights="all")
            for param in self.base_model.parameters():
                param.requires_grad = False

            in_features_dim = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()

        elif model_name == 'densenet121':
            self.base_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

            in_features_dim = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        elif model_name == 'densenet169':
            self.base_model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        elif model_name == 'densenet161':
            self.base_model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        elif model_name == 'densenet201':
            self.base_model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()

        # VGG系列
        elif model_name == 'vgg11':
            self.base_model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Identity()
        elif model_name == 'vgg11_bn':
            self.base_model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Identity()
        elif model_name == 'vgg13':
            self.base_model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Identity()
        elif model_name == 'vgg13_bn':
            self.base_model = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Identity()
        elif model_name == 'vgg16':
            self.base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Identity()
        elif model_name == 'vgg16_bn':
            self.base_model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Identity()
        elif model_name == 'vgg19':
            self.base_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Identity()
        elif model_name == 'vgg19_bn':
            self.base_model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Identity()

        # Inception系
        elif model_name == 'inception_v3':
            self.base_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'inception_v4':
            self.base_model = models.inception_v4(weights=models.Inception_V4_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'googlenet':
            self.base_model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()

        # EfficientNet系列
        elif model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'efficientnet_b1':
            self.base_model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'efficientnet_b2':
            self.base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'efficientnet_b3':
            self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'efficientnet_b4':
            self.base_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'efficientnet_b5':
            self.base_model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'efficientnet_b6':
            self.base_model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'efficientnet_b7':
            self.base_model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'efficientnet_v2_s':
            self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'efficientnet_v2_m':
            self.base_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'efficientnet_v2_l':
            self.base_model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()

        # MobileNet系列
        elif model_name == 'mobilenet_v1':
            self.base_model = models.mobilenet_v1(weights=models.MobileNet_V1_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'mobilenet_v3_large':
            self.base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[0].in_features
            self.base_model.classifier[0] = nn.Identity()
        elif model_name == 'mobilenet_v3_small':
            self.base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[0].in_features
            self.base_model.classifier[0] = nn.Identity()

        # Vision Transformer (ViT)
        elif model_name == 'vit_b_16':
            self.base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.heads.head.in_features
            self.base_model.heads.head = nn.Identity()
        elif model_name == 'vit_b_32':
            self.base_model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.heads.head.in_features
            self.base_model.heads.head = nn.Identity()
        elif model_name == 'vit_l_16':
            self.base_model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.heads.head.in_features
            self.base_model.heads.head = nn.Identity()
        elif model_name == 'vit_l_32':
            self.base_model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.heads.head.in_features
            self.base_model.heads.head = nn.Identity()
        elif model_name == 'vit_h_14':
            self.base_model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.heads.head.in_features
            self.base_model.heads.head = nn.Identity()

        # ConvNeXt系列
        elif model_name == 'convnext_tiny':
            self.base_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[2].in_features
            self.base_model.classifier[2] = nn.Identity()
        elif model_name == 'convnext_small':
            self.base_model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[2].in_features
            self.base_model.classifier[2] = nn.Identity()
        elif model_name == 'convnext_base':
            self.base_model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[2].in_features
            self.base_model.classifier[2] = nn.Identity()
        elif model_name == 'convnext_large':
            self.base_model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[2].in_features
            self.base_model.classifier[2] = nn.Identity()

        elif model_name == 'xception':
            self.base_model = models.xception(weights=models.Xception_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'mnasnet1_0':
            self.base_model = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Identity()
        elif model_name == 'shufflenet_v2_x0_5':
            self.base_model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'shufflenet_v2_x1_0':
            self.base_model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'shufflenet_v2_x1_5':
            self.base_model = models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'shufflenet_v2_x2_0':
            self.base_model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
            in_features_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()

        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        
        self.fc1 = nn.Linear(in_features_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, label_numbers)

    def forward(self, x):
        x = self.base_model(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        if self.label_numbers == 1:
            return torch.sigmoid(x)
        else:
            return x
    