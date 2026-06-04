import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        if isinstance(kernel_size, tuple):
            padding = tuple(k // 2 for k in kernel_size)
        else:
            padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.max_pool = nn.AdaptiveMaxPool3d(1)

        reduced_channels = max(1, channels // reduction_ratio)

        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=False),
            nn.Linear(reduced_channels, channels)
        )

    def forward(self, x):
        b, c = x.size(0), x.size(1)

        avg_out = self.fc(self.avg_pool(x).view(b, c))

        max_out = self.fc(self.max_pool(x).view(b, c))

        out = avg_out + max_out

        return torch.sigmoid(out).view(b, c, 1, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv3d(2, 1, kernel_size=(kernel_size, kernel_size, kernel_size), padding=(padding, padding, padding))

    def forward(self, x):
        b, c, d, h, w = x.size()

        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        fused = torch.cat([avg_out, max_out], dim=1)

        attention = self.conv(fused)

        attention = torch.sigmoid(attention)
        return attention

class MaskAttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
        self.mask_embed = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(channels),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        if mask is not None:
            if mask.dim() == 4:                 # [B, D, H, W] -> [B, 1, D, H, W]
                mask = mask.unsqueeze(1)
            mask = mask.to(x.dtype)
            mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')
            mask_weight = self.mask_embed(mask)
            x = x * mask_weight
        return x

class PANet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()

        self.input_conv = nn.Sequential(
            ConvBlock(in_channels, 16, kernel_size=(3, 7, 7), stride=(1, 2, 2)),
        )

        self.layer1 = nn.ModuleList([
            ConvBlock(16, 32, stride=(1, 2, 2)),
            ResBlock(32),
            MaskAttentionModule(32)
        ])

        self.layer2 = nn.ModuleList([
            ConvBlock(32, 64, stride=2),
            ResBlock(64),
            MaskAttentionModule(64)
        ])

        self.layer3 = nn.ModuleList([
            ConvBlock(64, 128, stride=2),
            ResBlock(128),
            MaskAttentionModule(128)
        ])

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None): #or mask=None
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if mask is not None and not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.float32)
        x = self.input_conv(x)

        for layer in [self.layer1, self.layer2, self.layer3]:
            x = layer[0](x)
            x = layer[1](x)
            x = layer[2](x, mask)


        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

def test_model():
    # 测试模型
    model = PANet(in_channels=1, num_classes=1)
    batch_size, channels, depth, height, width = 2, 1, 32, 224, 224
    x = torch.randn(batch_size, channels, depth, height, width)
    mask = torch.ones(2, 1, 32, 224, 224)
    output = model(x, mask)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    return model

if __name__ == "__main__":
    model = test_model()
