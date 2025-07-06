import torch
import torch.nn as nn
import math

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接特征
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 通过卷积学习空间注意力图
        out = self.conv(x_cat)
        return self.sigmoid(out)

# 改进的CBAM模块
class ImprovedCBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7, use_residual=True):
        super(ImprovedCBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.use_residual = use_residual
        
    def forward(self, x):
        # 保存输入作为残差连接
        identity = x
        
        # 通道注意力
        ca_out = self.channel_attention(x) * x
        
        # 空间注意力
        sa_out = self.spatial_attention(ca_out) * ca_out
        
        # 残差连接
        if self.use_residual:
            sa_out = sa_out + identity
            
        return sa_out

class LightDehaze_Net(nn.Module):
    def __init__(self):
        super(LightDehaze_Net, self).__init__()
        
        # 基本的LightDehazeNet架构
        self.relu = nn.ReLU(inplace=True)

        self.e_conv_layer1 = nn.Conv2d(3, 8, 1, 1, 0, bias=True)
        self.e_conv_layer2 = nn.Conv2d(8, 8, 3, 1, 1, bias=True)
        self.e_conv_layer3 = nn.Conv2d(8, 8, 5, 1, 2, bias=True)
        self.e_conv_layer4 = nn.Conv2d(16, 16, 7, 1, 3, bias=True)
        self.e_conv_layer5 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.e_conv_layer6 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.e_conv_layer7 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv_layer8 = nn.Conv2d(56, 3, 3, 1, 1, bias=True)
        
        self.attn1 = ImprovedCBAM(16, reduction_ratio=4, use_residual=True)
        self.attn2 = ImprovedCBAM(32, reduction_ratio=8, use_residual=True)
        self.attn3 = ImprovedCBAM(56, reduction_ratio=8, use_residual=True)
        
        # 分支特征增强
        self.branch_attn1 = ImprovedCBAM(8, reduction_ratio=2, use_residual=True)
        self.branch_attn2 = ImprovedCBAM(16, reduction_ratio=4, use_residual=True)
        
        # 额外的跳跃连接
        self.skip_conv = nn.Conv2d(8, 56, 1, bias=False)
        
    def forward(self, img):
        pipeline = []
        pipeline.append(img)

        # 初始特征提取
        conv_layer1 = self.relu(self.e_conv_layer1(img))
        conv_layer2 = self.relu(self.e_conv_layer2(conv_layer1))
        conv_layer3 = self.relu(self.e_conv_layer3(conv_layer2))
        
        # 增强第三个卷积层特征
        enhanced_conv3 = self.branch_attn1(conv_layer3)
        
        # 第一个特征融合点
        concat_layer1 = torch.cat((conv_layer1, enhanced_conv3), 1)
        concat_layer1 = self.attn1(concat_layer1)
        
        # 中间特征提取
        conv_layer4 = self.relu(self.e_conv_layer4(concat_layer1))
        conv_layer5 = self.relu(self.e_conv_layer5(conv_layer4))
        conv_layer6 = self.relu(self.e_conv_layer6(conv_layer5))
        
        # 增强第六个卷积层特征
        enhanced_conv6 = self.branch_attn2(conv_layer6)
        
        # 第二个特征融合点
        concat_layer2 = torch.cat((conv_layer4, enhanced_conv6), 1)
        concat_layer2 = self.attn2(concat_layer2)
        
        conv_layer7 = self.relu(self.e_conv_layer7(concat_layer2))
        
        # 第三个特征融合点
        concat_layer3 = torch.cat((conv_layer2, conv_layer5, conv_layer7), 1)
        
        # 添加跳跃连接
        skip = self.skip_conv(conv_layer2)
        concat_layer3 = concat_layer3 + skip
        
        # 注意力
        concat_layer3 = self.attn3(concat_layer3)
        
        conv_layer8 = self.relu(self.e_conv_layer8(concat_layer3))
        
        # 去雾公式
        dehaze_image = self.relu((conv_layer8 * img) - conv_layer8 + 1)
        
        return dehaze_image