import torch.nn as nn
import torch
import torch.nn.functional as F
import math


# 初始化模型参数
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# 空间注意力
class PALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


# 通道注意力
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


# SPP模块
class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='avg_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        global tensor, x_flatten
        num, c, h, w = x.size()
        # print('original', x.size())
        level = 1
        #         print(x.size())
        for i in range(self.num_levels):
            if i >= 1:
                level <<= 1
            '''
            The equation is explained on the following site:
            http://www.cnblogs.com/marsggbo/p/8572846.html#autoid-0-0-0
            '''
            kernel_size = (math.ceil(h / level), math.ceil(w / level))  # kernel_size = (h, w)
            padding = (
                math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            zero_pad = torch.nn.ZeroPad2d((padding[1], padding[1], padding[0], padding[0]))
            x_new = zero_pad(x)

            # update kernel and stride
            h_new, w_new = x_new.size()[2:]

            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))

            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
            elif self.pool_type == 'avg_pool':
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)

            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)

        return x_flatten


class net(nn.Module):
    """
      Model
    """

    def __init__(self):
        """

        """
        super(net, self).__init__()
        print('***************模型5加载完毕******************')
        kernel_size = 3
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size, bias=True, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size, bias=True, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size, bias=True, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size, bias=True, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size, bias=True, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size, bias=True, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size, bias=True, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size, bias=True, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size, bias=True, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        #  注意力
        self.stage1_ca = CALayer(192)
        self.stage1_pa = PALayer(192)
        self.stage2_ca = CALayer(384)
        self.stage2_pa = PALayer(384)
        self.stage3_ca = CALayer(768)
        self.stage3_pa = PALayer(768)

        self.SPP = SPPLayer(1, pool_type='avg_pool')
        self.fc = nn.Sequential(
            nn.Linear(1344, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

    def extract_features(self, x):
        """
        feature extraction
        :param x: the input image
        :return: the output feature
        """

        features_1 = self.stage1(x)  # B*64*H*W

        x = self.maxpool(features_1)
        features_2 = self.stage2(x)  # B*128*H/2*W/2

        x = self.maxpool(features_2)
        features_3 = self.stage3(x)  # B*256*H/4*W/4

        return features_1, features_2, features_3

    def forward(self, data):
        """
        :param data:  degraded and Restorted images
        :return: quality of images
        """
        ref, x = data
        x_stage1, x_stage2, x_stage3 = self.extract_features(x)
        ref_stage1, ref_stage2, ref_stage3 = self.extract_features(ref)

        #  多尺度差异特征融合
        features_1 = torch.cat((x_stage1 - ref_stage1, x_stage1, ref_stage1), 1)
        features_2 = torch.cat((x_stage2 - ref_stage2, x_stage2, ref_stage2), 1)
        features_3 = torch.cat((x_stage3 - ref_stage3, x_stage3, ref_stage3), 1)
        features_1 = self.stage1_ca(features_1)
        features_1 = self.stage1_pa(features_1)
        features_2 = self.stage2_ca(features_2)
        features_2 = self.stage2_pa(features_2)
        features_3 = self.stage3_ca(features_3)
        features_3 = self.stage3_pa(features_3)
        features_1 = self.SPP(features_1)
        features_2 = self.SPP(features_2)
        features_3 = self.SPP(features_3)
        features = torch.cat((features_1, features_2, features_3), 1)  # B*1344

        return self.fc(features)


if __name__ == '__main__':
    model = net()
    # model.apply(weight_init)
    # input = torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256)
    # print(type(input))
    # x = model(input)
    # print(x)
    # print(model)
    # # summary(model, (2, 3, 256, 256))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    pass
