import torch

from torch import nn
import torch.nn.functional as F
import math


import numpy as np



class Basic3x3(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic3x3, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, stride=1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class Basic1x1(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic1x1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, stride=1, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv1(x)
        out = self.tanh(out)
        return out


class Convlutioanl(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convlutioanl, self).__init__()
        self.padding = (2, 2, 2, 2)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=5, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = F.pad(input, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Convlutioanl_out(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convlutioanl_out, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1)
        self.tanh = nn.Tanh()

    def forward(self, input):
        out = self.conv(input)
        out = self.tanh(out)
        return out

# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate,kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.bn = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # out = F.relu(self.conv(x))
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = torch.cat((x, out), 1)
        return out


# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out



class NormLayer(nn.Module):

    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x
        else:
            assert 1 == 0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ReluLayer(nn.Module):

    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x
        else:
            assert 1 == 0, 'Relu type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', norm_type='none', relu_type='none',
                 use_pad=True):
        super(ConvLayer, self).__init__()
        self.use_pad = use_pad

        bias = True if norm_type in ['pixel', 'none'] else False
        stride = 2 if scale == 'down' else 1

        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    ------------------
    # Args
        - hg_depth: depth of HourGlassBlock. 0: don't use attention map.
        - use_pmask: whether use previous mask as HourGlassBlock input.
    """

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, att_name='spar'):
        super(ResidualBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        self.preact_func = nn.Sequential(
            NormLayer(c_in, norm_type=self.norm_type),
            ReluLayer(c_in, self.relu_type),
        )

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        self.conv1 = ConvLayer(c_in, c_out, 3, scales[0], **kwargs)
        self.conv2 = ConvLayer(c_out, c_out, 3, scales[1], norm_type=norm_type, relu_type='none')

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func = HourGlassBlock(self.hg_depth, c_out, c_attn, **kwargs)

    def forward(self, x):
        identity = self.shortcut_func(x)
        out = self.preact_func(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = identity + self.att_func(out)
        return out


class HourGlassBlock(nn.Module):

    def __init__(self, depth, c_in, c_out,
                 c_mid=64,
                 norm_type='bn',
                 relu_type='prelu',
                 ):
        super(HourGlassBlock, self).__init__()
        self.depth = depth
        self.c_in = c_in
        self.c_mid = c_mid
        self.c_out = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if self.depth:
            self._generate_network(self.depth)
            self.out_block = nn.Sequential(
                ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
                nn.Sigmoid()
            )

    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs))
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs))

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x, pmask=None):
        if self.depth == 0: return x
        input_x = x
        x = self._forward(self.depth, x)
        self.att_map = self.out_block(x)
        x = input_x * self.att_map
        return x


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init, 这个不参与反向传播
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))


    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


class MultiSpectralAttentionLayer(torch.nn.Module):

    " reduction=16, freq_sel_method='top16' "

    def __init__(self, channel, dct_h, dct_w, reduction=32, freq_sel_method='top32'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        # 关键部分，全文精华！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        # 关键部分，全文精华！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)



class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class ConvLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        return x


class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x


class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)

        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)



class BasicLayer(nn.Module):
    def __init__(self, in_channel, out_channel, dct_h, dct_w, reduction = 16, depth = 2 , relu_type='leakyrelu', norm_type='bn'):
        super(BasicLayer, self).__init__()
        nrargs = {'norm_type': norm_type, 'relu_type': relu_type}
        self.SAM = ResidualBlock(in_channel, out_channel, hg_depth=depth, **nrargs)
        self.CAM = MultiSpectralAttentionLayer(in_channel, dct_w, dct_h, reduction, freq_sel_method='top16')
        self.GAM = RGBD(in_channels=in_channel,out_channels=out_channel)

    def forward(self, x):
        x1 = self.SAM(x)
        x2 = self.CAM(x)
        x3 = self.GAM(x)
        x4 = torch.cat((x1,x2,x3),dim=1)
        return x4


class MODEL(nn.Module):
    def __init__(self,in_channel=2, out_channel=16, output_channel=1, dct_h=7, dct_w=7):
        super(MODEL, self).__init__()
        self.convInput = Basic3x3(in_channel, out_channel)
        self.conv = Basic3x3(out_channel*3, out_channel*3)
        self.Rconv = RDB(out_channel,3,16)
        self.convolutional_out = Basic1x1(out_channel*3, output_channel)
        self.basicLayer = BasicLayer(in_channel=out_channel, out_channel=out_channel, dct_h=dct_h, dct_w=dct_w)

    def forward(self, input):
        convInput = self.convInput(input)
        layer1 = self.Rconv(convInput)
        bl1 = self.basicLayer(layer1)
        layer2 = self.Rconv(self.Rconv(convInput))
        bl2 = self.basicLayer(layer2)
        layer3 = self.Rconv(self.Rconv(self.Rconv(convInput)))
        bl3 = self.basicLayer(layer3)
        add = bl1 + bl2 + bl3
        out = self.convolutional_out(add)
        return out


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y
