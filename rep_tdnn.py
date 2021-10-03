#!/usr/bin/python
# encoding: utf-8
'''
    this model use conv relu norm
    Rep-TDNN with AAM-Softmax
'''
from torch import nn
import torch.nn.functional as F
import torch
import math
import numpy as np


# ArcFace
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.25, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def update_param(self):
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m


class SE_BlOCK(torch.nn.Module):
    '''
    SE-attention;
    '''
    def __init__(self, channel, reduction=8,):
        super(SE_BlOCK, self).__init__()
        self.TDNN_MLP_1 = torch.nn.Conv1d(channel, channel//reduction, 1, groups=1)#reduction)
        self.TDNN_MLP_2 = torch.nn.Conv1d(channel//reduction, channel, 1, groups=1)#reduction)
        self.relu = torch.nn.LeakyReLU(0.2)
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        module_input = x
        std = torch.mean(x, dim=2)
        std = std.unsqueeze(2)
        std = self.TDNN_MLP_1(std)
        std = self.relu(std)

        std = self.TDNN_MLP_2(std)
        std = self.relu(std)

        mask = self.activate(std)

        return module_input+module_input*mask

    def statisitc_plooing(self, x):

        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        out = torch.cat([mean, std], dim=1)
        #  在batch内加一个dim 为了TDNN
        out = out.unsqueeze(2)
        # piandu
        # pian = mean*mean*mean
        return out


class CsRepTDNNBlock(nn.Module):
    '''
    The proposed CS-Rep block. To applying the CS-Rep simply, the branches of the sequential-layer does not
     loading into the sequential of torch.
    Sequential-TDNN-layer: bn(activation(TDNN(3) + TDNN(1) + x))
    '''
    def __init__(self, in_channels=512, inlayer_channles=512, out_channels=512, head_frame_layer_context=3, groups=8):
        super(CsRepTDNNBlock, self).__init__()
        # params of the net
        self.in_channels = in_channels
        self.inlayer_channles = inlayer_channles
        self.out_channels = out_channels
        self.head_frame_layer_context = head_frame_layer_context
        self.groups = groups

        # activation
        self.activation = torch.nn.LeakyReLU(0.2)
        # head TDNN layer
        self.conv0 = torch.nn.Conv1d(self.in_channels, self.out_channels, \
                                     self.head_frame_layer_context, dilation=1, padding=0)
        self.bn0 = torch.nn.BatchNorm1d(self.out_channels)

        # Sequential-TDNN-layer: bn(activation(TDNN(3) + TDNN(1) + x))
        self.conv1 = torch.nn.Conv1d(self.out_channels, self.inlayer_channles, 3, dilation=1, padding=1, groups=self.groups)
        self.conv1_0 = torch.nn.Conv1d(self.out_channels, self.inlayer_channles, 1, dilation=1, padding=0, groups=self.groups)
        self.bn1 = torch.nn.BatchNorm1d(self.inlayer_channles)

        self.conv2 = torch.nn.Conv1d(self.out_channels, self.inlayer_channles, 3, dilation=1, padding=1, groups=self.groups)
        self.conv2_0 = torch.nn.Conv1d(self.out_channels, self.inlayer_channles, 1, dilation=1, padding=0, groups=self.groups)
        self.bn2 = torch.nn.BatchNorm1d(self.inlayer_channles)

        self.conv3 = torch.nn.Conv1d(self.out_channels, self.inlayer_channles, 3, dilation=1, padding=1, groups=self.groups)
        self.conv3_0 = torch.nn.Conv1d(self.out_channels, self.inlayer_channles, 1, dilation=1, padding=0, groups=self.groups)
        self.bn3 = torch.nn.BatchNorm1d(self.inlayer_channles)

        self.conv4 = torch.nn.Conv1d(self.out_channels, self.inlayer_channles, 3, dilation=1, padding=1, groups=self.groups)
        self.conv4_0 = torch.nn.Conv1d(self.out_channels, self.inlayer_channles, 1, dilation=1, padding=0, groups=self.groups)
        self.bn4 = torch.nn.BatchNorm1d(self.inlayer_channles)
        self.rep_layer1 = None

        self.se = SE_BlOCK(self.out_channels)

    def forward(self, x):

        # rep topology
        if self.rep_layer1 is not None:
            frame0 = self.activation(self.conv0(x))
            frame1 = self.activation(self.rep_layer1(frame0))
            frame2 = self.activation(self.rep_layer2(frame1))
            frame3 = self.activation(self.rep_layer3(frame2))
            frame4 = self.bn4(self.activation(self.rep_layer4(frame3)))
            frame4 = self.se(frame4)
            return frame4

        # original topology
        frame0 = self.bn0(self.activation(self.conv0(x)))
        frame1 = self.bn1(self.activation(self.conv1(frame0) + self.conv1_0(frame0) + frame0))
        frame2 = self.bn2(self.activation(self.conv2(frame1) + self.conv2_0(frame1) + frame1))
        frame3 = self.bn3(self.activation(self.conv3(frame2) + self.conv3_0(frame2) + frame2))
        frame4 = self.bn4(self.activation(self.conv4(frame3) + self.conv4_0(frame3) + frame3))
        frame4 = self.se(frame4)
        return frame4

    def bn_first_merge(self, conv, bn):
        '''
        key step used to bn-first rep
        '''

        conv = self.__getattr__(conv)
        bn = self.__getattr__(bn)

        # param of conv
        kernel0 = conv.weight
        bias0 = conv.bias
        kernel0.requires_grad = False
        # param of BN
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        '''
        kernal
        '''

        t = (gamma / std).reshape(1, -1, 1)
        # rep for each kernel of groups
        new_kernel_list = []
        for i in range(0, self.groups):
            one_groups_inchannel = self.out_channels//self.groups
            # The formula of bn-first-rep is displayed in our paper.
            new_kernel_onegroups = kernel0[int(i*one_groups_inchannel):int((i+1)*one_groups_inchannel)] *\
                                   t[:, int(i*one_groups_inchannel):int((i+1)*one_groups_inchannel)]
            new_kernel_list.append(new_kernel_onegroups)
        new_kernel = torch.cat(new_kernel_list, dim=0)


        '''
        bias
        '''
        z = (beta - running_mean * gamma / std)
        z = z.reshape(z.shape[0], -1)

        new_bias_list = []
        for i in range(0, self.groups):
            one_groups_inchannel = self.out_channels // self.groups
            new_bias_onegroups = kernel0[int(i*one_groups_inchannel):int((i+1)*one_groups_inchannel)] *\
                        z[int(i*one_groups_inchannel):int((i+1)*one_groups_inchannel)]
            # calculate the influence of BN for weight by convolution operation.
            new_bias_onegroups = torch.sum(new_bias_onegroups, dim=1)
            new_bias_onegroups = torch.sum(new_bias_onegroups, dim=1)

            if bias0 is not None:
                new_bias_onegroups += bias0[int(i*one_groups_inchannel):int((i+1)*one_groups_inchannel)]
            new_bias_list.append(new_bias_onegroups)

        new_bias = torch.cat(new_bias_list, dim=0)

        # finished new weight and bias
        return new_kernel, new_bias

    def make_id_layer_conv(self, in_channels, out_channels, kernel_size, groups):
        '''
        Builds a identity TDNN layer form a  shortcut-layer
        '''

        groups_in_channels = in_channels // groups
        kernel_value = np.zeros((out_channels, groups_in_channels, 3), dtype=np.float32)
        '''
        egs:
        [
        [[0,1,0],
        [0,0,0],
        [0,0,0]],
        
        [[0,0,0],
        [0,1,0],
        [0,0,0]],
        
        [[0,0,0],
        [0,0,0],
        [0,0,1]],
        ]
        
        '''
        for i in range(out_channels):
            kernel_value[i, i % groups_in_channels, 1] = 1

        id_tensor = torch.from_numpy(kernel_value)
        self.id_layer = torch.nn.Conv1d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size, stride=1,
                                   padding=1, dilation=1,
                                   groups=groups, bias=False)
        self.id_layer.weight.data = id_tensor

    def _pad_Cx1_to_Cx3_tensor(self, kernelCx1):
        if kernelCx1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernelCx1, [1,1,0,0])


    def rep_all(self):

        for i in range(1,5):
            self.id_layer = None
            self.__delattr__('id_layer')
            # building a identity TDNN layer

            self.make_id_layer_conv(self.out_channels, self.out_channels, 3, self.groups)
            # C*3 cs-rep
            conv1bn0_kernel, conv1bn0_bias = self.bn_first_merge('conv{}'.format(i), 'bn{}'.format(i-1))

            # C*1 padding
            self.__getattr__('conv{}_0'.format(i)).weight.data = self._pad_Cx1_to_Cx3_tensor(self.__getattr__('conv{}_0'.format(i)).weight.data)
            # C*1 cs-rep
            conv1_0_bn0_kernel, conv1_0_bn0_bias = self.bn_first_merge('conv{}_0'.format(i), 'bn{}'.format(i-1))

            # id-layer cs-rep
            id_kernel, id_bias = self.bn_first_merge('id_layer', 'bn{}'.format(i-1))

            # merged layer
            rep_kernel = torch.nn.Conv1d(in_channels=self.out_channels,
                            out_channels=self.out_channels,
                            kernel_size=3, stride=1,
                            padding=1, dilation=1,
                            groups=self.groups, bias=True)

            rep_kernel.weight.data = conv1bn0_kernel + conv1_0_bn0_kernel + id_kernel
            rep_kernel.bias.data = conv1bn0_bias + conv1_0_bn0_bias + id_bias
            self.__setattr__('rep_layer{}'.format(i), rep_kernel)

        for i in range(0,4):
            self.__delattr__('bn{}'.format(i))
        for i in range(1,5):
            self.__delattr__('conv{}'.format(i))
            self.__delattr__('conv{}_0'.format(i))
        if self.id_layer is not None:
            self.__delattr__('id_layer')



class Xtractor(torch.nn.Module):
    def __init__(self,):
        super(Xtractor, self).__init__()

        self.out_channel = 512
        self.embedding_dim = 512
        self.frame_layer = self._make_layer(161, self.out_channel, self.out_channel, [3,1,1,5])

        self.frame_conv4 = torch.nn.Conv1d(self.out_channel, self.out_channel, 1)
        self.frame_conv5 = torch.nn.Conv1d(self.out_channel, 3 * self.out_channel, 1)

        self.seg_lin0 = torch.nn.Linear(3 * self.out_channel * 2, self.embedding_dim)
        self.seg_lin1 = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

        self.norm4 = torch.nn.BatchNorm1d(self.out_channel)
        self.norm5 = torch.nn.BatchNorm1d(self.out_channel*3)
        self.norm6 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.norm7 = torch.nn.BatchNorm1d(self.embedding_dim)

        self.activation = torch.nn.LeakyReLU(0.2)

    def _make_layer(self, inchannel, inblock_channel, outchannel, context_list, stride=1):
        layers = []
        layers.append(CsRepTDNNBlock(inchannel,inblock_channel, inblock_channel,context_list[0]))

        for context in context_list[1:]:
            layers.append(CsRepTDNNBlock(inblock_channel,inblock_channel,outchannel, context))
        return nn.Sequential(*layers)

    def rep_all(self):
        for layer in self.frame_layer:
            layer.rep_all()

    def forward(self, x):
        x = self.frame_layer(x)
        # tdnn 4
        frame_emb_4 = self.norm4(self.activation(self.frame_conv4(x)))
        # tdnn 5
        frame_emb_5 = self.norm5(self.activation(self.frame_conv5(frame_emb_4)))

        mean = torch.mean(frame_emb_5, dim=2)
        std = torch.std(frame_emb_5, dim=2)
        seg_emb_0 = torch.cat([mean, std], dim=1)

        seg_emb_1 = self.norm6(self.activation(self.seg_lin0(seg_emb_0)))
        seg_emb_2 = self.norm7(self.activation(self.seg_lin1(seg_emb_1)))
        result = seg_emb_2

        return result, result


class Xvector(nn.Module):
    def __init__(self, n_classes=1211, embedding_size=512, s=30, m=0.25):
        super(Xvector, self).__init__()
        self.embedding_net = Xtractor()
        self.arc = ArcMarginProduct(embedding_size, n_classes, s=s, m=m)


    def forward(self, x, target=None):
        out, embedding = self.embedding_net(x)

        if target is None:
            return out, embedding
        else:
            m_logits=self.arc(embedding, target)
            return m_logits, embedding

