# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn

class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out, chunk_length,
                 filter_widths, causal, dropout, channels):
                 
        super().__init__()
        
        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'
        
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.chunk=chunk_length
        self.pad = [ filter_widths[0] // 2 ]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out*3, 1)


    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
            
    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2*frames
    
    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames
        
    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
        
        sz = x.shape[:3]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = self._forward_blocks(x)
        x = x.permute(0, 2, 1)
        # x=torch.reshape(x,(sz[0],self.chunk,self.num_joints_out, 3))
        x=torch.reshape(x,(sz[0],-1,self.num_joints_out, 3))
        
        # x = x.view(sz[0], -1, self.num_joints_out, 3)
        
        return x    

class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out, chunk_length,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, chunk_length,filter_widths, causal, dropout, channels)
        
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
      
        return x
    
class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.
    
    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out, chunk_length,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(num_joints_in, in_features, num_joints_out, chunk_length, filter_widths, causal, dropout, channels)
        
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], stride=filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0] // 2) if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        return x

        #############################################################################################


        # -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
# from lib.models.attention import SelfAttention

import torch
from torch import nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        m.bias.data.fill_(0.01)

class SelfAttention(nn.Module):
    def __init__(self, attention_size,
                 batch_first=False,
                 layers=1,
                 dropout=.0,
                 non_linearity="relu"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)
        self.attention.apply(init_weights) 
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, inputs):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        # representations = weighted.sum(1).squeeze()
        representations = weighted.sum(1).squeeze()
        return representations, scores

class MotionDiscriminator(nn.Module):

    def __init__(self,
                 rnn_size=1024,
                 input_size=51,
                 num_layers=1,
                 output_size=1,
                 feature_pool="concat",
                 use_spectral_norm=False,
                 attention_size=1024,
                 attention_layers=1,
                 attention_dropout=0.5):

        super(MotionDiscriminator, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)

        linear_size = self.rnn_size if not feature_pool == "concat" else self.rnn_size * 2

        if feature_pool == "attention" :
            self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
        if use_spectral_norm:
            self.fc = spectral_norm(nn.Linear(linear_size, output_size))
        else:
            self.fc = nn.Linear(linear_size, output_size)

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, input_size = sequence.shape
        sequence = torch.transpose(sequence, 0, 1)

        outputs, state = self.gru(sequence)

        if self.feature_pool == "concat":
            outputs = F.relu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            output = self.fc(torch.cat([avg_pool, max_pool], dim=1))
        elif self.feature_pool == "attention":
            outputs = outputs.permute(1, 0, 2)
            y, attentions = self.attention(outputs)
            output = self.fc(y)
        else:
            output = self.fc(outputs[-1])

        return output

class MotionDiscriminatorKCS(nn.Module):

    def __init__(self,
                 rnn_size=1024,
                 input_size=51,
                 num_layers=1,
                 output_size=1,
                 feature_pool="concat",
                 use_spectral_norm=False,
                 attention_size=1024,
                 attention_layers=1,
                 attention_dropout=0.5):

        super(MotionDiscriminatorKCS, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        self.gru1 = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)
        self.gru2 = nn.GRU(16*16, self.rnn_size, num_layers=num_layers)
        self.leakyrelu=nn.LeakyReLU()

        self.Ct = torch.FloatTensor([
            [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0],
            [ 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
            [ 0, 0, 0, 0, 0,-1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0]]).cuda()
        
        
        linear_size = self.rnn_size if not feature_pool == "concat" else self.rnn_size * 2

        if feature_pool == "attention" :
            self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
        if use_spectral_norm:
            self.fc = spectral_norm(nn.Linear(linear_size, output_size))
        else:
            self.fc1 = nn.Linear(linear_size, 1000)
            self.fc2 = nn.Linear(1000, 1000)
            self.fc3 = nn.Linear(1000, 1000)
            self.fc4 = nn.Linear(1000, 1)
            self.fc5 = nn.Linear(linear_size, 1000)
            self.fc6 = nn.Linear(1000, 1000)
            self.fc7 = nn.Linear(1000, 1000)
            self.fc8 = nn.Linear(2000, 100)  
            self.fc9 = nn.Linear(100, 1)           

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, num_joints, 3]
        
        """

        batchsize, seqlen, num_joints, _ = sequence.shape
        sequence = torch.transpose(sequence, 0, 1)
        sequence = torch.transpose(sequence, -1, -2)
        C  = self.Ct.repeat(seqlen, batchsize, 1,1)


        B  = torch.matmul(sequence, C)
        Bt = B.transpose(-1,-2)
        psi= torch.matmul(Bt,B)
        # psi = torch.diagonal(psi,dim1=-2,dim2=-1)
        sequence = sequence.reshape(sequence.shape[0], sequence.shape[1], -1)
        psi = psi.view(psi.shape[0], psi.shape[1], -1)
        
        
        # sequence=torch.cat([sequence, psi],dim=-1)
        outputs, state = self.gru1(sequence)
        # outputs_KCS, state_KCS = self.gru2(psi)





        if self.feature_pool == "concat":
            outputs = F.leaky_relu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            # output = F.relu(self.fc1(torch.cat([avg_pool, max_pool], dim=1)))
            x = torch.cat([avg_pool, max_pool], dim=1)

            x  = self.leakyrelu(self.fc1(x))
            x1 = self.leakyrelu(self.fc2(x))
            x  = self.leakyrelu(self.fc3(x1)+x)
            x  = self.fc4(x)
             
            # outputs_KCS = F.leaky_relu(outputs_KCS)
            # avg_pool = F.adaptive_avg_pool1d(outputs_KCS.permute(1, 2, 0), 1).view(batchsize, -1)
            # max_pool = F.adaptive_max_pool1d(outputs_KCS.permute(1, 2, 0), 1).view(batchsize, -1)
            # # output_KCS = F.relu(self.fc2(torch.cat([avg_pool, max_pool], dim=1)))
            # x_psi  = torch.cat([avg_pool, max_pool], dim=1)

            # x_psi  = self.leakyrelu(self.fc5(x_psi))
            # x1_psi = self.leakyrelu(self.fc6(x_psi))
            # x_psi  = self.fc7(x1_psi)+x_psi

            # x_last = torch.cat([x,x_psi],dim=1)               
            # x_last = self.leakyrelu(self.fc8(x_last))
            # x_last = self.fc9(x_last)

        elif self.feature_pool == "attention":
            outputs = outputs.permute(1, 0, 2)
            y, attentions = self.attention(outputs)
            output = self.fc(y)
        else:
            output = self.fc(outputs[-1])

        return x


class MotionDiscriminator_CNN(nn.Module):

    def __init__(self, chunk_length):

        super(MotionDiscriminator_CNN, self).__init__()
        self.chunk_length = chunk_length
        
        
        # self.batchborms=self.ModuleList([nn.BatchNorm1d(chunk_length*i) for i in range(2,2,5)])
        self.conv1 = nn.Conv1d(self.chunk_length, self.chunk_length*2, kernel_size=3)
        self.conv2 = nn.Conv1d(self.chunk_length*2, self.chunk_length*4, kernel_size=3)
        self.maxpool=nn.MaxPool1d(2)
        #self.conv3 = nn.Conv1d(self.chunk_length*4, self.chunk_length*8, kernel_size=3)
        #self.conv4 = nn.Conv1d(self.chunk_length*8, self.chunk_length*8, kernel_size=3)
        self.fc1=nn.Linear(11*self.chunk_length*4,128)
        self.fc2=nn.Linear(128,1)
        #self.batchnorm1=nn.BatchNorm1d(chunk_length)
        self.batchnorm2=nn.BatchNorm1d(self.chunk_length*2)
        self.batchnorm3=nn.BatchNorm1d(self.chunk_length*4)


    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, input_size = sequence.shape
        # sequence = torch.transpose(sequence, 0, 1)
      
       # x=self.batchnorm1(sequence)
        x=self.maxpool(self.batchnorm2(F.relu(self.conv1(sequence))))
        x=self.maxpool(self.batchnorm3(F.relu(self.conv2(x))))
       # x=self.maxpool(x)
       # x=self.batchnorm2(F.relu(self.conv3(x)))
       # x=self.batchnorm1(F.relu(self.conv4(x)))
        x=x.view(-1,11*self.chunk_length*4)
        x=F.relu(self.fc1(x))
        output=self.fc2(x)

        return output

class MotionDiscriminator_RepNet(nn.Module):

    def __init__(self, chunk_length):

        super(MotionDiscriminator_RepNet,self).__init__()
        
        self.chunk_length=chunk_length
        
        # self.batchborms=self.ModuleList([nn.BatchNorm1d(chunk_length*i) for i in range(2,2,5)])
        self.fc1=nn.Linear(chunk_length*51,chunk_length*100)
        self.fc2=nn.Linear(chunk_length*100,chunk_length*100)
        self.fc3=nn.Linear(chunk_length*100,chunk_length*100)
        self.fc4=nn.Linear(chunk_length*51,chunk_length*100)
        self.fc5=nn.Linear(chunk_length*100,chunk_length*100)
        self.fc6=nn.Linear(chunk_length*100,chunk_length*100)
        self.fc7=nn.Linear(chunk_length*200,chunk_length*100)
        self.fc7=nn.Linear(100,1)

        self.leakyrelu=nn.LeakyReLU()


    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, input_size = sequence.shape
        
        # sequence = torch.transpose(sequence, 0, 1)
        
        x1 = sequence[:,:,0:51]
        x1=x1.contiguous()
        x1 = x1.view(-1,self.chunk_length*51)
        x1 = self.fc1(x1)
        x1 = self.leakyrelu(x1)
        x1 = self.fc2(x1)
        x1 = self.leakyrelu(x1)
        x1 = self.fc3(x1)
        x11 = self.leakyrelu(x1)
       
        
        
        x2=sequence[:,:,51:102]
        x2=x2.contiguous()
        x2 = x2.view(-1,self.chunk_length*51)
        x2 = self.fc4(x2)
        x2 = self.leakyrelu(x2)
        x2 = self.fc5(x2)
        x2 = self.leakyrelu(x2)
        x2 = self.fc6(x2)
        x22 = self.leakyrelu(x2)
        

        x3=torch.cat((x11,x22), dim=1)
        x3=self.leakyrelu(self.fc7(x3))
        x3=self.fc8(x3)


        return x3


class MotionDiscriminator_RepNet2(nn.Module):

    def __init__(self, chunk_length):

        super(MotionDiscriminator_RepNet2,self).__init__()
        
        self.chunk_length=chunk_length
        # self.batchborms=self.ModuleList([nn.BatchNorm1d(chunk_length*i) for i in range(2,2,5)])
        self.fc1=nn.Linear(self.chunk_length*51,self.chunk_length*100)
        self.fc2=nn.Linear(self.chunk_length*100,self.chunk_length*100)
        self.fc3=nn.Linear(self.chunk_length*100,100)
        self.fc4=nn.Linear(100,1)

        self.leakyrelu=nn.LeakyReLU()


    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, input_size = sequence.shape
        
        # sequence = torch.transpose(sequence, 0, 1)
        
        x1 = sequence
        x1=x1.contiguous()
        x1 = x1.view(-1,self.chunk_length*51)
        x1 = self.fc1(x1)
        x1 = self.leakyrelu(x1)
        x1 = self.fc2(x1)
        x1 = self.leakyrelu(x1)
        x1 = self.fc3(x1)
        x1 = self.leakyrelu(x1)
        x1=self.fc4(x1)


        return x1


class Camera_model(nn.Module):
    def __init__(self):

        super(Camera_model,self).__init__()
        
        
        # self.batchborms=self.ModuleList([nn.BatchNorm1d(chunk_length*i) for i in range(2,2,5)])
        self.fc1=nn.Linear(34,1000)
        self.fc2=nn.Linear(1000,1000)
        self.fc3=nn.Linear(1000,1000)
        self.fc4=nn.Linear(1000,1000)
        self.fc5=nn.Linear(1000,1000)
        self.fc6=nn.Linear(1000,1000)
        self.fc7=nn.Linear(1000,1000)
        self.fc8=nn.Linear(1000,6)
        self.leakyrelu=nn.LeakyReLU()


    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, input_size]
        """
        batchsize, seqlen, joint_number, input_features = sequence.shape
        
        # sequence = torch.transpose(sequence, 0, 1)
        sequence=sequence.contiguous()
        x=sequence.view(batchsize,seqlen,joint_number*input_features)
        
        x=self.fc1(x)
        x=self.leakyrelu(x)
         
        x1=self.fc2(x)
        x1=self.leakyrelu(x1)
        x1=self.fc3(x1)
        x=x+x1
        x=self.leakyrelu(x)

        x1=self.fc4(x)
        x1=self.leakyrelu(x1)
        x1=self.fc5(x1)
        x=x+x1
        x=self.leakyrelu(x)

        x1=self.fc6(x)
        x1=self.leakyrelu(x1)
        x1=self.fc7(x1)
        x=x+x1
        x=self.leakyrelu(x)
        
        x=self.fc8(x)
        return x


class Generator_RepNet(nn.Module):

    def __init__(self,seq_length,output_length=1):
        super(Generator_RepNet,self).__init__()
        
        self.input_shape=seq_length*34
        self.output_length=output_length
        self.output_shape=self.output_length*51
        self.leakyrelu=nn.LeakyReLU()
        
        self.fc1=nn.Linear(self.input_shape,1000)
        self.fc2=nn.Linear(1000,1000)
        self.fc3=nn.Linear(1000,1000)
        self.fc4=nn.Linear(1000,1000)
        self.fc5=nn.Linear(1000,1000)
        self.fc6=nn.Linear(1000,1000)
        self.fc7=nn.Linear(1000,1000)
        self.fc8=nn.Linear(1000,1000)
        self.fc9=nn.Linear(1000,self.output_shape)


    def forward(self,x):
        # print(x.shape)
        batchsize, seqlen, joint_number, input_features = x.shape
        x=x.contiguous()
        x=x.view(batchsize,seqlen*joint_number*input_features)
        # print(x.shape)
        x=self.fc1(x)
        x=self.leakyrelu(x)

        x1=self.fc2(x)
        x1=self.leakyrelu(x1)
        x1=self.fc3(x1)
        x=x+x1
        x=self.leakyrelu(x)

        x1=self.fc4(x)
        x1=self.leakyrelu(x1)
        x1=self.fc5(x)
        x=x+x1
        x=self.leakyrelu(x)

        x1=self.fc6(x)
        x1=self.leakyrelu(x1)
        x1=self.fc7(x)
        x=x+x1
        x=self.leakyrelu(x)

        x=self.fc8(x)
        x=self.leakyrelu(x)

        x=self.fc9(x)
        
        x=x.view(batchsize,self.output_length,joint_number,3)

        return x


class Generator_RepNet2(nn.Module):

    def __init__(self,seq_length,output_length=1):
        super(Generator_RepNet2,self).__init__()
        
        self.input_shape=seq_length*34
        self.output_length=output_length
        self.output_shape=self.output_length*51
        self.leakyrelu=nn.LeakyReLU()
        # print(self.input_shape)
        self.fc1=nn.Linear(self.input_shape,1000)
        self.fc2=nn.Linear(1000,1000)
        self.fc3=nn.Linear(1000,1000)
        self.fc4=nn.Linear(1000,1000)
        self.fc5=nn.Linear(1000,1000)
        self.fc6=nn.Linear(1000,1000)
        self.fc7=nn.Linear(1000,1000)
        self.fc8=nn.Linear(1000,1000)
        self.fc9=nn.Linear(1000,1000)
        self.fc10=nn.Linear(1000,1000)
        self.fc11=nn.Linear(1000,1000)
        self.fc12=nn.Linear(1000,1000)
        self.fc13=nn.Linear(1000,1000)
        self.fc14=nn.Linear(1000,1000)
        self.fc15=nn.Linear(1000,self.output_shape)


    def forward(self,x):
        # print(x.shape)
        batchsize, seqlen, joint_number, input_features = x.shape
        x=x.contiguous()
        x=x.view(batchsize,seqlen*joint_number*input_features)
        # print(self.input_shape)
        # print(x.shape)
        x=self.fc1(x)
        x=self.leakyrelu(x)

        x1=self.fc2(x)
        x1=self.leakyrelu(x1)
        x1=self.fc3(x1)
        x=x+x1
        x=self.leakyrelu(x)

        x1=self.fc4(x)
        x1=self.leakyrelu(x1)
        x1=self.fc5(x)
        x=x+x1
        x=self.leakyrelu(x)

        x1=self.fc6(x)
        x1=self.leakyrelu(x1)
        x1=self.fc7(x)
        x=x+x1
        x=self.leakyrelu(x)

        x1=self.fc8(x)
        x1=self.leakyrelu(x1)
        x1=self.fc9(x)
        x=x+x1
        x=self.leakyrelu(x)

        x1=self.fc10(x)
        x1=self.leakyrelu(x1)
        x1=self.fc11(x)
        x=x+x1
        x=self.leakyrelu(x)

        x1=self.fc12(x)
        x1=self.leakyrelu(x1)
        x1=self.fc13(x)
        x=x+x1
        x=self.leakyrelu(x)

        x=self.fc14(x)
        x=self.leakyrelu(x)

        x=self.fc15(x)
        
        x=x.view(batchsize,self.output_length,joint_number,3)

        return x


class Generator_RepNet3(nn.Module):

    def __init__(self,seq_length,output_length=1):
        super(Generator_RepNet3,self).__init__()
        
        self.input_shape=seq_length*34
        self.output_length=output_length
        self.output_shape=self.output_length*51
        self.leakyrelu=nn.LeakyReLU()
        
        self.fc1=nn.Linear(self.input_shape,1000)
        self.fc2=nn.Linear(1000,1000)
        self.fc3=nn.Linear(1000,1000)
        # self.fc4=nn.Linear(1000,1000)
        # self.fc5=nn.Linear(1000,1000)
        # self.fc6=nn.Linear(1000,1000)
        # self.fc7=nn.Linear(1000,1000)
        # self.fc8=nn.Linear(1000,1000)
        # self.fc9=nn.Linear(1000,1000)
        # self.fc10=nn.Linear(1000,1000)
        # self.fc11=nn.Linear(1000,1000)
        # self.fc12=nn.Linear(1000,1000)
        # self.fc13=nn.Linear(1000,1000)
        self.fc14=nn.Linear(1000,1000)
        self.fc15=nn.Linear(1000,self.output_shape)


    def forward(self,x):
        # print(x.shape)
        batchsize, seqlen, joint_number, input_features = x.shape
        x=x.contiguous()
        x=x.view(batchsize,seqlen*joint_number*input_features)
        # print(x.shape)
        x=self.fc1(x)
        x=self.leakyrelu(x)

        x1=self.fc2(x)
        x1=self.leakyrelu(x1)
        x1=self.fc3(x1)
        x=x+x1
        x=self.leakyrelu(x)

        x=self.fc14(x)
        x=self.leakyrelu(x)

        x=self.fc15(x)
        
        x=x.view(batchsize,self.output_length,joint_number,3)

        return x


class Generator_GRU(nn.Module):

    def __init__(self,
                 rnn_size=1024,
                 input_size=34,
                 num_layers=2,
                 output_length=1,
                 feature_pool="concat",
                 use_spectral_norm=False,
                 attention_size=1024,
                 attention_layers=1,
                 attention_dropout=0.5):

        super(Generator_GRU, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        self.output_length=output_length
        # self.output_shape=self.output_length*45
        self.output_shape=self.output_length*2

        self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)

        linear_size = self.rnn_size if not feature_pool == "concat" else self.rnn_size * 2

        if feature_pool == "attention" :
            self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
        # if use_spectral_norm:
        #     self.fc = spectral_norm(nn.Linear(linear_size, output_size))
        # else:
        #     self.fc = nn.Linear(linear_size, output_size)

        

        self.leakyrelu=nn.LeakyReLU()
        self.fc1=nn.Linear(linear_size,1000)
        self.fc2=nn.Linear(1000,1000)
        self.fc3=nn.Linear(1000,1000)
        self.fc4=nn.Linear(1000,1000)
        self.fc5=nn.Linear(1000,1000)
        self.fc6=nn.Linear(1000,1000)
        self.fc7=nn.Linear(1000,1000)
        self.fc8=nn.Linear(1000,1000)
        self.fc9=nn.Linear(1000,self.output_shape)

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, joint_number, 2]
        """
        batchsize, seqlen, joint_number, _ = sequence.shape
        sequence=sequence.contiguous()
        sequence=sequence.view(batchsize,seqlen,joint_number*2)
        sequence = torch.transpose(sequence, 0, 1)

        outputs, state = self.gru(sequence)
        if self.feature_pool == "concat":
            outputs = self.leakyrelu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            x = torch.cat([avg_pool, max_pool], dim=1)

            x=self.fc1(x)
            x=self.leakyrelu(x)

            x1=self.fc2(x)
            x1=self.leakyrelu(x1)
            x1=self.fc3(x1)
            x=x+x1
            x=self.leakyrelu(x)

            x1=self.fc4(x)
            x1=self.leakyrelu(x1)
            x1=self.fc5(x)
            x=x+x1
            x=self.leakyrelu(x)

            # x1=self.fc6(x)
            # x1=self.leakyrelu(x1)
            # x1=self.fc7(x)
            # x=x+x1
            # x=self.leakyrelu(x)

            x=self.fc8(x)
            x=self.leakyrelu(x)

            x=self.fc9(x)
            # output=x.view(batchsize,self.output_length,joint_number,3)
            output=x.view(batchsize,self.output_length,1,2)

        elif self.feature_pool == "attention":
            outputs = outputs.permute(1, 0, 2)
            y, attentions = self.attention(outputs)
            # output = self.fc(y)
            x=self.fc1(y)
            x=self.leakyrelu(x)

            x1=self.fc2(x)
            x1=self.leakyrelu(x1)
            x1=self.fc3(x1)
            x=x+x1
            x=self.leakyrelu(x)

            x1=self.fc4(x)
            x1=self.leakyrelu(x1)
            x1=self.fc5(x)
            x=x+x1
            x=self.leakyrelu(x)

            # x1=self.fc6(x)
            # x1=self.leakyrelu(x1)
            # x1=self.fc7(x)
            # x=x+x1
            # x=self.leakyrelu(x)

            x=self.fc8(x)
            x=self.leakyrelu(x)

            x=self.fc9(x)
            output=x.view(batchsize,self.output_length,joint_number,1)
        else:
            output = self.fc(outputs[-1])

        return output


class Generator_GRU_traj(nn.Module):

    def __init__(self,
                 rnn_size=1024,
                 input_size=60,
                 num_layers=1,
                 output_length=1,
                 feature_pool="concat",
                 use_spectral_norm=False,
                 attention_size=1024,
                 attention_layers=1,
                 attention_dropout=0.5):

        super(Generator_GRU_traj, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        self.output_length=output_length
        self.output_shape=self.output_length*9

        self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)

        linear_size = self.rnn_size if not feature_pool == "concat" else self.rnn_size * 2

        if feature_pool == "attention" :
            self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
        # if use_spectral_norm:
        #     self.fc = spectral_norm(nn.Linear(linear_size, output_size))
        # else:
        #     self.fc = nn.Linear(linear_size, output_size)

        

        self.leakyrelu=nn.LeakyReLU()
        self.fc1=nn.Linear(linear_size,1000)
        self.fc2=nn.Linear(1000,1000)
        self.fc3=nn.Linear(1000,1000)
        self.fc4=nn.Linear(1000,1000)
        self.fc5=nn.Linear(1000,1000)
        self.fc6=nn.Linear(1000,1000)
        self.fc7=nn.Linear(1000,1000)
        self.fc8=nn.Linear(1000,1000)
        self.fc9=nn.Linear(1000,self.output_shape)

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, joint_number, 2]
        """
       
        batchsize, seqlen, joint_number, _ = sequence.shape
        sequence=sequence.contiguous()
        sequence=sequence.view(batchsize,seqlen,joint_number*4)
        sequence = torch.transpose(sequence, 0, 1)

        outputs, state = self.gru(sequence)

        if self.feature_pool == "concat":
            outputs = self.leakyrelu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            x = torch.cat([avg_pool, max_pool], dim=1)
           
            x=self.fc1(x)
            x=self.leakyrelu(x)

            x1=self.fc2(x)
            x1=self.leakyrelu(x1)
            x1=self.fc3(x1)
            x=x+x1
            x=self.leakyrelu(x)

            x1=self.fc4(x)
            x1=self.leakyrelu(x1)
            x1=self.fc5(x)
            x=x+x1
            x=self.leakyrelu(x)
            # x1=self.fc6(x)
            # x1=self.leakyrelu(x1)
            # x1=self.fc7(x)
            # x=x+x1
            # x=self.leakyrelu(x)

            x=self.fc8(x)
            x=self.leakyrelu(x)

            x=self.fc9(x)
            
            output=x.view(batchsize,self.output_length,1,9)
            
        elif self.feature_pool == "attention":
            outputs = outputs.permute(1, 0, 2)
            y, attentions = self.attention(outputs)
            output = self.fc(y)
        else:
            output = self.fc(outputs[-1])

        return output

class Generator_GRU_Cam(nn.Module):

    def __init__(self,
                 rnn_size=1024,
                 input_size=30,
                 num_layers=2,
                 output_length=1,
                 feature_pool="concat",
                 use_spectral_norm=False,
                 attention_size=1024,
                 attention_layers=1,
                 attention_dropout=0.5):

        super(Generator_GRU, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        self.output_length=output_length
        self.output_shape=self.output_length*9

        self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)

        linear_size = self.rnn_size if not feature_pool == "concat" else self.rnn_size * 2

        if feature_pool == "attention" :
            self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
        # if use_spectral_norm:
        #     self.fc = spectral_norm(nn.Linear(linear_size, output_size))
        # else:
        #     self.fc = nn.Linear(linear_size, output_size)

        

        self.leakyrelu=nn.LeakyReLU()
        self.fc1=nn.Linear(linear_size,1000)
        self.fc2=nn.Linear(1000,1000)
        self.fc3=nn.Linear(1000,1000)
        self.fc4=nn.Linear(1000,1000)
        self.fc5=nn.Linear(1000,1000)
        self.fc6=nn.Linear(1000,1000)
        self.fc7=nn.Linear(1000,1000)
        self.fc8=nn.Linear(1000,1000)
        self.fc9=nn.Linear(1000,self.output_shape)

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, joint_number, 2]
        """
        batchsize, seqlen, joint_number, _ = sequence.shape
        sequence=sequence.contiguous()
        sequence=sequence.view(batchsize,seqlen,joint_number*2)
        sequence = torch.transpose(sequence, 0, 1)

        outputs, state = self.gru(sequence)

        if self.feature_pool == "concat":
            outputs = self.leakyrelu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            x = torch.cat([avg_pool, max_pool], dim=1)

            x=self.fc1(x)
            x=self.leakyrelu(x)

            x1=self.fc2(x)
            x1=self.leakyrelu(x1)
            x1=self.fc3(x1)
            x=x+x1
            x=self.leakyrelu(x)

            x1=self.fc4(x)
            x1=self.leakyrelu(x1)
            x1=self.fc5(x)
            x=x+x1
            x=self.leakyrelu(x)

            # x1=self.fc6(x)
            # x1=self.leakyrelu(x1)
            # x1=self.fc7(x)
            # x=x+x1
            # x=self.leakyrelu(x)

            x=self.fc8(x)
            x=self.leakyrelu(x)

            x=self.fc9(x)
            output=x.view(batchsize,self.output_length,9)
        elif self.feature_pool == "attention":
            outputs = outputs.permute(1, 0, 2)
            y, attentions = self.attention(outputs)
            # output = self.fc(y)
            x=self.fc1(y)
            x=self.leakyrelu(x)

            x1=self.fc2(x)
            x1=self.leakyrelu(x1)
            x1=self.fc3(x1)
            x=x+x1
            x=self.leakyrelu(x)

            x1=self.fc4(x)
            x1=self.leakyrelu(x1)
            x1=self.fc5(x)
            x=x+x1
            x=self.leakyrelu(x)

            # x1=self.fc6(x)
            # x1=self.leakyrelu(x1)
            # x1=self.fc7(x)
            # x=x+x1
            # x=self.leakyrelu(x)

            x=self.fc8(x)
            x=self.leakyrelu(x)

            x=self.fc9(x)
            output=x.view(batchsize,self.output_length,joint_number,3)
        else:
            output = self.fc(outputs[-1])

        return output

class Skleton_Morphin(nn.Module):

    def __init__(self,
                 rnn_size=1024,
                 input_size=30,
                 num_layers=1,
                 output_length=1,
                 feature_pool="concat",
                 use_spectral_norm=False,
                 attention_size=1024,
                 attention_layers=1,
                 attention_dropout=0.5):

        super(Skleton_Morphin, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.feature_pool = feature_pool
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout

        self.output_length=output_length
        self.output_shape=self.output_length*30

        self.gru = nn.GRU(self.input_size, self.rnn_size, num_layers=num_layers)

        linear_size = self.rnn_size if not feature_pool == "concat" else self.rnn_size * 2

        if feature_pool == "attention" :
            self.attention = SelfAttention(attention_size=self.attention_size,
                                       layers=self.attention_layers,
                                       dropout=self.attention_dropout)
        # if use_spectral_norm:
        #     self.fc = spectral_norm(nn.Linear(linear_size, output_size))
        # else:
        #     self.fc = nn.Linear(linear_size, output_size)

        

        self.leakyrelu=nn.LeakyReLU()
        self.fc1=nn.Linear(linear_size,1000)
        self.fc2=nn.Linear(1000,1000)
        self.fc3=nn.Linear(1000,1000)
        self.fc4=nn.Linear(1000,self.output_shape)
        # self.fc5=nn.Linear(1000,1000)
        # self.fc6=nn.Linear(1000,1000)
        # self.fc7=nn.Linear(1000,1000)
        # self.fc8=nn.Linear(1000,1000)
        # self.fc9=nn.Linear(1000,self.output_shape)

    def forward(self, sequence):
        """
        sequence: of shape [batch_size, seq_len, joint_number, 2]
        """
        batchsize, seqlen, joint_number, _ = sequence.shape
        sequence=sequence.contiguous()
        sequence=sequence.view(batchsize,seqlen,joint_number*2)
        sequence = torch.transpose(sequence, 0, 1)

        outputs, state = self.gru(sequence)

        if self.feature_pool == "concat":
            outputs = self.leakyrelu(outputs)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(1, 2, 0), 1).view(batchsize, -1)
            x = torch.cat([avg_pool, max_pool], dim=1)

            x=self.fc1(x)
            x=self.leakyrelu(x)

            x1=self.fc2(x)
            x1=self.leakyrelu(x1)
            x1=self.fc3(x1)
            x=x+x1
            x=self.leakyrelu(x)

            x=self.fc4(x)
            # x1=self.leakyrelu(x1)
            # x1=self.fc5(x)
            # x=x+x1
            # x=self.leakyrelu(x)


            # x=self.fc8(x)
            # x=self.leakyrelu(x)

            # x=self.fc9(x)
            output=x.view(batchsize,self.output_length,joint_number,2)
        elif self.feature_pool == "attention":
            outputs = outputs.permute(1, 0, 2)
            y, attentions = self.attention(outputs)
            output = self.fc(y)
        else:
            output = self.fc(outputs[-1])

        return output