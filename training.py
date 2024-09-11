import numpy as np
import numpy as np
import json
from common.arguments import parse_args
import torch
import matplotlib.pyplot as plt
# import ignite.contrib.metrics.regression.r2_score as R2_SCORE
from sklearn.metrics import r2_score,mean_squared_error
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import os
import sys
import errno
from time import time
from common.h36m_dataset import Human36mDataset
from common.camera import *
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import Dataset
# from progress.bar import Bar
from torch.utils.data import DataLoader
import pytorch3d.transforms
def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def plot_15j(poses):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_idx = 1
        frames = np.linspace(start=0, stop=poses.shape[0]-1, num=poses.shape[0]).astype(int)
        for i in range(poses.shape[0]):
                # ax = fig.add_subplot(1, 2, plot_idx)
                width=1
                x=poses[i,:,0]
                y=poses[i,:,1]
                ax.scatter(poses[i,:,0],poses[i,:,1])
                ax.plot(poses[i,[0,1],0], poses[i,[0,1],1],linewidth=width,color='red')
                ax.plot(poses[i,[1,2],0], poses[i,[1,2],1],linewidth=width,color='red')
                ax.plot(poses[i,[2,3],0], poses[i,[2,3],1],linewidth=width,color='red')
                ax.plot(poses[i,[0,4],0], poses[i,[0,4],1],linewidth=width,color='blue')
                ax.plot(poses[i,[4,5],0], poses[i,[4,5],1],linewidth=width,color='blue')
                ax.plot(poses[i,[5,6],0], poses[i,[5,6],1],linewidth=width,color='blue')
                ax.plot(poses[i,[0,7],0], poses[i,[0,7],1],linewidth=width,color='green')
                ax.plot(poses[i,[7,8],0], poses[i,[7,8],1],linewidth=width,color='green')
                ax.plot(poses[i,[7,9],0], poses[i,[7,9],1],linewidth=width,color='blue')
                ax.plot(poses[i,[9,10],0], poses[i,[9,10],1],linewidth=width,color='blue')
                ax.plot(poses[i,[10,11],0], poses[i,[10,11],1],linewidth=width,color='blue')
                ax.plot(poses[i,[7,12],0], poses[i,[7,12],1],linewidth=width,color='red')
                ax.plot(poses[i,[12,13],0], poses[i,[12,13],1],linewidth=width,color='red')
                ax.plot(poses[i,[13,14],0], poses[i,[13,14],1],linewidth=width,color='red')
                max_range = np.array([x.max() - x.min(), y.max() - y.min()]).max()
                Xb = 0.1 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.1 * (x.max() + x.min())
                Yb = 0.1 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.1 * (y.max() + y.min())
                for xb, yb in zip(Xb, Yb):
                        ax.plot([xb], [yb], 'w')

                # ax.axis('equal')
                ax.axis('auto')
                ax.axis('on')

                ax.set_title('frame = ' + str(i))

                plot_idx += 1
            # this uses QT5Agg backend
        # you can identify the backend using plt.get_backend()
        # delete the following two lines and resize manually if it throws an error
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        plt.grid(True)
        plt.show()

class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
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
        """
        input: bx16x2 / bx32
        output: bx16x3
        """

        # pre-processing
        # x = x.view(x.shape[0], x.shape[2], 16, 2)
        x = x.view(x.shape[0], -1, 15, 2)

        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
        
        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        x = self._forward_blocks(x)
        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)
        

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
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
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
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        
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

##########################numpy align#######################################
def angle(v1, v2):
        return np.arccos((dotproduct(v1, v2) / (length(v1) * length(v2)+0.000001)))

def dotproduct(v1, v2):
        return sum((a*b) for a, b in zip(v1, v2))

def length(v):
        return np.sqrt(dotproduct(v, v))

def align(X,ref_vec):
        Z=np.zeros_like(X)
        v1=X[:,4]-X[:,1]
        v1[:,2]=0
        v2=[0,1,0]
        # v2=ref_vec
        def dotproduct(v1, v2):
                return sum((a*b) for a, b in zip(v1, v2))

        def length(v):
                return np.sqrt(dotproduct(v, v))

        def angle(v1, v2):
                return np.arccos((dotproduct(v1, v2) / (length(v1) * length(v2)+0.000001)))
        ii=0
        while ii<X.shape[0]:
                theta=angle(v2,v1[ii])
                R=[[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
                
                X[ii]=np.matmul(R,X[ii].T).T
                v1[ii]=X[ii,4]-X[ii,1]
                v1[:,2]=0
                res=angle(v2,v1[ii])

                if res<0.01:
                        ii+=1   
        return X
##################################################################


##########################torch align#######################################
def angle_batch(v1, v2):
        return torch.acos((dotproduct_batch(v1, v2) / (length_batch(v1) * length_batch(v2)+0.000001)))

def dotproduct_batch(v1, v2):
        return torch.sum(v1*v2,dim=-1)

def length_batch(v):
        return torch.sqrt(dotproduct_batch(v, v))

def align_batch(X,ref_vec,plot):

        v1=X[:,4]-X[:,1]
        v1[:,2]=0
        v2=torch.tensor([0.0,1.0,0.0]).unsqueeze(0).repeat(X.shape[0],1).cuda()

        def angle_batch(v1, v2):
                return torch.acos((dotproduct_batch(v1, v2) / (length_batch(v1) * length_batch(v2)+0.000001)))

        def dotproduct_batch(v1, v2):
                return torch.sum(v1*v2,dim=-1)

        def length_batch(v):
                return torch.sqrt(dotproduct_batch(v, v))
                
        ii=0
        res=torch.ones(20)
        while torch.sum(res>0.1)>10 and ii<10:
                theta=angle_batch(v2,v1)+0.001
                R=torch.zeros((X.shape[0],3,3)).cuda()
                R[:,0,0]=torch.cos(theta)
                R[:,0,1]=-torch.sin(theta)
                R[:,1,0]=torch.sin(theta)
                R[:,1,1]=torch.cos(theta)
                R[:,2,2]=1.0
                
                # Y=torch.zeros_like(X).cuda()
                X=torch.matmul(R,X.transpose(2,1)).transpose(2,1)
                v1=X[:,4]-X[:,1]
                v1[:,2]=0
                res=angle_batch(v2,v1)
                if plot:
                        plt.plot(res.detach().cpu().numpy()/3.14*180)
                        plt.title(ii)
                        plt.show()
                ii+=1
        # if res>0.01:
        #         print(res)
                # ii+=1   
        return X
##################################################################
                      
def gait_batch(X,plot=False):


        ref_vec=torch.tensor([0,1,0]).cuda()
        X_aligned=align_batch(X,ref_vec,plot=plot)


        # plot17j(A)
        Angle=torch.zeros((X.shape[0],4)).cuda()

        v1=X_aligned[:,2]-X_aligned[:,1]
        v2=X_aligned[:,3]-X_aligned[:,2]
        v3=X_aligned[:,0]-X_aligned[:,7]
        v4=X_aligned[:,5]-X_aligned[:,4]
        v5=X_aligned[:,6]-X_aligned[:,5]
        v6=X_aligned[:,0]-X_aligned[:,7]
                                
        
        v1[:,1]=0
        v2[:,1]=0
        v3[:,1]=0
        v4[:,1]=0
        v5[:,1]=0
        v6[:,1]=0
        v3=torch.flip(v3,dims=[-1])
        v3[:,2]*=-1
        v6=torch.flip(v6,dims=[-1])
        v6[:,2]*=-1
        # Knee0_r,Knee0_l=5/180*3.14159,5/180*3.14159
        # Hip0_r,Hip0_l=80/180*3.14159,80/180*3.14159
        Angle[:,0]=(angle_batch(v1,v2))/3.14159*180-5
        Angle[:,1]=(-(angle_batch(v1,v3)))/3.14159*180+80
        Angle[:,2]=((angle_batch(v4,v5))/3.14459*180)-5
        Angle[:,3]=(-(angle_batch(v4,v6))/3.14159*180)+80
        plt.plot(Angle[:,0].detach().cpu().numpy(),label='Knee Angle R')
        plt.legend()
        plt.show()
        plt.plot(Angle[:,1].detach().cpu().numpy(),label='Hip Angle R')
        plt.legend()
        plt.show()
        plt.plot(Angle[:,2].detach().cpu().numpy(),label='Knee Angle L ')
        plt.legend()
        plt.show()
        plt.plot(Angle[:,3].detach().cpu().numpy(),label='Hip Angle L')
        plt.legend()
        plt.show()
        return Angle

def fetch_data(subj_train,dataset,keypoints,Angle):
        data_angle=[]
        data_2d=[]
        data_3d=[]
        if Angle is None:
                for sub in subj_train:
                        for action in dataset[sub].keys():
                                for cam in range(4):
                                        data_2d.append(keypoints[sub][action][cam])
                                        data_3d.append(dataset[sub][action]['positions_3d'][cam])
                data_angle=None    
                data_3d=np.concatenate(data_3d)
                data_2d=np.concatenate(data_2d)            
        else:
                for sub in subj_train:
                        for action in Angle[sub].keys():
                                for cam in range(4):
                                        data_2d.append(keypoints[sub][action][cam])
                                        data_3d.append(dataset[sub][action]['positions_3d'][cam])
                                        data_angle.append(np.concatenate((np.expand_dims(Angle[sub][action]['Hip_R'],axis=1),
                                        np.expand_dims(Angle[sub][action]['Knee_R'],axis=1),
                                        np.expand_dims(Angle[sub][action]['Hip_L'],axis=1),
                                        np.expand_dims(Angle[sub][action]['Knee_L'],axis=1)),axis=1))
                data_angle=np.concatenate(data_angle)    
                data_3d=np.concatenate(data_3d)
                data_2d=np.concatenate(data_2d)  

        return data_2d,data_3d,data_angle
#################################################################### 
class PoseDataSet(Dataset):
    def __init__(self, poses_3d, poses_2d, data_angle):
        assert poses_3d is not None

        self._poses_3d = poses_3d
        self._poses_2d = poses_2d
        self._data_angle = data_angle

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0]
        print('Generating {} poses...')

    def __getitem__(self, index):

        pad=40
        stride=np.random.randint(1,20)
        pad_idx=np.arange(2*pad+1)*stride-(pad*stride)+index
        if pad_idx[0]>0 and pad_idx[-1]<len(self._poses_3d):
            out_pose_3d = np.reshape(self._poses_3d[index],(1,15,3))
            out_pose_2d = np.reshape(self._poses_2d[pad_idx],(1,2*pad+1,15,2))
            if self._data_angle is not None:
              out_angle   = np.reshape(self._data_angle[index],(1,1,4))
            else:
              out_angle  = None
        else:
            out_pose_3d=np.zeros((1,15,3))
            out_pose_2d=np.zeros((1,2*pad+1,15,2))
            if self._data_angle is not None:
              out_angle   = np.zeros((1,1,4))
            else:
              out_angle  = None

        # out_pose_3d = self._poses_3d[index]
        # out_pose_2d = self._poses_2d[index]
        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        if self._data_angle is not None:
          out_angle = torch.from_numpy(out_angle).float()

          return  out_pose_2d, out_pose_3d, out_angle
        else:

          return  out_pose_2d, out_pose_3d


    def __len__(self):
        return len(self._poses_2d)

def chunck_generator(data_2d,data_3d,data_angle,pad,aug=False):
        rev_kpt=[0,4,5,6,1,2,3,7,8,9,10,14,15,16,11,12,13]
        padded_2d=[]
        padded_3d=[]
        padded_angle=[]
        if data_angle is None:
          for ii in range(pad,len(data_2d)-pad):
                  padded_2d.append(np.reshape(data_2d[ii-pad:ii+pad+1],(1,81,17,2)))
                  padded_3d.append(np.reshape(data_3d[ii],(1,17,3)))
                  if aug:
                          padded_2d.append(np.reshape(data_2d[ii-pad:ii+pad+1,rev_kpt],(1,81,17,2)))
                          padded_angle.append(np.reshape(data_angle[ii],(1,1,2)))
          data_angle=None    
          data_3d=np.concatenate(padded_3d)
          data_2d=np.concatenate(padded_2d) 
        else:
          for ii in range(pad,len(data_2d)-pad):
                  padded_2d.append(np.reshape(data_2d[ii-pad:ii+pad+1],(1,81,17,2)))
                  padded_3d.append(np.reshape(data_3d[ii],(1,17,3)))
                  padded_angle.append(np.reshape(data_angle[ii],(1,1,4)))
                  if aug:
                          padded_2d.append(np.reshape(data_2d[ii-pad:ii+pad+1,rev_kpt],(1,81,17,2)))
                          padded_angle.append(np.reshape(data_angle[ii],(1,1,2)))
          data_angle=np.concatenate(padded_angle)    
          data_3d=np.concatenate(padded_3d)
          data_2d=np.concatenate(padded_2d)  
                
        return data_2d,data_3d,data_angle
#####################################################################
actions=['Walking','Walking 1']
subjs=['S1','S5','S7','S8','S9','S11']
keypoints = np.load('data/data_2d_h36m_gt.npz', allow_pickle=True) #C:/Users/mghol/OneDrive/Desktop/PhD_UBC/Thesis/Projects/Project1_VideoPose3D/Code/
keypoints = keypoints['positions_2d'].item()
dataset = Human36mDataset('data/data_3d_h36m.npz') #C:/Users/mghol/OneDrive/Desktop/PhD_UBC/Thesis/Projects/Project1_VideoPose3D/Code/

joints_15mpii=[0,4,5,6,1,2,3,8,10,14,15,16,11,12,13]        
for subject in dataset.subjects():
        for action in dataset[subject].keys():
                anim = dataset[subject][action]
                if 'positions' in anim:
                        positions_3d = []
                        for cam in anim['cameras']:
                                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                                pos_3d=pos_3d[:,joints_15mpii]
                                pos_3d[:,:,:] -= pos_3d[:, :1,:] # Remove global offset, but keep trajectory in first position
                                positions_3d.append(pos_3d)
                        anim['positions_3d'] = positions_3d


for subject in keypoints.keys():
        for action in keypoints[subject].keys():
                
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                        #     kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=1000, h=1000)
                        kps=kps[:,joints_15mpii]
                        kps=kps-kps[:,:1,:]
                        kps=kps/np.mean(np.linalg.norm(kps[:,:,:],axis=-1,keepdims=True),axis=-2,keepdims=True)
                        keypoints[subject][action][cam_idx] = kps

Angle={}
for subj in dataset.subjects():
        Angle[subj]={}
        for action in dataset[subj].keys():
                if action[0:11] == 'SittingDown':
                        continue
                
        

                # for ii in range(len(dataset[subj][action]['positions'])):
                # plot17j(dataset['S1']['Walking 1']['positions'][240:250])
                ref_vec=[0,1,0]
                A=align(dataset[subj][action]['positions'][:],ref_vec)
               

                # plot17j(A)
                Angle[subj][action]={}
                Angle[subj][action]['Knee_R']=[]
                Angle[subj][action]['Hip_R']=[]
                Angle[subj][action]['Knee_L']=[]
                Angle[subj][action]['Hip_L']=[]

                for jj in range(len(A)):
                        v1=A[jj,2]-A[jj,1]
                        v2=A[jj,3]-A[jj,2]
                        v3=A[jj,0]-A[jj,7]
                        v4=A[jj,5]-A[jj,4]
                        v5=A[jj,6]-A[jj,5]
                        v6=A[jj,0]-A[jj,7]
                                                
                        
                        v1[1]=0
                        v2[1]=0
                        v3[1]=0
                        v4[1]=0
                        v5[1]=0
                        v6[1]=0
                        v3=np.flip(v3)
                        v3[2]*=-1
                        v6=np.flip(v6)
                        v6[2]*=-1

                        if jj==0:
                                Knee0_r=angle(v1,v2)
                                Hip0_r=angle(v1,v3)
                                Knee0_l=angle(v4,v5)
                                Hip0_l=angle(v4,v6)
                                Knee0_r,Knee0_l=5/180*np.pi,5/180*np.pi
                                Hip0_r,Hip0_l=80/180*np.pi,80/180*np.pi
                                # print(np.asarray([Knee0_r,Hip0_r,Knee0_l,Hip0_l])/np.pi*180)

                        Angle[subj][action]['Knee_R'].append((angle(v1,v2)-Knee0_r)/np.pi*180)
                        Angle[subj][action]['Hip_R'].append(-(angle(v1,v3)-Hip0_r)/np.pi*180)
                        Angle[subj][action]['Knee_L'].append((angle(v4,v5)-Knee0_l)/np.pi*180)
                        Angle[subj][action]['Hip_L'].append(-(angle(v4,v6)-Hip0_l)/np.pi*180)
                        # Hip_Angle.append(angle(v1,v3)/np.pi*180)
                        # print(angle(v1,v2))
                # Knee_Angle=np.concatenate(Knee_Angle)
                
                # plt.plot(Angle[subj][action]['Knee_L'],label='Knee Angle')
                # plt.legend()
                # plt.show()
                # plt.plot(Angle[subj][action]['Hip_L'],label='Hip Angle')
                # plt.legend()
                # plt.show()
  
# Knee_Angle=np.concatenate(Knee_Angle)

# plt.plot(Angle[subj][action]['Knee_L'],label='Knee Angle')
# # # plt.show()
# plt.plot(Angle[subj][action]['Hip_L'],label='Hip Angle')
# plt.legend()
# plt.show()

subj_train=['S1','S5','S7','S8']
subj_test=['S9','S11']
data_2d_train,data_3d_train,data_angle_train=fetch_data(subj_train,dataset,keypoints,Angle)
data_2d_test,data_3d_test,data_angle_test=fetch_data(subj_test,dataset,keypoints,Angle)

data_2d_train_only_pose,data_3d_train_only_pose,_=fetch_data(subj_train,dataset,keypoints,None)
data_2d_test_only_pose,data_3d_test_only_pose,_=fetch_data(subj_test,dataset,keypoints,None)

print(data_2d_train.shape)
print(data_angle_train.shape)
print(data_2d_test.shape)
print(data_angle_test.shape)

batch_size=1024
train_loader = DataLoader(PoseDataSet(data_3d_train, data_2d_train, data_angle_train),
                          batch_size=batch_size,
                          shuffle=False, pin_memory=False)
test_loader = DataLoader(PoseDataSet(data_3d_test, data_2d_test, data_angle_test),
                          batch_size=batch_size,
                          shuffle=False,  pin_memory=False)

train_loader_only_pose = DataLoader(PoseDataSet(data_3d_train_only_pose, data_2d_train_only_pose, None),
                          batch_size=batch_size,
                          shuffle=True,  pin_memory=False)
test_loader_only_pose = DataLoader(PoseDataSet(data_3d_test_only_pose, data_2d_test_only_pose, None),
                          batch_size=batch_size,
                          shuffle=False, pin_memory=False)

# data_2d_train,data_3d_train,data_angle_train=chunck_generator(data_2d_train,data_3d_train,data_angle_train,pad=40)
# data_2d_test,data_3d_test,data_angle_test=chunck_generator(data_2d_test,data_3d_test,data_angle_test,pad=40)
# data_2d_train_only_pose,data_3d_train_only_pose,_=chunck_generator(data_2d_train_only_pose,data_3d_train_only_pose,data_angle=None,pad=40)
# data_2d_test_only_pose,data_3d_test_only_pose,_=chunck_generator(data_2d_test_only_pose,data_3d_test_only_pose,data_angle=None,pad=40)
# print(data_2d_train.shape)
# print(data_angle_train.shape)
# print(data_2d_test.shape)
# print(data_angle_test.shape)

# shuffled_rows=np.arange(len(data_2d_train))
# np.random.shuffle(shuffled_rows)
# data_2d_train=data_2d_train[shuffled_rows]
# data_3d_train=data_3d_train[shuffled_rows]
# data_angle_train=data_angle_train[shuffled_rows]

# shuffled_rows=np.arange(len(data_2d_train_only_pose))
# np.random.shuffle(shuffled_rows)
# data_2d_train_only_pose=data_2d_train_only_pose[shuffled_rows]
# data_3d_train_only_pose=data_3d_train_only_pose[shuffled_rows]

architecture='3,3,3,3'
filter_widths=[int(x) for x in architecture.split(',')]
model_angle_train=TemporalModelOptimized1f(num_joints_in=15,in_features=2, num_joints_out=19,filter_widths=filter_widths)
model_angle=TemporalModelOptimized1f(num_joints_in=15,in_features=2, num_joints_out=19,filter_widths=filter_widths)
# model_angle_train = Generator_GRU()
# model_angle = Generator_GRU()
if torch.cuda.is_available():
        model_angle = model_angle.cuda()
        model_angle_train = model_angle_train.cuda()
optimizer = optim.Adam(list(model_angle_train.parameters()),lr=0.001)
mse_loss = nn.MSELoss()
lr=0.001
epochs=100
epoch=0

lr_decay=0.95
num_batch=int(len(data_2d_train)/batch_size)
num_batch_test=int(len(data_2d_test)/batch_size)
num_batch_only_pose=int(len(data_2d_train_only_pose)/batch_size)
num_batch_test_only_pose=int(len(data_2d_test_only_pose)/batch_size)



while epoch<epochs:
        # start_time = time()
        epoch+=1
        epoch_loss_3d_train = 0
        N=0
        model_angle_train.train()
        model_angle.train()
        
        # for idx, (batch_2d,batch_3d) in enumerate(train_loader_only_pose):

        #         if torch.cuda.is_available():
        #                 batch_2d = batch_2d.cuda()
        #                 batch_3d = batch_3d.cuda()
        #         batch_2d=batch_2d[:,0]
        #         batch_3d=batch_3d[:,0]
        #         optimizer.zero_grad()

        #         pred=model_angle_train(batch_2d[:,:,0:17])
        #         loss_pos=mse_loss(pred[:,0,:17],batch_3d[:,:17])
        #         loss=loss_pos 
        #         # loss.backward()
        #         # optimizer.step()   
        #         if idx%10==0:
        #                 print("[Epoch %d/%d] [Batch %d/%d] [loss_pos: %f]"
        #                 % (epoch, epochs, idx , num_batch, loss_pos.item() ))             
        for idx, (batch_2d,batch_3d,batch_angle) in enumerate(train_loader):

                if torch.cuda.is_available():
                        batch_2d = batch_2d.cuda()
                        batch_3d = batch_3d.cuda()
                        batch_angle = batch_angle.cuda()
                batch_2d=batch_2d[:,0]
                batch_3d=batch_3d[:,0]
                batch_angle=batch_angle[:,0]
                optimizer.zero_grad()
                pred=model_angle_train(batch_2d[:,:,0:15])
                # plot_15j(batch_2d[500:501,40,:15].cpu().numpy())
                loss_angle=mse_loss(pred[:,0,15:19,0],batch_angle[:,0])
                loss_pos=mse_loss(pred[:,0,:15],batch_3d[:,:15])*400
                
                # pred_angles=gait_batch(pred[:,0,:15],plot=False)
                # ref_angle=gait_batch(batch_3d[:,:15],plot=True)
                # loss_angle2=mse_loss(pred_angles,batch_angle[:,0])
                loss= loss_angle# + loss_pos #+ loss_angle2/1000
                loss.backward()
                optimizer.step()
                if idx%10==0:
                        print("[Epoch %d/%d] [Batch %d/%d] [loss_angle: %f]  [loss_pos: %f]"
                        % (epoch, epochs, idx , num_batch, loss_angle.item(), loss_pos.item() ))

        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

        with torch.no_grad():
                model_angle.load_state_dict(model_angle_train.state_dict())
                model_angle.eval()
                val_loss=0
                val_loss_pos=0
                val_r2=0
                N=0
                N_pose=0
                pred_pose_tot=[]
                # for idx, (batch_2d,batch_3d)in enumerate(test_loader_only_pose):

                #         if torch.cuda.is_available():
                #                 batch_2d = batch_2d.cuda()
                #                 batch_3d = batch_3d.cuda()
                #         batch_2d=batch_2d[:,0]
                #         batch_3d=batch_3d[:,0]
                #         pred=model_angle(batch_2d[:,:,0:17])  


                #         val_loss_pos+=mpjpe(pred[:,0,:17],batch_3d)*batch_3d.shape[0]

                #         N_pose+=batch_3d.shape[0]

                for idx, (batch_2d,batch_3d,batch_angle)in enumerate(test_loader):

                        if torch.cuda.is_available():
                                batch_2d = batch_2d.cuda()
                                batch_3d = batch_3d.cuda()
                                batch_angle = batch_angle.cuda()
                        batch_2d=batch_2d[:,0]
                        batch_3d=batch_3d[:,0]
                        batch_angle=batch_angle[:,0]
                        pred=model_angle(batch_2d[:,:,0:15])  

                        # plt.plot(np.asarray(pred[:,0,17,0].cpu()),label='Ped')
                        # # # plt.show()
                        # plt.plot(np.asarray(batch_angle[:,0,0].cpu()),label='Target')
                        # plt.legend()
                        # plt.show()

                        # plt.plot(np.asarray(pred[:,0,18,0].cpu()),label='Ped')
                        # # # plt.show()
                        # plt.plot(np.asarray(batch_angle[:,0,1].cpu()),label='Target')
                        # plt.legend()
                        # plt.show()

                        # plt.plot(np.asarray(pred[:,0,19,0].cpu()),label='Ped')
                        # # # plt.show()
                        # plt.plot(np.asarray(batch_angle[:,0,2].cpu()),label='Target')
                        # plt.legend()
                        # plt.show()


                        # plt.plot(np.asarray(pred[:,0,20,0].cpu()),label='Ped')
                        # # # plt.show()
                        # plt.plot(np.asarray(batch_angle[:,0,3].cpu()),label='Target')
                        # plt.legend()
                        # plt.show()
                        pred_pose_tot.append(np.asarray(pred[:,0,:15].cpu()))
                        pred_angle=torch.reshape(pred[:,0,15:19,0],(pred.shape[0]*4,))
                        batch_angle=torch.reshape(batch_angle[:,0],(batch_angle.shape[0]*4,))

                        # pred_angle=np.array(pred_angle.cpu())/np.pi*180
                        # batch_angle=np.array(batch_angle.cpu())/np.pi*180  
                        val_loss_pos+=mpjpe(pred[:,0,:15],batch_3d)*batch_3d.shape[0]
                        
                        loss=mse_loss(pred_angle,batch_angle)
                        val_loss+=(loss**0.5)*pred_angle.shape[0]
                        S_res=torch.sum((batch_angle-pred_angle)**2)
                        S_tot=torch.sum((batch_angle-torch.mean(batch_angle,dim=0,keepdim=True))**2)
                        val_r2+=(1-S_res/S_tot)*pred_angle.shape[0]
                        N+=pred_angle.shape[0]
                        N_pose+=batch_3d.shape[0]
                print('Validation Error',val_loss/N,'Validation R2',val_r2/N, 'Validation 3D Pose', val_loss_pos/N_pose)
        if epoch % 10 == 0:
                modelname='angle_estimation'+'_epoch_{}.bin'.format(epoch)
                chk_path = os.path.join('checkpoint', modelname)
                print('Saving checkpoint to', chk_path)
                torch.save({
                        'epoch': epoch,
                        'lr': lr,
                        'optimizer': optimizer.state_dict() ,
                        'model_pos': model_angle.state_dict(),
                }, chk_path)
        # pred_pose_tot=np.concatenate(pred_pose_tot)
        # print(pred_pose_tot.shape)
        # if epoch>0:
        #         ref_vec=[0,1,0]
        #         # A=align(pred_pose_tot,ref_vec)
        #         A=pred_pose_tot

        #         Knee_r=[]
        #         hip_r=[]
        #         Knee_l=[]
        #         hip_l=[]
        #         for jj in range(len(A)):
        #                 v1=A[jj,2]-A[jj,1]
        #                 v2=A[jj,3]-A[jj,2]
        #                 v3=A[jj,0]-A[jj,7]
        #                 v4=A[jj,5]-A[jj,4]
        #                 v5=A[jj,6]-A[jj,5]
        #                 v6=A[jj,0]-A[jj,7]
                                                
                        
        #                 v1[1]=0
        #                 v2[1]=0
        #                 v3[1]=0
        #                 v4[1]=0
        #                 v5[1]=0
        #                 v6[1]=0
        #                 v3=np.flip(v3)
        #                 v3[2]*=-1
        #                 v6=np.flip(v6)
        #                 v6[2]*=-1

        #                 if jj==0:
        #                         Knee0_r=angle(v1,v2)
        #                         Hip0_r=angle(v1,v3)
        #                         Knee0_l=angle(v4,v5)
        #                         Hip0_l=angle(v4,v6)
        #                 Knee_r.append((angle(v1,v2)-Knee0_r)/np.pi*180)
        #                 hip_r.append(-(angle(v1,v3)-Hip0_r)/np.pi*180)
        #                 Knee_l.append((angle(v4,v5)-Knee0_l)/np.pi*180)
        #                 hip_l.append(-(angle(v4,v6)-Hip0_l)/np.pi*180)
        #         # Knee_r=np.concatenate(Knee_r)
        #         print(np.array(Knee_r).shape)
        #         print(data_2d_test.shape)
