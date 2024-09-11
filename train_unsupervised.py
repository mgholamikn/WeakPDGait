# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import numpy as np
import json
from common.arguments import parse_args
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import os
import sys
import errno
from time import time
from pytorch3d.transforms import so3_exponential_map,axis_angle_to_matrix
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl

def numpy_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1),axis=-1)

def numpy_nmpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    norm_predicted = np.mean(np.sum(predicted**2, axis=-1,keepdims=True), axis=-2,keepdims=True)
    norm_target = np.mean(np.sum(target*predicted, axis=-1,keepdims=True), axis=-2,keepdims=True)
    scale = norm_target / (norm_predicted+0.0001)
    return numpy_mpjpe(scale * predicted, target)

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    # predicted_aligned = a*predicted 

    # Return MPJPE
    e=np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1),axis=-1)

    return e

#####################################
# data loader 
#####################################
class PoseDataSet(Dataset):
    def __init__(self, poses_3d, poses_2d, pad=13):
        assert poses_3d is not None

        self._poses_3d = poses_3d
        self._poses_2d = poses_2d
        # self._cams = cams
        self._pad=pad

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] 
        # assert self._poses_3d.shape[0] == self._cams.shape[0]
        

    def __getitem__(self, index):


        if self._pad>0:
            stride=1
            if index<self._pad:
                index=14
            elif index+14>len(self._poses_3d):
                index-=14
            pad_idx=np.arange(2*self._pad+1)*stride-(self._pad*stride)+index

            # if pad_idx[0]>=0 and pad_idx[-1]<len(self._poses_3d):
   
            out_pose_3d = np.reshape(self._poses_3d[pad_idx],(2*self._pad+1,self._poses_3d.shape[1],self._poses_3d.shape[2]))
            out_pose_2d = np.reshape(self._poses_2d[pad_idx],(2*self._pad+1,self._poses_2d.shape[1],self._poses_2d.shape[2]))
            # else:
            #     out_pose_3d=np.zeros((2*self._pad+1,self._poses_3d.shape[1],self._poses_3d.shape[2]))
            #     out_pose_2d=np.zeros((2*self._pad+1,self._poses_2d.shape[1],self._poses_2d.shape[2]))
                
        else:
            out_pose_3d = np.reshape(self._poses_3d[index],(2*self._pad+1,self._poses_3d.shape[-2],self._poses_3d.shape[-1]))
            out_pose_2d = np.reshape(self._poses_2d[index],(2*self._pad+1,self._poses_2d.shape[-2],self._poses_2d.shape[-1]))

        # out_cam = self._cams[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return  out_pose_3d, out_pose_2d 

    def __len__(self):
        return len(self._poses_2d)
#####################################
# Plot
#####################################
def plot_18j_3d(poses):
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        plot_idx = 1
        frames = np.linspace(start=0, stop=poses.shape[0]-1, num=2).astype(int)
        for i in range(poses.shape[0]):
                ax = fig.add_subplot(1, poses.shape[0], plot_idx, projection='3d')
                x=poses[i,:,0]
                y=poses[i,:,1]
                z=poses[i,:,2]
                ax.scatter(poses[i,:,0],poses[i,:,1],poses[i,:,2])
                width=1
                ax.plot(poses[i,[0,1],0], poses[i,[0,1],1], poses[i,[0,1],2],linewidth=width,color='red')
                ax.plot(poses[i,[1,2],0], poses[i,[1,2],1], poses[i,[1,2],2],linewidth=width,color='red')
                ax.plot(poses[i,[2,3],0], poses[i,[2,3],1], poses[i,[2,3],2],linewidth=width,color='red')
                ax.plot(poses[i,[0,4],0], poses[i,[0,4],1], poses[i,[0,4],2],linewidth=width,color='blue')
                ax.plot(poses[i,[4,5],0], poses[i,[4,5],1], poses[i,[4,5],2],linewidth=width,color='blue')
                ax.plot(poses[i,[5,6],0], poses[i,[5,6],1], poses[i,[5,6],2],linewidth=width,color='blue')
                ax.plot(poses[i,[0,7],0], poses[i,[0,7],1], poses[i,[0,7],2],linewidth=width,color='green')
                ax.plot(poses[i,[7,8],0], poses[i,[7,8],1], poses[i,[7,8],2],linewidth=width,color='green')
                ax.plot(poses[i,[7,9],0], poses[i,[7,9],1], poses[i,[7,9],2],linewidth=width,color='blue')
                ax.plot(poses[i,[9,10],0], poses[i,[9,10],1], poses[i,[9,10],2],linewidth=width,color='blue')
                ax.plot(poses[i,[10,11],0], poses[i,[10,11],1], poses[i,[10,11],2],linewidth=width,color='blue')
                ax.plot(poses[i,[7,12],0], poses[i,[7,12],1], poses[i,[7,12],2],linewidth=width,color='red')
                ax.plot(poses[i,[12,13],0], poses[i,[12,13],1], poses[i,[12,13],2],linewidth=width,color='red')
                ax.plot(poses[i,[13,14],0], poses[i,[13,14],1], poses[i,[13,14],2],linewidth=width,color='red')
                max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
                # Create cubic bounding box to simulate equal aspect ratio
                max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
                Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
                Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
                Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

                for xb, yb, zb in zip(Xb, Yb, Zb):
                        ax.plot([xb], [yb], [zb], 'w')

                # ax.axis('equal')
                ax.axis('auto')
                ax.axis('on')

                ax.set_title('frame = ' + str(i))

                plot_idx += 1
            # this uses QT5Agg backend
        # you can identify the backend using plt.get_backend()
        # delete the following two lines and resize manually if it throws an error
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.grid(True)
        plt.show()

def plot_18j(poses):
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        plot_idx = 1
        frames = np.linspace(start=0, stop=poses.shape[0]-1, num=2).astype(int)
        for i in range(poses.shape[0]):
                ax = fig.add_subplot(1, poses.shape[0], plot_idx)
                x=poses[i,:,0]
                y=poses[i,:,1]
                ax.scatter(poses[i,:,0],poses[i,:,1])
                ax.plot(poses[i,[0,1],0], poses[i,[0,1],1])
                ax.plot(poses[i,[1,2],0], poses[i,[1,2],1])
                ax.plot(poses[i,[2,3],0], poses[i,[2,3],1])
                ax.plot(poses[i,[0,4],0], poses[i,[0,4],1])
                ax.plot(poses[i,[4,5],0], poses[i,[4,5],1])
                ax.plot(poses[i,[5,6],0], poses[i,[5,6],1])
                ax.plot(poses[i,[0,7],0], poses[i,[0,7],1])
                ax.plot(poses[i,[7,8],0], poses[i,[7,8],1])
                ax.plot(poses[i,[7,9],0], poses[i,[7,9],1])
                ax.plot(poses[i,[9,10],0], poses[i,[9,10],1])
                ax.plot(poses[i,[10,11],0], poses[i,[10,11],1])
                ax.plot(poses[i,[7,12],0], poses[i,[7,12],1])
                ax.plot(poses[i,[12,13],0], poses[i,[12,13],1])
                ax.plot(poses[i,[13,14],0], poses[i,[13,14],1])
                max_range = np.array([x.max() - x.min(), y.max() - y.min()]).max()
                Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
                Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
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
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.grid(True)
        plt.show()
#####################################
# data loader 
#####################################
args = parse_args()
print(args)

data=np.load('data/ski.npz',allow_pickle=True)
data=data['arr_0'].item()
##########################
##########################
def fetch_train(sequence):
    kpts_2d=[]
    kpts_3d=[]
    cam=[]
    pairs=[[0,1],[1,2],[2,3],[3,4],[4,5],[0,5],[0,2],[3,5],[2,4]]
    for seq in sequence:
        for cam in range(len(pairs)):
            
            idx1=pairs[cam][0]
            idx2=pairs[cam][1]
            kpts_2d_1=data[seq][idx1]['2D']
            kpts_2d_2=data[seq][idx2]['2D']
            kpts_3d_1=data[seq][idx1]['3D']
            kpts_3d_2=data[seq][idx2]['3D']
            row=min(len(kpts_2d_1),len(kpts_2d_2))
            cam_1=data[seq][idx1]['R']
            cam_2=data[seq][idx2]['R']
            kpts_2d_1-=kpts_2d_1[:,:1]
            kpts_2d_2-=kpts_2d_2[:,:1]
            kpts_2d_1/=np.linalg.norm(kpts_2d_1,axis=(1,2),keepdims=True)
            kpts_2d_2/=np.linalg.norm(kpts_2d_2,axis=(1,2),keepdims=True)
            # plot_18j(np.concatenate((kpts_2d_1[:1],kpts_2d_2[:1]),axis=0))
            # plot_18j_3d(np.concatenate((kpts_3d_1[:1],kpts_3d_2[:1]),axis=0))
            kpts_2d.append(np.concatenate((kpts_2d_1[:row],kpts_2d_2[:row]),axis=-1))
            kpts_3d.append(np.concatenate((kpts_3d_1[:row],kpts_3d_2[:row]),axis=-1))
            # cam.append(np.concatenate((cam_1,cam_2),axis=-1))
    kpts_2d=np.concatenate(kpts_2d)
    kpts_3d=np.concatenate(kpts_3d)
    # cam=np.concatenate(cam)
    return kpts_3d,kpts_2d
###########################
###########################
def fetch_test(sequence):
    kpts_2d_=[]
    kpts_3d_=[]
    cam=[]
    for seq in sequence:
        for cam in range(6):
            kpts_2d=data[seq][cam]['2D']
            kpts_3d=data[seq][cam]['3D']
            kpts_2d-=kpts_2d[:,:1]
            kpts_2d/=np.linalg.norm(kpts_2d,axis=(1,2),keepdims=True)
            # plot_18j(np.concatenate((kpts_2d[:1],kpts_2d[:1]),axis=0))
            # plot_18j_3d(np.concatenate((kpts_3d_1[:1],kpts_3d_2[:1]),axis=0))
            kpts_2d_.append(kpts_2d)
            kpts_3d_.append(kpts_3d)
    kpts_2d_=np.concatenate(kpts_2d_)
    kpts_3d_=np.concatenate(kpts_3d_)

    return kpts_3d_,kpts_2d_
###########################
###########################
sequence_test=['405','412']
sequence_train=['103','110','115','124','202','207','214','221','302','309']
poses_valid, poses_valid_2d = fetch_test(sequence_test) 
poses_3d, poses_2d = fetch_train(sequence_train)
train_loader = DataLoader(PoseDataSet(poses_3d, poses_2d,pad=13),
                                    batch_size=args.batch_size,
                                    shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(PoseDataSet(poses_valid, poses_valid_2d,pad=13),
                                    batch_size=512,
                                    shuffle=False, num_workers=2, pin_memory=True)            
print(poses_3d.shape,poses_2d.shape)
filter_widths = [int(x) for x in args.architecture.split(',')]

# When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
pad=13
chunk_length=args.stride
model_pos_train = TemporalModelOptimized1f(num_joints_in=15, in_features=2, num_joints_out=15, chunk_length=chunk_length,filter_widths=filter_widths)
    
model_cam_train = TemporalModelOptimized1f(num_joints_in=15, in_features=4, num_joints_out=1, chunk_length=chunk_length,filter_widths=filter_widths)
# model_pos_train = Generator_GRU(output_length=chunk_length, num_layers=2)
    
# model_pos = Generator_GRU(output_length=chunk_length, num_layers=2)

if torch.cuda.is_available():
    model_cam_train = model_cam_train.cuda()
    model_pos_train = model_pos_train.cuda()


if not args.evaluate:
    lr = args.learning_rate

        
    optimizer_G = optim.Adam(list(model_pos_train.parameters()) + list(model_cam_train.parameters()),
                        lr=lr, amsgrad=True) #, amsgrad=True
    if torch.cuda.is_available():
        model_cam_train = model_cam_train.cuda()
        model_pos_train = model_pos_train.cuda()
         
    lr_decay = args.lr_decay

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001
    


    # Pos model only
    rec_w=0.1
    while epoch < args.epochs:
        start_time = time()
        N = 0
        N_semi = 0
        batches_done=0
        
        model_pos_train.train()
        model_cam_train.train()
        for i,(inputs_3d, inputs_2d) in enumerate(train_loader):
                            
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_3d = inputs_3d.cuda()
            
            inputs_2d_1=inputs_2d[:,:,:,:2].clone()
            inputs_2d_2=inputs_2d[:,:,:,2:].clone()
        
            inputs_3d[:, :, 0] = 0
            inputs_3d_1=inputs_3d[:,:,:,:3].clone()
            inputs_3d_2=inputs_3d[:,:,:,3:].clone()

            optimizer_G.zero_grad()
            # print(inputs_2d_1.shape)
            predicted_1 = model_pos_train(inputs_2d_1)
            predicted_2 = model_pos_train(inputs_2d_2)
            predicted_cam_1=model_cam_train(torch.cat((inputs_2d_1,inputs_2d_2),dim=-1))
            predicted_cam_2=model_cam_train(torch.cat((inputs_2d_2,inputs_2d_1),dim=-1))

            loss_G=0

            if pad > 0:
                target_1 = inputs_2d_1[:, pad:-pad, :, :2].contiguous()
                target_2 = inputs_2d_2[:, pad:-pad, :, :2].contiguous()
            else:
                target_1 = inputs_2d_1[:, :, :, :2].contiguous()    
                target_2 = inputs_2d_2[:, :, :, :2].contiguous()   
            
            projection_func = project_to_2d_linear if args.linear_projection else project_to_2d     
            rec1=0
            rec2=0
            rec3=0
            rec4=0                    
            if args.multi_projection: 

                R1_2=axis_angle_to_matrix(predicted_cam_1.squeeze())[..., :3, :3]
             
                # R2_1=R1_2.transpose(-1,-2)
                R2_1=axis_angle_to_matrix(predicted_cam_2.squeeze())[..., :3, :3]
                loss_cam=torch.abs(torch.mean(R2_1-R1_2.transpose(-1,-2)))

                # loss_cam=torch.abs(torch.mean(torch.matmul(R1_2,R2_1)-torch.eye(3).unsqueeze(0).repeat(R1_2.shape[0],1,1).cuda()))
                # cam[:,0:9] = R1_2.reshape(-1,9)
                # cam_semi[:,9:18]= R2_1.reshape(-1,9)
                

                reconstruction_1=predicted_1[:,:,:,:2]-predicted_1[:,:,:1,:2]
                target_n1=target_1-target_1[:,:,:1,:]
                rec1 = n_mpjpe(reconstruction_1, target_n1) # On 2D poses
                
                predicted_1_=predicted_1-predicted_1[:,:,:1]
                if epoch>1:
                    predicted_1_rev=predicted_1_.squeeze().detach().cpu().numpy()
                    predicted_1_rev[:,:,2]*=-1
                    e1=p_mpjpe(predicted_1_.squeeze().detach().cpu().numpy(),inputs_3d_1[:,pad].detach().cpu().numpy())
                    e2=p_mpjpe(predicted_1_rev,inputs_3d_1[:,pad].detach().cpu().numpy())
                    loss_3d_1=np.mean(np.minimum(e1,e2))
                else:
                    loss_3d_1=0

                 
                reconstruction_2_ = world_to_camera_batch2(predicted_1,R1_2.reshape(-1,9)) #predicted_traj_semi1.view(-1,9)
                reconstruction_2=reconstruction_2_[:,:,:,:2]-reconstruction_2_[:,:,:1,:2]
                target_n2=target_2-target_2[:,:,:1,:]
                rec2 = n_mpjpe(reconstruction_2, target_n2) # On 2D poses

                reconstruction_3=predicted_2[:,:,:,:2]-predicted_2[:,:,:1,:2]
                target_n3=target_2-target_2[:,:,:1,:]
                rec3 = n_mpjpe(reconstruction_3, target_n3)

                reconstruction_4_ = world_to_camera_batch2(predicted_2,R2_1.reshape(-1,9)) #predicted_traj_semi2.view(-1,9)
                reconstruction_4=reconstruction_4_[:,:,:,:2]-reconstruction_4_[:,:,:1,:2]
                target_n4=target_1-target_1[:,:,:1,:]
                rec4 = n_mpjpe(reconstruction_4, target_n4)
                loss_reconstruction=(4*rec1+4*rec3+rec2+rec4)/2
            
            else:                                
    
                reconstruction_semi=predicted_1[:,:,:,:2]
                loss_3d_1=n_mpjpe(predicted_1,inputs_3d_1)

                target_semi_n=target_1-target_1[:,:,:1,:]
                reconstruction_semi=reconstruction_semi-reconstruction_semi[:,:,:1,:]
                loss_reconstruction = n_mpjpe(reconstruction_semi, target_semi_n) # On 2D poses
                

            loss_G += loss_reconstruction/2 +loss_cam/10

            print(
                "[Epoch %d/%d] [Batch %d/%d] [GLoss: %f] [3DLoss_1: %f] [%f][%f][%f][%f]"
                    % (epoch, args.epochs, batches_done % len(train_loader), len(train_loader), loss_G.item(), loss_3d_1,rec1,rec2, rec3, rec4)# rec1,rec2, rec3, rec4)
                )

            loss_G.backward()
            optimizer_G.step()
            batches_done += 1
        epoch+=1


        #         # End-of-epoch evaluation
        with torch.no_grad():
            # for ii in range(0,100,10):
            ii=20
            rev=predicted_1[ii:ii+1,0].clone().cpu().numpy()
            rev[:,:,2]*=-1
            # plot_18j_3d(np.concatenate((predicted_1[ii:ii+1,0].clone().cpu().numpy(),rev,inputs_3d_1[ii:ii+1,13].clone().cpu().numpy()),axis=0))
            #     plot_18j(inputs_2d_1[ii:ii+1,13].clone().cpu().numpy())

            
            for i,( input_3d, input_2d) in enumerate(test_loader):
                start_time = time()
                N = 0
                loss_3d_p = 0
                loss_3d_n = 0
                
                model_pos_train.eval()
                for i,(inputs_3d, inputs_2d) in enumerate(test_loader):
                                    
                    if torch.cuda.is_available():
                        inputs_2d = inputs_2d.cuda()
                        inputs_3d = inputs_3d.cuda()
                    target_3d=inputs_3d[:,pad]
                    predicted = model_pos_train(inputs_2d).squeeze()
                    norm_predicted = torch.linalg.norm(predicted[:,7]-predicted[:,0],dim=-1)
                    norm_target = torch.linalg.norm(target_3d[:,7]-predicted[:,0],dim=-1)
                    # print(norm_target.shape)
                    # scale = norm_target / (norm_predicted+0.0001)
                    # print(torch.mean(scale))
                    # scale=scale.unsqueeze(-1).unsqueeze(-1)
                    # scale=scale.repeat(1,predicted.shape[1],predicted.shape[2])
                    # print(scale.shape)
                    # predicted=scale* predicted
                    predicted=predicted-predicted[:,:1,:]
                    predicted_rev=predicted.clone().squeeze().cpu().numpy()
                    predicted_rev[:,:,2]*=-1
                    ii=100
                    # plot_18j(np.concatenate((predicted[ii:ii+1,:,:2].clone().cpu().numpy(),predicted_rev[ii:ii+1,:,:2],target_3d[ii:ii+1,:,:2].clone().cpu().numpy()),axis=0))
                    # plot_18j_3d(np.concatenate((predicted[ii:ii+1].clone().cpu().numpy(),predicted_rev[ii:ii+1],target_3d[ii:ii+1].clone().cpu().numpy()),axis=0))
                    

                    e1=p_mpjpe(predicted.squeeze().detach().cpu().numpy(),target_3d.detach().cpu().numpy())
                    e2=p_mpjpe(predicted_rev,target_3d.detach().cpu().numpy())
                    loss_3d_p+=np.mean(np.minimum(e1,e2))*len(e1)*15
                    
                    e1=numpy_nmpjpe(predicted.squeeze().detach().cpu().numpy(),target_3d.detach().cpu().numpy())
                    e2=numpy_nmpjpe(predicted_rev,target_3d.detach().cpu().numpy())
                    loss_3d_n+=np.mean(np.minimum(e1,e2))*len(e1)*15
                    N+=len(e1)*15
            print('PA-MPJPE',loss_3d_p/N,'N-MPJPE',loss_3d_n/N)
           



