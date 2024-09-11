import torch
import torch.nn
import torch.optim
import numpy as np
from torch.utils import data
from utils.data import *
import torch.optim as optim
import model_confidences
from model_confidences import *
from utils.print_losses import print_losses
from types import SimpleNamespace
from pytorch3d.transforms import so3_exponential_map as rodrigues
import pytorch3d
from numpy.random import default_rng
from utils.camera import *
from utils.loss import *
from utils.plot import *
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
with torch.autograd.set_detect_anomaly(True):
    config = SimpleNamespace()

    config.learning_rate = 0.0001
    config.BATCH_SIZE = 256
    config.N_epochs = 100
    config.NoEval=True
    # weights for the different losses
    config.weight_rep = 1
    config.weight_view = 1
    config.weight_camera = 0.1

    data_folder = './data/'
    joints16=[4,5,6,1,2,3,0,8,9,10,11,12,13,14,15,16]
    dataset_test = Human36mDataset('data/data_3d_h36m.npz')
    for subject in dataset_test.subjects():
        for action in dataset_test[subject].keys():
            anim = dataset_test[subject][action]
            if 'positions' in anim:
                positions_3d = []
                ii=0
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d -= pos_3d[:, :1,:] # Remove global offset, but keep trajectory in first position
                    pos_3d=pos_3d[:,joints16]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    keypoints = np.load('data/data_2d_h36m_gt.npz', allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset_test.cameras()[subject][cam_idx]
                kps=kps-kps[:,:1,:]
                kps=kps[:,joints16,:]
                kps=np.transpose(kps,[0,2,1])
                kps=kps.reshape(-1,32)
                kps/=np.linalg.norm(kps,ord=2,axis=1,keepdims=True)
                keypoints[subject][action][cam_idx] = kps

    def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
        out_poses_3d = []
        out_poses_2d = []
        for subject in subjects:
            for action in dataset_test[subject].keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue
                    
                poses_2d = keypoints[subject][action]
                
                # for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(np.concatenate((poses_2d[0],poses_2d[1],poses_2d[2],poses_2d[3]),axis=-1))
                    
                    
                if parse_3d_poses and 'positions_3d' in dataset_test[subject][action]:
                    poses_3d = dataset_test[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    # for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(np.concatenate((poses_3d[0],poses_3d[1],poses_3d[2],poses_3d[3]),axis=-1))

        return  np.concatenate(out_poses_2d,axis=0),  np.concatenate( out_poses_3d,axis=0)

    def fetch_train(subjects, action_filter=None, subset=1, parse_3d_poses=True):
        out_subject = []
        out_poses_2d = []
        for i in range(4):
            out_poses_2d.append([])
        for subject in subjects:
            for action in dataset_test[subject].keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue
                    
                poses_2d = keypoints[subject][action]

                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d[i].append(poses_2d[i])

                out_subject.append(np.ones(len(poses_2d[0]))*int(subject[-1]))
                    
        for i in range(len(poses_2d)): # Iterate across cameras
            out_poses_2d[i]=np.concatenate(out_poses_2d[i],axis=0)               
        out_subject=np.concatenate(out_subject,axis=0)
        return  out_poses_2d, out_subject

    config.datafile = data_folder + 'detections.pickle'


    def loss_weighted_rep_no_scale(p2d, p3d, confs,ave=True,diff=False):
        # the weighted reprojection loss as defined in Equation 5

        # normalize by scale
        scale_p2d = torch.sqrt(p2d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)+0.000001
        p2d_scaled = p2d[:, 0:32]/scale_p2d

        # only the u,v coordinates are used and depth is ignored
        # this is a simple weak perspective projection
        scale_p3d = torch.sqrt(p3d[:, 0:32].square().sum(axis=1, keepdim=True) / 32)+0.000001
        p3d_scaled = p3d[:, 0:32]/scale_p3d

        if diff:
            p2d_scaled_diff=p2d_scaled.reshape(-1,200,[1:]-p2d_scaled[:-1]
            p3d_scaled_diff=p3d_scaled[1:]-p3d_scaled[:-1]

            loss = ((p2d_scaled_diff - p3d_scaled_diff).abs().reshape(-1, 2, 16).sum(axis=1) ).sum() / (p2d_scaled_diff.shape[0] * p2d_scaled_diff.shape[1])
       
        elif ave:
            loss = ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, 16).sum(axis=1) * confs).sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])

        else:
            loss = ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, 16).sum(axis=1) * confs).sum(axis=-1) / (32)


        return loss
    


    # loading the H36M dataset
    poses_2d_valid, poses_3d_valid= fetch(subjects=['S11','S9'])
    poses_2d_train, subj_train= fetch_train(subjects=['S1','S5','S6','S7','S8'])
    # print(poses_2d_train[0].shape)
    # print(subj_train.shape)
    my_dataset = H36MDataset(poses_2d_train, subj_train, normalize_2d=True)
    my_dataset_test = H36MDataset_test(poses_2d_valid,poses_3d_valid)
    train_loader = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    train_loader2 = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(my_dataset_test, batch_size=config.BATCH_SIZE, shuffle=False)
    # load the skeleton morphing model as defined in Section 4.2
    # for another joint detector it needs to be retrained -> train_skeleton_morph.py
    model_skel_morph = torch.load('models/model_skeleton_morph_S1_gh.pt')
    model_skel_morph.eval()

    # loading the lifting network
    model = model_confidences.Lifter_Multi_View().cuda()
    model_eval=model_confidences.Lifter_Multi_View().cuda()
    filter_widths=[1,1,1,1,1]
    model_single_view=TemporalModelOptimized1f(16,2,16,filter_widths=filter_widths).cuda()
    model_single_view_eval=TemporalModelOptimized1f(16,2,16,filter_widths=filter_widths).cuda()
    params = list(model.parameters())
    params_single_view = list(model_single_view.parameters())

    optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=1e-5)
    optimizer_single_view = optim.Adam(params_single_view, lr=config.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    losses = SimpleNamespace()
    losses_mean = SimpleNamespace()

    cam_names = ['54138969', '55011271', '58860488', '60457274']
    all_cams = ['cam0', 'cam1', 'cam2', 'cam3']

    if config.NoEval:
        for epoch in range(config.N_epochs):
            
            
            # for i, sample in enumerate(train_loader):
            for i, (sample,sample2) in enumerate(zip(train_loader, train_loader2)):
                for p in model.parameters():
                    p.requires_grad = True  # to avoid computation
                # not the most elegant way to extract the dictionary
                
                ii=0
                division_point=sample['cam0'].shape[0]
                for key in all_cams:
                    sample[key]=torch.cat((sample[key],sample2[key]),dim=0)
                    sample['confidences'][ii]=torch.cat((sample['confidences'][ii],sample2['confidences'][ii]),dim=0)
                    ii+=1
                
                poses_2d = {key:sample[key] for key in all_cams}
                
                inp_poses = torch.zeros((poses_2d['cam0'].shape[0], len(all_cams)* 32)).cuda()
                inp_confidences = torch.zeros((poses_2d['cam0'].shape[0], len(all_cams)* 16)).cuda()

                


                # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.

                cnt = 0
                for b in range(poses_2d['cam0'].shape[0]):
                    idx=np.arange(len(poses_2d))
                    np.random.shuffle(idx)
                    inp_poses[cnt] = torch.cat((poses_2d['cam0'][b:b+1],poses_2d['cam1'][b:b+1],poses_2d['cam2'][b:b+1],poses_2d['cam3'][b:b+1]),dim=-1).contiguous()
                    inp_confidences[cnt] = torch.cat((sample['confidences'][0][b:b+1],sample['confidences'][1][b:b+1],sample['confidences'][2][b:b+1],sample['confidences'][3][b:b+1]),dim=-1)
                    cnt += 1
                

                inp_poses_shuffled=inp_poses
                inp_confidences_shuffled=inp_confidences

                pred = model(inp_poses_shuffled, inp_confidences_shuffled)
                pred_poses = pred[0]
                pred_cam_angles = pred[1]
                losses.rep=0
                pred_rot=torch.zeros((pred_cam_angles.shape[0],len(all_cams),3,3)).cuda()
                rot_poses=torch.zeros((pred_cam_angles.shape[0],len(all_cams),48)).cuda()
                rec=torch.zeros((4))
                for ii in range(len(all_cams)):
                    pred_rot[:,ii] = rodrigues(pred_cam_angles[:,ii])
                    rot_poses_ii = pred_rot[torch.randperm(rot_poses.shape[0]),ii].matmul(pred_poses.reshape(-1, 3, 16))
                    bone_pred=torch.linalg.norm(rot_poses_ii[:,:,[7]]-rot_poses_ii[:,:,[6]],dim=-2,keepdim=True)
                    rot_poses_ii= rot_poses_ii/bone_pred*0.52
                    rot_poses[:,ii] =rot_poses_ii.reshape(-1, 48)

            
                losses.rep= loss_weighted_rep_no_scale(inp_poses_shuffled[0:division_point,0:32], rot_poses[0:division_point,0], inp_confidences_shuffled[0:division_point,0:16])+\
                    loss_weighted_rep_no_scale(inp_poses_shuffled[0:division_point,32:64], rot_poses[0:division_point,1], inp_confidences_shuffled[0:division_point,16:32])+\
                    loss_weighted_rep_no_scale(inp_poses_shuffled[0:division_point,64:96], rot_poses[0:division_point,2], inp_confidences_shuffled[0:division_point,32:48])+\
                    loss_weighted_rep_no_scale(inp_poses_shuffled[0:division_point,96:128], rot_poses[0:division_point,3], inp_confidences_shuffled[0:division_point,48:64])

                losses.rep_diff= loss_weighted_rep_no_scale(inp_poses_shuffled[division_point:,0:32], rot_poses[division_point:,0], inp_confidences_shuffled[division_point:,0:16],diff=True)+\
                    loss_weighted_rep_no_scale(inp_poses_shuffled[division_point:,32:64], rot_poses[division_point:,1], inp_confidences_shuffled[division_point:,16:32],diff=True)+\
                    loss_weighted_rep_no_scale(inp_poses_shuffled[division_point:,64:96], rot_poses[division_point:,2], inp_confidences_shuffled[division_point:,32:48],diff=True)+\
                    loss_weighted_rep_no_scale(inp_poses_shuffled[division_point:,96:128], rot_poses[division_point:,3], inp_confidences_shuffled[division_point:,48:64],diff=True)
               
                # get combined loss
                # get combined loss
                # if epoch>0:
                losses.loss = losses.rep + losses.rep_diff*0.1
                # else:
                    # losses.loss = losses.rep

                if epoch<3:
                    optimizer.zero_grad()
                    losses.loss.backward()
                    optimizer.step()         
                
                for p in model.parameters():
                    p.requires_grad = False  # to avoid computation

                # losses_rec = loss_weighted_rep_no_scale(inp_poses.reshape(-1,32), rot_poses.reshape(-1,48), inp_confidences.reshape(-1,16),ave=False)
                # if epoch> 1:
                #     rows=((losses_rec < 0.08).nonzero(as_tuple=True)[0]) 
                # else:
                #     rows=((losses_rec < 5).nonzero(as_tuple=True)[0])
                
                # rot_poses_shuffled=rot_poses.clone()
                # rot_poses[shuffled]= rot_poses_shuffled
                
                # new_pose_2d=rot_poses.reshape(-1,48)[:,:32]/torch.linalg.norm(rot_poses.reshape(-1,48)[:,:32],ord=2,dim=1,keepdim=True)
                # new_pose_2d=(new_pose_2d.reshape(-1,2,16)-new_pose_2d.reshape(-1,2,16)[:,:,[6]]).transpose(2,1).detach()
                loss_single_view=torch.zeros(4)
                for ii in range(4):
                    new_pose_3d=rot_poses[0:division_point,ii].reshape(-1,48).reshape(-1,3,16).transpose(2,1).detach()
                    new_pose_2d=inp_poses_shuffled[0:division_point,ii*32:ii*32+32].reshape(-1,1,32).reshape(-1,1,2,16).transpose(3,2).detach()
                    
                    # pad=13
                    # new_pose_2d_=torch.zeros((new_pose_2d.shape[0],2*pad+1,new_pose_2d.shape[1],new_pose_2d.shape[2])).cuda()
                    # for ii in range(pad,len(new_pose_2d)-pad):
                    #     new_pose_2d_[ii]=new_pose_2d[ii-pad:ii+pad+1]

                    


                    # rows=torch.randperm(len(new_pose_2d))
                    pred_pose_3d=model_single_view(new_pose_2d).squeeze()

                    # axis_angle=torch.rand(new_pose_3d.shape[0],3).cuda()-0.5
                    # theta=torch.rand(new_pose_3d.shape[0],1).cuda()*3.14159
                    # axis_angle=axis_angle/torch.linalg.norm(axis_angle,dim=-1,keepdim=True)*theta
                    # rand_rot=pytorch3d.transforms.axis_angle_to_matrix(axis_angle)
                    # rand_pose_3d=(rand_rot.matmul(new_pose_3d.transpose(2,1))).transpose(2,1)
                    # rand_pose_2d=rand_pose_3d.reshape(-1,48)[:,:32]/torch.linalg.norm(rand_pose_3d.reshape(-1,48)[:,:32],ord=2,dim=1,keepdim=True)
                    # rand_pose_2d=(rand_pose_2d.reshape(-1,2,16)-rand_pose_2d.reshape(-1,2,16)[:,:,[6]]).transpose(2,1).detach()
                    # pred_rand=model_single_view(rand_pose_2d).squeeze()

                    
                    loss_single_view[ii]=n_mpjpe(pred_pose_3d,new_pose_3d) #+n_mpjpe(pred_rand,rand_pose_3d)*0.1
                loss_single_view=(loss_single_view[0]+loss_single_view[1]+loss_single_view[2]+loss_single_view[3])/4
                # if epoch>=3:
                optimizer_single_view.zero_grad()
                loss_single_view.backward()
                optimizer_single_view.step()
                
                for key, value in losses.__dict__.items():
                    if key not in losses_mean.__dict__.keys():
                        losses_mean.__dict__[key] = []

                    losses_mean.__dict__[key].append(value.item())

                # print progress every 100 iterations
                if not i % 100:
                    # print the losses to the console
                    print_losses(epoch, i, len(my_dataset) / config.BATCH_SIZE, losses_mean.__dict__, print_keys=not(i % 1000))
                    print('loss 3D', loss_single_view.item())
                    # this line is important for logging!
                    losses_mean = SimpleNamespace()

            # save the new trained model every epoch
            torch.save(model, 'models/model_lifter.pt')
            torch.save(model_single_view, 'models/model_lifter_single_view.pt')

            scheduler.step()
            with torch.no_grad():
                model_eval.load_state_dict(model.state_dict())
                model_eval.eval()
                model_single_view_eval.load_state_dict(model_single_view.state_dict())
                model_single_view_eval.eval()
                N=0
                loss_3d_tot=0
                loss_3d=0
                loss_3d_single_view=0
                pad=0
                for i, sample in enumerate(test_loader):
                    # input_2d=model_skel_morph(sample['poses_2d'].cuda())

                    pred=model_eval(sample['poses_2d'].squeeze().cuda(),sample['confidences'].cuda())
                    pred_poses = pred[0]
                    pred_cam_angles = pred[1]

                    # # angles are in axis angle notation
                    # # use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
                    # pred_rot = rodrigues(pred_cam_angles)

                    # # reproject to original cameras after applying rotation to the canonical poses
                    # rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, 16))
                    pred_rot=torch.zeros((pred_cam_angles.shape[0],len(all_cams),3,3)).cuda()
                    rot_poses=torch.zeros((pred_cam_angles.shape[0],len(all_cams),16,3)).cuda()
                    rec=torch.zeros((4))
                    for ii in range(len(all_cams)):
                        pred_rot[:,ii] = rodrigues(pred_cam_angles[:,ii])
                        rot_poses[:,ii] = pred_rot[:,ii].matmul(pred_poses.reshape(-1, 3, 16)).transpose(2,1)
                        
                        loss_3d+=n_mpjpe(rot_poses[:,ii],sample['poses_3d'][:,:,ii*3:ii*3+3])*sample['poses_3d'].shape[0]
                        N+=sample['poses_3d'].shape[0]

                    # A=sample['poses_3d'][100:101].cpu().numpy()
                    # B=rot_poses[100:101].cpu().numpy()
                    # plot17j(np.concatenate((A,B),axis=0), show_animation=False)
                    # loss_3d_tot+=loss_3d
                    
                    # print(poses_2d.shape)
                    for ii in range(len(all_cams)):                       
                        pred_single_view=model_single_view_eval(sample['poses_2d'][:,ii*32:ii*32+32].reshape(-1,1,2,16).transpose(3,2).cuda())

                        loss_3d_single_view+=n_mpjpe(pred_single_view.squeeze(),sample['poses_3d'][:,:,ii*3:ii*3+3])*sample['poses_3d'].shape[0]
                print('Error 3D',loss_3d/N*1000, loss_3d_single_view/N*1000)
        print('done')

    else:
        with torch.no_grad():
            checkpoint=torch.load('models/model_lifter.pt')
            model.load_state_dict(checkpoint.state_dict())
            model.eval()
        

            checkpoint=torch.load('models/model_lifter_single_view.pt')
            model_single_view.load_state_dict(checkpoint.state_dict())
            model_single_view.eval()
        
            N=0
            loss_3d_tot=0
            loss_3d=0
            loss_3d_single_view=0
            pad=0
            for i, sample in enumerate(test_loader):
                # input_2d=model_skel_morph(sample['poses_2d'].cuda())
                pred=model_eval(sample['poses_2d'][:,0].cuda(),sample['confidences'].cuda())
                pred_poses = pred[0]
                pred_cam_angles = pred[1]

                # # angles are in axis angle notation
                # # use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
                # pred_rot = rodrigues(pred_cam_angles)

                # # reproject to original cameras after applying rotation to the canonical poses
                # rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, 16))
                pred_rot=torch.zeros((pred_cam_angles.shape[0],len(all_cams),3,3)).cuda()
                rot_poses=torch.zeros((pred_cam_angles.shape[0],len(all_cams),16,3)).cuda()
                rec=torch.zeros((4))
                for ii in range(len(all_cams)):
                    pred_rot[:,ii] = rodrigues(pred_cam_angles[:,ii])
                    rot_poses[:,ii] = pred_rot[:,ii].matmul(pred_poses.reshape(-1, 3, 16)).transpose(2,1)

                    loss_3d+=n_mpjpe(rot_poses[:,ii],sample['poses_3d'][:,:,ii*3:ii*3+3])*sample['poses_3d'].shape[0]
                    N+=sample['poses_3d'].shape[0]

                # A=sample['poses_3d'][100:101].cpu().numpy()
                # B=rot_poses[100:101].cpu().numpy()
                # plot17j(np.concatenate((A,B),axis=0), show_animation=False)
                # loss_3d_tot+=loss_3d
                
                # print(poses_2d.shape)
                for ii in range(len(all_cams)):                       
                    pred_single_view=model_single_view_eval(sample['poses_2d'][:,:,ii*32:ii*32+32].reshape(-1,2*pad+1,2,16).transpose(-2,-1).cuda())
                    loss_3d_single_view+=n_mpjpe(pred_single_view.squeeze(),sample['poses_3d'][:,:,ii*3:ii*3+3])*sample['poses_3d'].shape[0]
            print('Error 3D',loss_3d/N*1000, loss_3d_single_view/N*1000)
    print('done')