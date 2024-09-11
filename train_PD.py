import torch
import torch.nn
import torch.optim
import numpy as np
from torch.utils import data
from utils.data_PD import *
import torch.optim as optim
import model_confidences
from utils.print_losses import print_losses
from types import SimpleNamespace
from pytorch3d.transforms import so3_exponential_map as rodrigues
from numpy.random import default_rng
from utils.camera import *
from utils.loss import *
from utils.plot import *
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = SimpleNamespace()

config.learning_rate = 0.0001
config.BATCH_SIZE = 8
config.N_epochs = 10
config.NoEval=False
# weights for the different losses
config.weight_rep = 1
config.weight_view = 1
config.weight_teacher = 1
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
                # pos_3d=pos_3d[:,joints16]
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

# keypoints = np.load('data/data_2d_h36m_gt.npz', allow_pickle=True)
# keypoints = keypoints['positions_2d'].item()

keypoints_PD = np.load('data/data_PD.npz', allow_pickle=True)
keypoints_PD = keypoints_PD['positions_2d'].item()
num_joints=15
# for subject in keypoints.keys():
#     for action in keypoints[subject]:
#         for cam_idx, kps in enumerate(keypoints[subject][action]):
#             # Normalize camera frame
#             cam = dataset_test.cameras()[subject][cam_idx]
#             kps=kps-kps[:,:1,:]
#             kps=kps[:,joints16,:]
#             kps=np.transpose(kps,[0,2,1])
#             kps=kps.reshape(-1,32)
#             kps/=np.linalg.norm(kps,ord=2,axis=1,keepdims=True)
#             keypoints[subject][action][cam_idx] = kps
len_data=0
for subject in keypoints_PD.keys():
    for action in keypoints_PD[subject]:
        for cam_idx in range(len(keypoints_PD[subject][action]['pos'])):
            # Normalize camera frame
            kps=keypoints_PD[subject][action]['pos'][cam_idx]
            conf=keypoints_PD[subject][action]['conf'][cam_idx]
            kps=kps[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15],:]
            conf=conf[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15],:]
            kps=kps-kps[:,:1,:]
            kps=np.transpose(kps,[0,2,1])
            kps=kps.reshape(-1,num_joints*2)
            kps/=np.linalg.norm(kps,ord=2,axis=1,keepdims=True)
            keypoints_PD[subject][action]['pos'][cam_idx] = kps
            keypoints_PD[subject][action]['conf'][cam_idx] = conf
##########################numpy align#######################################

def procrustes_torch(X, Y):
    """
    Reimplementation of MATLAB's `procrustes` function to Numpy.
    """
    X1=X[:,[0,1,4,7,8,9,10,13]]
    Y1=Y[:,[0,1,4,7,8,9,10,13]]
    batch,n, m = X1.shape
    batch, ny, my = Y1.shape

    muX = torch.mean(X1,dim=1,keepdim=True)
    muY = torch.mean(Y1,dim=1,keepdim=True)

    X0 = X1 - muX
    Y0 = Y1 - muY

    # optimum rotation matrix of Y
    A = torch.matmul(torch.transpose(X0,-1,-2), Y0)
    U,s,V = torch.svd(A)
    T = torch.matmul(V, torch.transpose(U,-1,-2))

    X1=X
    Y1=Y
    muX = torch.mean(X1,dim=1,keepdim=True)
    muY = torch.mean(Y1,dim=1,keepdim=True)

    X0 = X1 - muX
    Y0 = Y1 - muY

    Z = torch.matmul(Y0, T) + muX


    return np.array(Z.cpu())
    
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
            
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                
                
            if parse_3d_poses and 'positions_3d' in dataset_test[subject][action]:
                poses_3d = dataset_test[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    return  np.concatenate(out_poses_2d,axis=0),  np.concatenate( out_poses_3d,axis=0)

def fetch_train(subjects,actions,tag='train'):
    out_subject = []
    out_poses_2d = []
    out_confidences=[]
    for i in range(2):
        out_poses_2d.append([])
        out_confidences.append([])

    for subject in subjects:
        for action in actions:
                
            poses_2d = keypoints_PD[subject][action]['pos']
            conf_2d = keypoints_PD[subject][action]['conf']

            for i in range(2): # Iterate across cameras
                # if i<2:
                out_poses_2d[i].append(poses_2d[i])
                out_confidences[i].append(conf_2d[i])
                # print(len(poses_2d[i]))
                # else:
                #     out_poses_2d[i].append(poses_2d[i-2])
                #     out_confidences.append(conf_2d[i-2])

            out_subject.append(np.ones(len(poses_2d[0]))*int(subject[1:]))
            

    for i in range(len(poses_2d)): # Iterate across cameras
        out_poses_2d[i]=np.concatenate(out_poses_2d[i],axis=0)        
        out_confidences[i]=np.concatenate(out_confidences[i],axis=0) 
        # print(len(out_poses_2d[i]))        
    out_subject=np.concatenate(out_subject,axis=0)
    return  out_poses_2d, out_confidences, out_subject

config.datafile = data_folder + 'detections.pickle'


def loss_weighted_rep_no_scale(p2d, p3d, confs):
    # the weighted reprojection loss as defined in Equation 5

    # normalize by scale
    scale_p2d = torch.sqrt(p2d[:, 0:num_joints*2].square().sum(axis=1, keepdim=True) / num_joints*2)
    p2d_scaled = p2d[:, 0:num_joints*2]/scale_p2d

    # only the u,v coordinates are used and depth is ignored
    # this is a simple weak perspective projection
    scale_p3d = torch.sqrt(p3d[:, 0:num_joints*2].square().sum(axis=1, keepdim=True) / num_joints*2)
    p3d_scaled = p3d[:, 0:num_joints*2]/scale_p3d

    loss = ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, num_joints).sum(axis=1) * confs).sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])

    return loss

# loading the H36M dataset
print('Train Data')
poses_2d_train, conf_2d_train, out_subject= fetch_train(['S01','S02','S25','S27','S28','S29'],['WalkingOval']) #,'S02','S25','S27','S28','S29'
print('Test Data')
# test_subjects_normal=['S22','S23','S24','S25',
#             'S26','S27','S28','S29','S30',
#             'S31','S32','S33','S34','S36',
#             'S38','S39','S40','S41','S42',
#             'S44']
# test_subjects_slight=['S37','S43']
# test_subjects_moderate=['S35']
# test_S22=['S22']
subjects=[['S01'],['S02'],['S03'],['S04'],['S05'],['S06'],
            ['S07'],['S08'],['S09'],['S10'],['S11'],
            ['S12'],['S13'],['S14'],['S15'],['S16'],
            ['S17'],['S18'],['S19'],['S20'],['S21'],
            ['S22'],['S23'],['S24'],['S25'],['S26'],
            ['S27'],['S28'],['S29'],['S30'],['S31'],
            ['S32'],['S33'],['S34'],['S35']]

# subjects=[['S01'],['S02'],['S25'],['S26'],['S27'],['S28'],['S29'],['S31'],
# ['S35']]
# poses_2d_train2, conf_2d_train2, out_subject2= fetch_train(test_subjects_slight,['WalkingOval'])
                                                            
# print(poses_2d_train[0].shape)
# print(subj_train.shape)
my_dataset = H36MDataset(poses_2d_train, conf_2d_train, out_subject, normalize_2d=True)
# my_dataset2 = H36MDataset(poses_2d_train2, conf_2d_train2, out_subject2, normalize_2d=True)
# my_dataset_test = H36MDataset_test(poses_2d_valid,poses_3d_valid)
train_loader = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
# train_loader2 = data.DataLoader(my_dataset2, batch_size=2000, shuffle=False, num_workers=0)
# test_loader = data.DataLoader(my_dataset_test, batch_size=1024 )
# load the skeleton morphing model as defined in Section 4.2
# for another joint detector it needs to be retrained -> train_skeleton_morph.py
model_skel_morph = torch.load('models/model_skeleton_morph_S1_gh.pt')
model_skel_morph.eval()

# loading the lifting network
model_teacher = model_confidences.Lifter().cuda()
model = model_confidences.Lifter().cuda()
model_eval=model_confidences.Lifter().cuda()
params = list(model.parameters())
checkpoint=torch.load('models/model_pretrain.pt')
model_teacher.load_state_dict(checkpoint.state_dict())
model.load_state_dict(checkpoint.state_dict())
optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

losses = SimpleNamespace()
losses_mean = SimpleNamespace()

cam_names = ['54138969', '55011271', '58860488', '60457274']
all_cams = ['cam0', 'cam1']

if config.NoEval:
    for epoch in range(config.N_epochs):

        for i, sample in enumerate(train_loader):

            # not the most elegant way to extract the dictionary
            poses_2d = {key:sample[key] for key in all_cams}

            inp_poses = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), num_joints*2)).cuda()
            inp_confidences = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), num_joints)).cuda()

            # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.
            cnt = 0
            for b in range(poses_2d['cam0'].shape[0]):
                for c_idx, cam in enumerate(poses_2d):
                    inp_poses[cnt] = poses_2d[cam][b]
                    inp_confidences[cnt] = sample['confidences'][c_idx][b]
                    cnt += 1

            # morph the poses using the skeleton morphing network
            # inp_poses = model_skel_morph(inp_poses)
            # print(inp_poses.shape)
            # plot17j_2d(inp_poses[100:101].reshape(1,2,16).transpose(2,1).cpu().detach().numpy())
            # predict 3d poses
    

            pred = model(inp_poses, inp_confidences)
            pred_poses = pred[0]
            pred_cam_angles = pred[1]

            pred_teacher = model_teacher(inp_poses, inp_confidences)
            pred_poses_teacher = pred_teacher[0]
            pred_cam_angles_teacher = pred_teacher[1]
            

            # angles are in axis angle notation
            # use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
            pred_rot = rodrigues(pred_cam_angles)
            pred_rot_teacher = rodrigues(pred_cam_angles_teacher)
            
            # reproject to original cameras after applying rotation to the canonical poses
            rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, num_joints))
            rot_poses_teacher = pred_rot_teacher.matmul(pred_poses_teacher.reshape(-1, 3, num_joints))
            rot_poses_student=torch.transpose(rot_poses,2,1)
            rot_poses_teacher=torch.transpose(rot_poses_teacher,2,1)
            losses.teacher=n_mpjpe(rot_poses_student[:,7:],rot_poses_teacher[:,7:])
            rot_poses=rot_poses.reshape(-1, num_joints*3)
            # reprojection loss
            losses.rep = loss_weighted_rep_no_scale(inp_poses, rot_poses, inp_confidences)

            # view-consistency and camera-consistency
            # to compute the different losses we need to do some reshaping
            pred_poses_rs = pred_poses.reshape((-1, len(all_cams), num_joints*3))
            pred_rot_rs = pred_rot.reshape(-1, len(all_cams), 3, 3)
            confidences_rs = inp_confidences.reshape(-1, len(all_cams), num_joints)
            inp_poses_rs = inp_poses.reshape(-1, len(all_cams), num_joints*2)
            rot_poses_rs = rot_poses.reshape(-1, len(all_cams), num_joints*3)

            # view and camera consistency are computed in the same loop
            losses.view = 0
            losses.camera = 0
            for c_cnt in range(len(all_cams)):
                ## view consistency
                # get all cameras and active cameras
                ac = np.array(range(len(all_cams)))
                coi = np.delete(ac, c_cnt)

                # view consistency
                projected_to_other_cameras = pred_rot_rs[:, coi].matmul(pred_poses_rs.reshape(-1, len(all_cams), 3, num_joints)[:, c_cnt:c_cnt+1].repeat(1, len(all_cams)-1, 1, 1)).reshape(-1, len(all_cams)-1, num_joints*3)
                losses.view += loss_weighted_rep_no_scale(inp_poses.reshape(-1, len(all_cams), num_joints*2)[:, coi].reshape(-1, num_joints*2),
                                                        projected_to_other_cameras.reshape(-1, num_joints*3),
                                                        inp_confidences.reshape(-1, len(all_cams), num_joints)[:, coi].reshape(-1, num_joints))

                ## camera consistency
                relative_rotations = pred_rot_rs[:, coi].matmul(pred_rot_rs[:, [c_cnt]].permute(0, 1, 3, 2))

                # only shuffle in between subjects
                rng = default_rng()
                for subject in sample['subjects'].unique():
                    # only shuffle if enough subjects are available
                    if (sample['subjects'] == subject).sum() > 1:
                        shuffle_subjects = (sample['subjects'] == subject)
                        num_shuffle_subjects = shuffle_subjects.sum()
                        rand_perm = rng.choice(num_shuffle_subjects.cpu().numpy(), size=num_shuffle_subjects.cpu().numpy(), replace=False)
                        samp_relative_rotations = relative_rotations[shuffle_subjects]
                        samp_rot_poses_rs = rot_poses_rs[shuffle_subjects]
                        samp_inp_poses = inp_poses_rs[shuffle_subjects][:, coi].reshape(-1, num_joints*2)
                        samp_inp_confidences = confidences_rs[shuffle_subjects][:, coi].reshape(-1, num_joints)

                        random_shuffled_relative_projections = samp_relative_rotations[rand_perm].matmul(samp_rot_poses_rs.reshape(-1, len(all_cams), 3, num_joints)[:, c_cnt:c_cnt+1].repeat(1, len(all_cams)-1, 1, 1)).reshape(-1, len(all_cams)-1, num_joints*3)

                        losses.camera += loss_weighted_rep_no_scale(samp_inp_poses,
                                                                    random_shuffled_relative_projections.reshape(-1, num_joints*3),
                                                                    samp_inp_confidences)

            # get combined loss
            losses.loss = config.weight_rep * losses.rep + \
                        config.weight_view * losses.view +\
                        config.weight_teacher*losses.teacher
                        #+ \
                        #config.weight_camera * losses.camera

            optimizer.zero_grad()
            losses.loss.backward()

            optimizer.step()

            for key, value in losses.__dict__.items():
                if key not in losses_mean.__dict__.keys():
                    losses_mean.__dict__[key] = []

                losses_mean.__dict__[key].append(value.item())

            # print progress every 100 iterations
            if not i % 100:
                # print the losses to the console
                print_losses(epoch, i, len(my_dataset) / config.BATCH_SIZE, losses_mean.__dict__, print_keys=not(i % 1000))

                # this line is important for logging!
                losses_mean = SimpleNamespace()

        # save the new trained model every epoch
        torch.save(model, 'models/model_lifter.pt')

        scheduler.step()
        if epoch>config.N_epochs-2:
            with torch.no_grad():
                
                model_eval.load_state_dict(model.state_dict())
                model_eval.eval()
                N=0
                loss_3d_tot=0
                for subj in subjects:
                    pred_save=[]
                    poses_2d_train2, conf_2d_train2, out_subject2= fetch_train(subj,['WalkingOval'])
                    my_dataset2 = H36MDataset(poses_2d_train2, conf_2d_train2, out_subject2, normalize_2d=True)
                    train_loader2 = data.DataLoader(my_dataset2, batch_size=2000, shuffle=False, num_workers=0)                
                    for i, sample in enumerate(train_loader2):
                        # input_2d=model_skel_morph(sample['poses_2d'].cuda())
                        # not the most elegant way to extract the dictionary
                        poses_2d = {key:sample[key] for key in all_cams}

                        inp_poses = torch.zeros((poses_2d['cam0'].shape[0] , num_joints*2)).cuda()
                        inp_confidences = torch.zeros((poses_2d['cam0'].shape[0] , num_joints)).cuda()

                        # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.
                        cnt = 0
                        for b in range(poses_2d['cam0'].shape[0]):

                            inp_poses[cnt] = poses_2d['cam0'][b]
                            inp_confidences[cnt] = sample['confidences'][0][b]
                            cnt += 1

                        # morph the poses using the skeleton morphing network
                        # inp_poses = model_skel_morph(inp_poses)
                        # print(inp_poses.shape)
                        # plot17j_2d(inp_poses[100:101].reshape(1,2,16).transpose(2,1).cpu().detach().numpy())
                        # predict 3d poses
                        pred = model(inp_poses, inp_confidences)
                        pred_poses = pred[0]
                        pred_cam_angles = pred[1]

                        # angles are in axis angle notation
                        # use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
                        pred_rot = rodrigues(pred_cam_angles)

                        # reproject to original cameras after applying rotation to the canonical poses
                        rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, num_joints))

                        rot_poses=torch.transpose(rot_poses,2,1)
                        # loss_3d=n_mpjpe(rot_poses,sample['poses_3d'])
                        # A=sample['poses_3d'][100:101].cpu().numpy()

                        pred_aligned=procrustes_torch(rot_poses[0:1],rot_poses)
                        pred_save.append(pred_aligned)

                        
                        ref_vec=[0,1,0]
                        rot_poses-=rot_poses[:,:1]
                        pred_aligned=procrustes_torch(rot_poses[0:1],rot_poses)
                        pred_save.append(pred_aligned)
                        # plot16j_PD(pred_aligned, show_animation=True)
                        # import matplotlib.pyplot as plt
                        # plt.plot(B[:,2,:])
                        # plt.show()
                #     loss_3d_tot+=loss_3d*sample['poses_3d'].shape[0]
                #     N+=sample['poses_3d'].shape[0]
                # print('Error 3D',loss_3d_tot/N*1000)
                    print(subj[0])
                    np.save('Predictions_'+subj[0],np.concatenate(pred_save))
    print('done')

else:
    with torch.no_grad():
        checkpoint=torch.load('models/model_pretrain.pt')
        model.load_state_dict(checkpoint.state_dict())
        model.eval()
        
        N=0
        loss_3d_tot=0
        for subj in subjects:
            pred_save=[]
            poses_2d_train2, conf_2d_train2, out_subject2= fetch_train(subj,['WalkingOval'])
            my_dataset2 = H36MDataset(poses_2d_train2, conf_2d_train2, out_subject2, normalize_2d=True)
            train_loader2 = data.DataLoader(my_dataset2, batch_size=2000, shuffle=False, num_workers=0)                
            for i, sample in enumerate(train_loader2):
                # input_2d=model_skel_morph(sample['poses_2d'].cuda())
                # not the most elegant way to extract the dictionary
                poses_2d = {key:sample[key] for key in all_cams}

                inp_poses = torch.zeros((poses_2d['cam0'].shape[0] , num_joints*2)).cuda()
                inp_confidences = torch.zeros((poses_2d['cam0'].shape[0] , num_joints)).cuda()

                # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.
                cnt = 0
                for b in range(poses_2d['cam0'].shape[0]):

                    inp_poses[cnt] = poses_2d['cam0'][b]
                    inp_confidences[cnt] = sample['confidences'][0][b]
                    cnt += 1

                # morph the poses using the skeleton morphing network
                # inp_poses = model_skel_morph(inp_poses)
                # print(inp_poses.shape)
                # plot17j_2d(inp_poses[100:101].reshape(1,2,16).transpose(2,1).cpu().detach().numpy())
                # predict 3d poses
                pred = model(inp_poses, inp_confidences)
                pred_poses = pred[0]
                pred_cam_angles = pred[1]

                # angles are in axis angle notation
                # use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
                pred_rot = rodrigues(pred_cam_angles)

                # reproject to original cameras after applying rotation to the canonical poses
                rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, num_joints))

                rot_poses=torch.transpose(rot_poses,2,1)
                # loss_3d=n_mpjpe(rot_poses,sample['poses_3d'])
                # A=sample['poses_3d'][100:101].cpu().numpy()

                # pred_aligned=procrustes_torch(rot_poses[0:1],rot_poses)
                # pred_save.append(pred_aligned)

                
                ref_vec=[0,1,0]
                rot_poses-=rot_poses[:,:1]
                pred_aligned=procrustes_torch(rot_poses[0:1],rot_poses)
                pred_save.append(pred_aligned)
                # plot16j_PD(pred_aligned, show_animation=True)
                # import matplotlib.pyplot as plt
                # plt.plot(B[:,2,:])
                # plt.show()
        #     loss_3d_tot+=loss_3d*sample['poses_3d'].shape[0]
        #     N+=sample['poses_3d'].shape[0]
        # print('Error 3D',loss_3d_tot/N*1000)
            print(subj[0],'-Len data:',len(pred_aligned))
            np.save('Predictions_'+subj[0],np.concatenate(pred_save))
