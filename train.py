import torch
import torch.nn
import torch.optim
import numpy as np
from torch.utils import data
from utils.data import *
import torch.optim as optim
import model_confidences
from utils.print_losses import print_losses
from types import SimpleNamespace
from pytorch3d.transforms import so3_exponential_map as rodrigues
import pytorch3d.transforms as transform
from numpy.random import default_rng
from utils.camera import *
from utils.loss import *
from utils.plot import *
from utils.correct_action import *
from utils.epipolar import *
import copy
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
num_joints=15
joints16=[4,5,6,1,2,3,0,8,9,10,11,12,13,14,15,16]
joints_15sh=[6,3,4,5,2,1,0,8,8,9,12,11,10,13,14,15]  

triangulated_poses=np.load('data/triangulated_3d_h36m_ap.npz',allow_pickle=True)
triangulated_poses=triangulated_poses['positions_3d'].item()

dataset_test = np.load('data/data_3d_h36m_ap.npz',allow_pickle=True)
dataset_test = dataset_test['positions_3d'].item()
print('Bone Length Calulation...')
bone_length=[]
for subject in dataset_test.keys():
    if subject!='S9' and subject!='S11':
        for action in dataset_test[subject].keys():
            anim = dataset_test[subject][action][0]
            bone_length.append(np.mean(np.linalg.norm(anim[:,0,:]-anim[:,7,:],axis=-1)))
bone_real= np.mean(bone_length) 

print('bone_real:',bone_real) 
error=0
N=0
for subject in dataset_test.keys():
    for action in dataset_test[subject].keys():
        anim_triang=triangulated_poses[subject][action]['positions_triang']
        anim = dataset_test[subject][action]
        # if 'positions' in anim:
        positions_3d = []
        positions_3d_triang=[]
        ii=0

        for cam in range(4):

            pos_3d = anim[ii]
            pos_3d = pos_3d-pos_3d[:, 0:1,:] # Remove global offset, but keep trajectory in first position
            positions_3d.append(pos_3d)

            pos_3d_triang=anim_triang[ii] - anim_triang[ii][:, 0:1,:]
            
            bone_triang=np.mean(np.linalg.norm(pos_3d_triang[0,0,:]-pos_3d_triang[0,7,:],axis=-1))
            pos_3d_triang=pos_3d_triang*bone_real/bone_triang                
            pos_3d_triang-=pos_3d_triang[:,0:1,:]
                                           
            positions_3d_triang.append(pos_3d_triang)
            if subject in ['S1','S5','S6','S7','S8']: 
                n=len(pos_3d_triang)*num_joints
                if subject=='S5' and action in ['Sitting', 'Sitting 1']:
                    pass
                else:
                    N+=n
                    error+=numpy_nmpjpe(pos_3d_triang,pos_3d)*1000*n
                    

            ii+=1
        anim['positions_3d'] = positions_3d
        anim['positions_triang'] = positions_3d
print('Triangulation Error',error/N)
keypoints = np.load('data/data_2d_h36m_ap.npz', allow_pickle=True)
keypoints = keypoints['positions_2d'].item()


for subject in dataset_test.keys():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset_test[subject].keys():
        action_corrected=correct_action[subject][action]
        assert action_corrected in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions' not in dataset_test[subject][action]:
            continue
            
        for cam_idx in range(len(keypoints[subject][action_corrected])):
            
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset_test[subject][action]['positions_3d'].shape[0]
            
            # assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
            if keypoints[subject][action_corrected][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action_corrected][cam_idx] = keypoints[subject][action_corrected][cam_idx][:mocap_length]

###### keypoints and confidences
error=0
confidences={}
N=0
for subject in dataset_test.keys():
    confidences[subject]={}
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset_test[subject].keys():
        action_corrected=correct_action[subject][action]
        # action_corrected=action
        confidences[subject][action_corrected]=[]
        kps_confs=keypoints[subject][action_corrected]
       

        for cam_idx in range(4):
            # Normalize camera frame
            confidences[subject][action_corrected].append([])
            kps = copy.deepcopy(kps_confs)
            conf= copy.deepcopy(kps_confs)
            kps=kps[cam_idx][:,:,:2]
            # conf=conf[cam_idx][:,:,2]
            conf=np.ones((kps.shape[0],15))

            kps = kps-kps[:,0:1]

            kps=np.transpose(kps,[0,2,1])
            kps=kps.reshape(-1,num_joints*2)
            kps=kps/(np.linalg.norm(kps,ord=2,axis=1,keepdims=True)+0.0001)
            keypoints[subject][action_corrected][cam_idx] = kps
            confidences[subject][action_corrected][cam_idx] = conf

####################################
def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_conf = []
    for subject in subjects:
        for action in dataset_test[subject].keys():
            action_corrected=correct_action[subject][action]
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
                
            poses_2d = keypoints[subject][action_corrected]
            conf= confidences[subject][action_corrected]
            
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_conf.append(conf[i])
                
                
            if parse_3d_poses and 'positions_3d' in dataset_test[subject][action]:
                poses_3d = dataset_test[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    return  np.concatenate(out_poses_2d,axis=0), np.concatenate(out_conf,axis=0), np.concatenate( out_poses_3d,axis=0)

def fetch_3dhp(data_path,keypoints):
    data3d = np.load(data_path, allow_pickle=True)
    data3d = data3d['positions_3d'].item()
    data2d=keypoints

    data_3d=[]
    data_2d=[] 
    subjects=['TS4']  
    for subject in subjects:
        data3d[subject]=data3d[subject][:,[0,4,5,6,1,2,3,8,9,13,14,15,10,11,12]]
        data3d[subject]-=data3d[subject][:,:1]
        row=np.sum(np.abs(data2d[subject])>1,axis=(-1,-2))==0
        data_3d.append(data3d[subject][row]/1000)
        data_2d.append(data2d[subject][row])   
   
        
    return np.concatenate(data_3d[0],axis=0),np.concatenate(data_2d[0],axis=0)
def create_2d_data_3dhp_test(data_path):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()
    image_shape={'TS1':[2048,2048],'TS2':[2048,2048],'TS3':[2048,2048],'TS4':[2048,2048],'TS5':[1080,1920],'TS6':[1080,1920]}
     
    for subject in keypoints.keys():
        print(subject)

        keypoints[subject] = keypoints[subject][:,[0,4,5,6,1,2,3,8,9,13,14,15,10,11,12],:] 
        keypoints[subject]= keypoints[subject][:,:,:] -keypoints[subject][:,:1,:]
        keypoints[subject] = np.transpose(keypoints[subject][:,:,:],(0,2,1))
        keypoints[subject] = np.reshape(keypoints[subject],(-1,30))
        keypoints[subject] = keypoints[subject]/(np.linalg.norm(keypoints[subject],ord=2,axis=1,keepdims=True)+0.0001)
        print(keypoints[subject].shape)
    return keypoints

def fetch_train(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_subject = []
    out_poses_2d = []
    out_poses_3d = []
    out_conf= []
    for i in range(4):
        out_poses_2d.append([])
        out_poses_3d.append([])
        out_conf.append([])
    for subject in subjects:
        for action in dataset_test[subject].keys():
            action_corrected=correct_action[subject][action]
            # action_corrected=action
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
            if subject=='S5':
                if action in ['Sitting','Sitting 1']:
                    continue    
            poses_2d = keypoints[subject][action_corrected]
            conf=confidences[subject][action_corrected]

            for i in range(len(poses_2d)): # Iterate across cameras
                
                out_poses_2d[i].append(poses_2d[i])
                out_conf[i].append(conf[i])

            if parse_3d_poses and 'positions_3d' in dataset_test[subject][action]:
                poses_3d = dataset_test[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d[i].append(poses_3d[i])

            out_subject.append(np.ones(len(poses_2d[0]))*int(subject[-1]))
                
    for i in range(len(poses_2d)): # Iterate across cameras
        out_poses_2d[i]=np.concatenate(out_poses_2d[i],axis=0) 
        out_conf[i]=np.concatenate(out_conf[i],axis=0)   
        out_poses_3d[i]=np.concatenate(out_poses_3d[i],axis=0)             
    out_subject=np.concatenate(out_subject,axis=0)
    return  out_poses_3d, out_poses_2d, out_conf, out_subject

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
poses_2d_valid, poses_conf_valid, poses_3d_valid= fetch(subjects=['S9', 'S11'])
poses_3d_train, poses_2d_train,poses_2d_conf, subj_train= fetch_train(subjects=['S1', 'S5', 'S6', 'S7', 'S8'])
my_dataset = H36MDataset(poses_3d_train,poses_2d_train,poses_2d_conf, subj_train, normalize_2d=True)
my_dataset_test = H36MDataset_test(poses_2d_valid,  None, poses_3d_valid)
train_loader = data.DataLoader(my_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = data.DataLoader(my_dataset_test, batch_size=1000)
# load the skeleton morphing model as defined in Section 4.2
# for another joint detector it needs to be retrained -> train_skeleton_morph.py
model_skel_morph = torch.load('models/model_skeleton_morph_S1_gh.pt')
model_skel_morph.eval()

# loading the lifting network
model = model_confidences.Lifter().cuda()
model_eval=model_confidences.Lifter().cuda()
params = list(model.parameters())
checkpoint=torch.load('models/model_pretrain.pt')
model.load_state_dict(checkpoint.state_dict())
optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

losses = SimpleNamespace()
losses_mean = SimpleNamespace()


K1=np.empty((3,3))
f1=np.array([1145.0494384765625, 1143.7811279296875]) #"/1000 ";
c1= np.array([512.54150390625, 515.4514770507812])#"/1000-np.array([512.54150390625, 515.4514770507812])/1000"; 
K1[0,0]=f1[0]; K1[1,1]=f1[1];K1[0,2]=c1[0]; K1[1,2]=c1[1]; K1[2,2]=1; 

K2=np.empty((3,3))
f2=np.array([1149.67569986785, 1147.59161666764]) #"/1000" ;  
c2=np.array([508.848621645943, 508.064917088557]) #"/1000-np.array([512.541504956548, 515.4514869776])/1000"; 
K2[0,0]=f2[0]; K2[1,1]=f2[1];K2[0,2]=c2[0]; K2[1,2]=c2[1]; K2[2,2]=1; 


K3=np.empty((3,3))
f3=[1149.1407470703125, 1148.7989501953125] 
c3=[519.8158569335938, 501.40264892578125] 
K3[0,0]=f3[0]; K3[1,1]=f3[1];K3[0,2]=c3[0]; K3[1,2]=c3[1]; K3[2,2]=1; 

K4=np.empty((3,3))
f4=[1145.5113525390625, 1144.77392578125] 
c4=[514.9682006835938, 501.88201904296875]
K4[0,0]=f4[0]; K4[1,1]=f4[1];K4[0,2]=c4[0]; K4[1,2]=c4[1]; K4[2,2]=1;  

K=np.zeros((4,3,3));K[0,:,:]=K1;K[1,:,:]=K2;K[2,:,:]=K3;K[3,:,:]=K4
K=torch.from_numpy(K.astype('float32'))


cam_names = ['54138969', '55011271', '58860488', '60457274']
all_cams = ['cam0', 'cam1', 'cam2', 'cam3']
all_cams_3d = ['cam0_3d', 'cam1_3d', 'cam2_3d', 'cam3_3d']
nmpjpe_tot_min=200
if config.NoEval:
    for epoch in range(config.N_epochs):

        for i, sample in enumerate(train_loader):

            # not the most elegant way to extract the dictionary
            poses_2d = {key:sample[key] for key in all_cams}
            poses_3d = {key:sample[key] for key in all_cams_3d}

            inp_poses = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), num_joints*2)).cuda()
            out_poses_triang = torch.zeros((poses_3d['cam0_3d'].shape[0] * len(all_cams), num_joints,3)).cuda()
            inp_confidences = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), num_joints)).cuda()

            cnt = 0
            for b in range(poses_2d['cam0'].shape[0]):
                for c_idx, cam in enumerate(poses_2d):
                    inp_poses[cnt] = poses_2d[cam][b]
                    out_poses_triang[cnt] = poses_3d[cam+'_3d'][b]
                    inp_confidences[cnt] = sample['confidences'][c_idx][b]
                    cnt += 1


            pred = model(inp_poses, inp_confidences)
            pred_poses = pred[0]
            pred_cam_angles = pred[1]

            pred_rot=transform.euler_angles_to_matrix(pred_cam_angles,convention=['X','Y','Z'])
            
            # reproject to original cameras after applying rotation to the canonical poses
            rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, num_joints))
            
            # triangulation loss
            losses.triang=mpjpe(rot_poses.transpose(2,1),out_poses_triang)

            rot_poses=rot_poses.reshape(-1, num_joints*3)
            # reprojection loss
            losses.rep = loss_weighted_rep_no_scale(inp_poses, rot_poses, inp_confidences)

            # view-consistency and camera-consistency
            # to compute the different losses we need to do some reshaping
            pred_poses_rs = pred_poses.reshape((-1, len(all_cams), num_joints*3))
            pred_rot_rs = pred_rot.reshape(-1, len(all_cams), 3, 3)
            # pred_trans_rs = pred_cam_t.reshape(-1, len(all_cams), 3)
            confidences_rs = inp_confidences.reshape(-1, len(all_cams), num_joints)
            inp_poses_rs = inp_poses.reshape(-1, len(all_cams), num_joints*2)
            rot_poses_rs = rot_poses.reshape(-1, len(all_cams), num_joints*3)
            inp_poses_rs2 = inp_poses_rs.reshape(-1, len(all_cams), 2,num_joints)
            inp_poses_rs2 = inp_poses_rs2.transpose(-2,-1)
            
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
            
            losses.loss =  config.weight_rep * losses.rep + \
                           config.weight_view * losses.view + \
                           config.weight_camera * losses.camera +\
                            losses.triang *10
                          

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



        scheduler.step()
        with torch.no_grad():
            model_eval.load_state_dict(model.state_dict())
            model_eval.eval()
            N=0
            nmpjpe_tot=0
            mpjpe_tot=0
            pmpjpe_tot=0
            
            for i, sample in enumerate(test_loader):
                # input_2d=model_skel_morph(sample['poses_2d'].cuda())

                pred=model_eval(sample['poses_2d'].cuda(),sample['confidences'].cuda())
                pred_poses = pred[0]
                pred_cam_angles = pred[1]

                # angles are in axis angle notation
                # use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
                # pred_rot = rodrigues(pred_cam_angles)
                pred_rot = transform.euler_angles_to_matrix(pred_cam_angles,convention=['X','Y','Z'])

                # reproject to original cameras after applying rotation to the canonical poses
                rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, num_joints))
                rot_poses=torch.transpose(rot_poses,2,1)
                nmpjpe_batch=n_mpjpe(rot_poses,sample['poses_3d'])
                mpjpe_batch=mpjpe(rot_poses,sample['poses_3d'])
                pmpjpe_batch=p_mpjpe(rot_poses.cpu().numpy(),sample['poses_3d'].cpu().numpy())
                A=sample['poses_3d'][100:101].cpu().numpy()
                B=pred_poses.reshape(-1, 3, num_joints).transpose(2,1)[100:101].cpu().numpy()
                # plot_15j_3d(np.concatenate((A,B),axis=0))
                nmpjpe_tot+=nmpjpe_batch*sample['poses_3d'].shape[0]
                mpjpe_tot+=mpjpe_batch*sample['poses_3d'].shape[0]
                pmpjpe_tot+=pmpjpe_batch*sample['poses_3d'].shape[0]
                N+=sample['poses_3d'].shape[0]
            print('N-MPJPE',mpjpe_tot/N*1000,'MPJPE',nmpjpe_tot/N*1000,'P-MPJPE',pmpjpe_tot/N*1000)
            # plot_15j_3d(np)
            if nmpjpe_tot/N*1000<nmpjpe_tot_min:
                # save the new trained model every epoch
                print('Saving...')
                torch.save(model, 'models/model_lifter.pt')
                nmpjpe_tot_min=nmpjpe_tot
    print('done')

else:
    with torch.no_grad():
        checkpoint=torch.load('models/model_lifter.pt')
        model.load_state_dict(checkpoint.state_dict())
        model.eval()
        N=0
        loss_3d_tot=0
        for i, sample in enumerate(test_loader):
            # input_2d=model_skel_morph(sample['poses_2d'].cuda())
            pred=model(sample['poses_2d'].cuda(),sample['confidences'].cuda())
            pred_poses = pred[0]
            pred_cam_angles = pred[1]

            # angles are in axis angle notation
            # use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
            # pred_rot = rodrigues(pred_cam_angles)
            pred_rot=transform.euler_angles_to_matrix(pred_cam_angles,convention=['X','Y','Z'])
            # reproject to original cameras after applying rotation to the canonical poses
            rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, num_joints))
            rot_poses=torch.transpose(rot_poses,2,1)
            loss_3d=n_mpjpe(rot_poses,sample['poses_3d'])
            A=sample['poses_3d'][100:101].cpu().numpy()
            B=rot_poses[100:101].cpu().numpy()
            plot_15j_3d(np.concatenate((A,B),axis=0))
            loss_3d_tot+=loss_3d*sample['poses_3d'].shape[0]
            N+=sample['poses_3d'].shape[0]
        print('Error 3D',loss_3d_tot/N*1000)
