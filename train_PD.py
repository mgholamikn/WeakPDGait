from matplotlib import animation, colors
import numpy as np
from scipy.signal.wavelets import cascade
from utils.plot import *
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from mpl_toolkits import mplot3d
# import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from sklearn.cluster import KMeans
from sklearn import svm

# subjects=['S01','S02','S03','S04','S06','S07','S08','S09',
# 'S10','S11','S12','S13','S14','S16','S17','S18','S19','S20',
# 'S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31',
# 'S32','S33','S34','S35']

subjects_PD=['S01','S02','S03','S04','S05','S06','S07','S09', #22,23,,25,1,
'S10','S11','S12','S13','S14','S16','S17','S18','S19',
'S21','S22','S23','S24','S28','S29','S30','S31',
'S32','S33','S34','S35']
subjects_All=['S01','S02','S03','S04','S05','S06','S07','S08','S09',
'S10','S11','S12','S13','S14','S16','S17','S18','S19','S20',
'S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31',
'S32','S33','S34','S35']
subjects_All_date=['20210223','20191114','20191120','20191112','20191119','20200220','20191121',
'20191126','20191128','20191203','20191204','20200108','20200109','20200121','20200122','20200123',
'20200124','20200127','20200130','20200205','20200206_9339','20200207','20200213','20200214','20200218',
'26','20200221','20210706','20210804','20200206_9629','20210811','20191210','20191212','20191218','20200227']
healthy_controls=['S08','S20','S27','S25','S26']
# subjects=['S08','S20','S']


plt.rcParams['font.size'] = 10
plt.rc('axes', labelsize=15) 
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15) 
fig,ax = plt.subplots(3,3,figsize = (12,7))
# clrs = sns.color_palette('husl', n_colors=len(subjects))  # a list of RGB tuples
# fig.add_subplot(211)
# NUM_COLORS = len(subjects)
# cm = plt.get_cmap('gist_rainbow')
# ax.set_prop_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
num=0
data_train=np.zeros((len(subjects_All),2))
output={}
output_timeseries={}
output_csv=np.zeros((35,16))
output_csv_header=['Subject Name','Step Width','Right Step Length', 'Left Step Length', 'Cadence','Right Foot Clearance', 'Left Foot Clearance', 'Right Arm Swing', 'Left Arm Swing', 'Right Hip Flexion', 'Left Hip Flexion', 'Right Knee Flexion','Left Knee Flexion','Right Trunk rotation (calculated by right side key points)','Left Trunk rotation (calculated by left side key points)','Arm swing symmetry']
subj_pd=[]
num_sample=0
for idx,(subj,subj_date) in enumerate(zip(subjects_All,subjects_All_date)):
    output_csv[idx,0]=subj_date
    output[subj]=np.zeros(30)
    output_timeseries[subj]={}
    data_normal=np.load('outputs_finetuned/Predictions_'+subj+'.npy')
    for feature in range(30):
        output_timeseries[subj][feature]=[]
    
    for ii in range(15):
        for jj in range(3):
            data_normal[:,ii,jj]=savgol_filter(data_normal[:,ii,jj],11,3)
    rows=np.concatenate((np.arange(0,600),np.arange(1000,2600),np.arange(2800,3900),np.arange(4000,4200)))
    x_vec=data_normal[:,1]-data_normal[:,4]
    y_vec=data_normal[:,7]-data_normal[:,0]
    x_vec/=np.linalg.norm(x_vec,keepdims=True,axis=-1)
    y_vec/=np.linalg.norm(y_vec,keepdims=True,axis=-1)
    z_vec=np.cross(x_vec,y_vec)
    rotation_matrix=np.ones((len(x_vec),3,3))
    rotation_matrix[:,:,0]=x_vec
    rotation_matrix[:,:,1]=y_vec
    rotation_matrix[:,:,2]=z_vec
    data_normal=np.matmul(data_normal,rotation_matrix)
    # fig1,ax1 = plt.subplots(1,1)
    # ax1.plot(data_normal[0:300,14,0],'deepskyblue',linewidth=3)
    # ax1.plot(data_normal[0:300,14,1],'yellowgreen',linewidth=3)
    # ax1.plot(data_normal[0:300,14,2],'crimson',linewidth=3)
    # # ax1[1].plot(data_normal[:,6,:])
    # plt.legend(['x','y','z'])
    # plt.show()
    num_sample+=len(data_normal)
    # plot15j_PD(data_normal[:300],show_animation=False)
    # print('Len data:',len(data_normal))
    ############# Step Width #################
    ##########################################
    step_width=np.abs(data_normal[:,3,0]-data_normal[:,6,0])
    # plt.plot(step_width)
    # plt.show()
    bone_length=np.linalg.norm((data_normal[:,1]-data_normal[:,4]),axis=-1)
    # plt.plot(bone_length)
    # plt.show()
    step_width=step_width/bone_length

    A=[]
    for ii in range(30,len(step_width)):
        A.append(np.mean(step_width[ii-30:ii]))
        output_timeseries[subj][0].append(np.mean(step_width[ii-30:ii]))
    step_width=np.asarray(A)

    # output[subj][0]=np.mean(step_width)
    output[subj][0]=np.mean(step_width)
    output_csv[idx,1]=np.mean(step_width)
    # output[subj][15]=np.std(step_width)

    print(subj,'step_width',np.mean(step_width))    
    ############# Step Length R ##############
    ########################################
    step_length=np.linalg.norm((data_normal[:,3]-data_normal[:,6]),axis=-1)
    row_r=(data_normal[:,6,2]-data_normal[:,3,2]>0)
    row_l=(data_normal[:,3,2]-data_normal[:,6,2]>0)
    step_length_r=step_length[row_r]
    step_length_l=step_length[row_l]
    # plt.plot(step_length)
    # plt.plot(step_length[row_l])
    # plt.show()
    bone_length_r=np.linalg.norm((data_normal[:,5]-data_normal[:,6]),axis=-1)[row_r]
    bone_length_l=np.linalg.norm((data_normal[:,3]-data_normal[:,2]),axis=-1)[row_l]
    bone_length=np.linalg.norm((data_normal[:,5]-data_normal[:,6]),axis=-1)
    step_length=step_length /bone_length
    step_length_r=step_length_r/bone_length_r
    step_length_l=step_length_l /bone_length_l
    # peaks, _ = find_peaks(step_length_n, height=0,distance=0.5*15)
    # time=np.arange(len(step_length_n))/15

    A=[]
    for ii in range(30,len(step_length_r)):
        A.append(np.max(step_length_r[ii-30:ii]))
        output_timeseries[subj][1].append(np.mean(step_length_r[ii-30:ii]))
    step_length_r=np.asarray(A)
    B=[]
    for ii in range(30,len(step_length_l)):
        B.append(np.max(step_length_l[ii-30:ii]))
        output_timeseries[subj][2].append(np.mean(step_length_l[ii-30:ii]))
    step_length_l=np.asarray(B)
    C=[]
    for ii in range(30,len(step_length)):
        C.append(np.max(step_length[ii-30:ii]))
    step_length=np.asarray(C)
    output[subj][1]=np.mean(step_length_r)
    output[subj][2]=np.mean(step_length_l)
    output_csv[idx,2]=np.mean(step_length_r)
    output_csv[idx,3]=np.mean(step_length_l)
    # output[subj][16]=np.std(step_length_r)
    # output[subj][17]=np.std(step_length_l)
    print(subj,'step_length_r',np.mean(step_length_r))
    print(subj,'step_length_l',np.mean(step_length_l))

    ############# Cadence ##############
    ########################################
    toe_traj=np.abs(data_normal[:,6,2]-data_normal[:,3,2])
    peaks, _ = find_peaks(toe_traj,distance=5,height=-0.2)
    ave=np.mean(toe_traj[peaks])-0.3
    peaks, _ = find_peaks(toe_traj,distance=5,height=ave)
    # plt.plot(toe_traj)
    # plt.plot(peaks,toe_traj[peaks],'*')
    # plt.show()
    # plt.plot(toe_traj)
    # plt.plot(data_normal[:,6,2])
    # plt.plot(data_normal[:,3,2])
    # plt.show()
    if subj in ['S01','S28','S29','S31']:
        cadence=60/((peaks[1:]-peaks[:-1])/30)
        gait_speed=toe_traj[peaks[1:]]*cadence
    else:
        cadence=60/((peaks[1:]-peaks[:-1])/15)
        gait_speed=toe_traj[peaks[1:]]*cadence
    
    # cadence=np.mean(cadence)
    output_timeseries[subj][3]=cadence
    output[subj][3]=np.mean(cadence)
    output_csv[idx,4]=np.mean(cadence)
    # output[subj][18]=np.std(cadence)
    output[subj][15]=np.mean(gait_speed)
    output[subj][16]=np.std(gait_speed)
    output_timeseries[subj][15].append(gait_speed)
   
    print(subj,'cadence',np.mean(cadence))


    ############# Foot Lifting R##############
    ########################################
    foot_height_r=data_normal[:,6,1]-data_normal[:,4,1]
    foot_height_l=data_normal[:,3,1]-data_normal[:,1,1]
    bone_length_l=np.linalg.norm((data_normal[:,1]-data_normal[:,2]),axis=-1)
    bone_length_r=np.linalg.norm((data_normal[:,5]-data_normal[:,4]),axis=-1)
    foot_height_r=foot_height_r/bone_length_r
    foot_height_l=foot_height_l/bone_length_l

    A=[]
    B=[]
    for ii in range(30,len(foot_height_r)):
        A.append(np.max(foot_height_r[ii-30:ii])-np.min(foot_height_r[ii-30:ii]))
        B.append(np.max(foot_height_l[ii-30:ii])-np.min(foot_height_l[ii-30:ii]))
        output_timeseries[subj][4].append(np.max(foot_height_r[ii-30:ii])-np.min(foot_height_r[ii-30:ii]))
        output_timeseries[subj][5].append(np.max(foot_height_l[ii-30:ii])-np.min(foot_height_l[ii-30:ii]))
    foot_height_n_r=np.asarray(A)
    foot_height_n_l=np.asarray(B)

    # A=[]
    # B=[]
    # for ii in range(90,len(foot_height_r)):
    #     A.append(np.max(foot_height_r[ii-30:ii])-np.min(foot_height_r[ii-30:ii]))
    #     B.append(np.max(foot_height_l[ii-30:ii])-np.min(foot_height_l[ii-30:ii]))

    # foot_height_s_r=np.asarray(A)
    # foot_height_s_l=np.asarray(B)
    output[subj][4]=np.mean(foot_height_n_r)
    output[subj][5]=np.mean(foot_height_n_l)
    output_csv[idx,5]=np.mean(foot_height_n_r)
    output_csv[idx,6]=np.mean(foot_height_n_l)
    # output[subj][19]=np.std(foot_height_n_r)
    # output[subj][20]=np.std(foot_height_n_l)

    print(subj,'foot_height_n_r',np.mean(foot_height_n_r))
    print(subj,'foot_height_n_l',np.mean(foot_height_n_l))
    # print('foot_height_n_r',np.mean(foot_height_n_r))
    

    ############# Hand Movement R ##############
    ########################################
    ave=np.mean(data_normal[:,14,1],axis=0,keepdims=True)
    dist=np.linalg.norm((data_normal[:,14]-data_normal[:,4]),axis=-1)
    # dist=data_normal[:,15,2]-data_normal[:,4,2]
    bone_length=np.linalg.norm((data_normal[:,4]-data_normal[:,1]),axis=-1)
    dist=dist/bone_length

    A=[]
    for ii in range(30,len(dist)):
        A.append(np.max(dist[ii-30:ii])-np.min(dist[ii-30:ii]))
        output_timeseries[subj][6].append(np.max(dist[ii-30:ii])-np.min(dist[ii-30:ii]))
    hand_mov_n_r=np.asarray(A)
    output[subj][6]=np.mean(hand_mov_n_r)
    output_csv[idx,7]=np.mean(hand_mov_n_r)
    # output[subj][21]=np.std(hand_mov_n_r)
    print(subj,'hand_mov_n_r',np.mean(hand_mov_n_r))
    # fig = plt.figure(figsize = (10, 7))
    # ax3= plt.axes()
    # ax3.plot(data_normal[:,15,:])
    # plt.show()
    ############# Hand Movement L ##############
    ########################################
    ave=np.mean(data_normal[:,11,:],axis=0,keepdims=True)
    dist=np.linalg.norm((data_normal[:,11]-data_normal[:,1]),axis=-1)
    # dist=data_normal[:,12,2]-data_normal[:,1,2]
    bone_length=np.linalg.norm((data_normal[:,4]-data_normal[:,1]),axis=-1)
    dist=dist/bone_length

    A=[]
    for ii in range(30,len(dist)):
        A.append(np.max(dist[ii-30:ii])-np.min(dist[ii-30:ii]))
        output_timeseries[subj][7].append(np.max(dist[ii-30:ii])-np.min(dist[ii-30:ii]))
        
    hand_mov_n_l=np.asarray(A)
    output_timeseries[subj][14]=hand_mov_n_l/hand_mov_n_r
    output[subj][7]=np.mean(hand_mov_n_l)
    output_csv[idx,8]=np.mean(hand_mov_n_l)
    output[subj][14]=np.mean(hand_mov_n_l)/np.mean(hand_mov_n_r)
    output_csv[idx,15]=np.mean(hand_mov_n_l)/np.mean(hand_mov_n_r)
    # output[subj][22]=np.std(hand_mov_n_l)
    # output[subj][23]=np.std(hand_mov_n_l/hand_mov_n_r)
    if np.mean(hand_mov_n_r)<0.189 or np.mean(hand_mov_n_l)<0.2:
        subj_pd.append(subj)
    print(subj,'hand_mov_n_l',np.mean(hand_mov_n_l))

    ## data for clustering
    data_train[num,0]=np.mean(hand_mov_n_r[:])
    data_train[num,1]=np.mean(hand_mov_n_l[:])
    
    # data_train[num,2]=np.mean(hand_mov_n_l[:])

    ############# Hip Flexion ##############
    ########################################
    dist_l=data_normal[:,1,2]-data_normal[:,2,2]
    bone_l=np.linalg.norm((data_normal[:,1]-data_normal[:,2]),axis=-1)
    dist_r=data_normal[:,4,2]-data_normal[:,5,2]
    bone_r=np.linalg.norm((data_normal[:,4]-data_normal[:,5]),axis=-1)
    # bone_length=np.linalg.norm((data_normal[:,4]-data_normal[:,1]),axis=-1)
    # dist=data_normal[:,12,2]-data_normal[:,1,2]
    # hip_flex=angle_between(thigh_vec,torso_vec)
    hip_flex_r=dist_r/bone_r
    hip_flex_l=dist_l/bone_l
    A=[]
    B=[]
    for ii in range(30,len(hip_flex_r)):
        A.append(np.max(hip_flex_r[ii-30:ii])-np.min(hip_flex_r[ii-30:ii]))
        B.append(np.max(hip_flex_l[ii-30:ii])-np.min(hip_flex_l[ii-30:ii]))
        output_timeseries[subj][8].append(np.max(hip_flex_r[ii-30:ii])-np.min(hip_flex_r[ii-30:ii])) 
        output_timeseries[subj][9].append(np.max(hip_flex_l[ii-30:ii])-np.min(hip_flex_l[ii-30:ii]))     
    hip_flex_r=np.asarray(A)
    hip_flex_l=np.asarray(B)
    output[subj][8]=np.mean(hip_flex_r)
    output[subj][9]=np.mean(hip_flex_l)
    output_csv[idx,9]=np.mean(hip_flex_r)
    output_csv[idx,10]=np.mean(hip_flex_l)
    # output[subj][24]=np.std(hip_flex_r)
    # output[subj][25]=np.std(hip_flex_l)
    print(subj,'hip_flex_l',np.mean(hip_flex_l))
    print(subj,'hip_flex_r',np.mean(hip_flex_r))
    # plt.plot(hip_flex_r)
    # plt.show() 

    ############# Knee Flexion ##############
    ########################################
    thigh_l=np.linalg.norm((data_normal[:,1]-data_normal[:,2]),axis=-1)
    shin_l=np.linalg.norm((data_normal[:,3]-data_normal[:,2]),axis=-1)
    leg_l=np.linalg.norm((data_normal[:,1]-data_normal[:,3]),axis=-1)
    thigh_r=np.linalg.norm((data_normal[:,5]-data_normal[:,4]),axis=-1)
    shin_r=np.linalg.norm((data_normal[:,5]-data_normal[:,6]),axis=-1)
    leg_r=np.linalg.norm((data_normal[:,4]-data_normal[:,6]),axis=-1)
    # bone_length=np.linalg.norm((data_normal[:,4]-data_normal[:,1]),axis=-1)
    # dist=data_normal[:,12,2]-data_normal[:,1,2]
    # hip_flex=angle_between(thigh_vec,torso_vec)
    knee_flex_r=leg_r**2/(thigh_r*shin_r)-thigh_r/shin_r-shin_r/thigh_r
    knee_flex_l=leg_l**2/(thigh_l*shin_l)-thigh_l/shin_l-shin_l/thigh_l
    A=[]
    B=[]
    # plt.plot(knee_flex_r/2)
    # plt.show()
    for ii in range(30,len(knee_flex_r)):
        A.append(np.max(knee_flex_r[ii-30:ii])-np.min(knee_flex_r[ii-30:ii]))
        B.append(np.max(knee_flex_l[ii-30:ii])-np.min(knee_flex_l[ii-30:ii]))
        output_timeseries[subj][10].append(np.max(knee_flex_r[ii-30:ii])-np.min(knee_flex_r[ii-30:ii])) 
        output_timeseries[subj][11].append(np.max(knee_flex_l[ii-30:ii])-np.min(knee_flex_l[ii-30:ii]))    
    knee_flex_r=np.asarray(A)
    knee_flex_l=np.asarray(B)
    output[subj][10]=np.mean(knee_flex_l)
    output[subj][11]=np.mean(knee_flex_r)
    output_csv[idx,11]=np.mean(knee_flex_l)
    output_csv[idx,12]=np.mean(knee_flex_r)
    # output[subj][26]=np.std(knee_flex_l)
    # output[subj][27]=np.std(knee_flex_r)
    print(subj,'knee_flex_l',np.mean(knee_flex_l))
    print(subj,'knee_flex_r',np.mean(knee_flex_r))

    ############# Trunk Rotation  ##############
    ########################################
    data_normal-=data_normal[:,:1]
    shoulder_l=np.linalg.norm((data_normal[:,9,[0,2]]-data_normal[:,0,[0,2]]),axis=-1)
    hip_l=np.linalg.norm((data_normal[:,4,[0,2]]-data_normal[:,0,[0,2]]),axis=-1)
    hip2shoulder_l=np.linalg.norm((data_normal[:,4,[0,2]]-data_normal[:,9,[0,2]]),axis=-1)
    shoulder_r=np.linalg.norm((data_normal[:,12,[0,2]]-data_normal[:,0,[0,2]]),axis=-1)
    hip_r=np.linalg.norm((data_normal[:,1,[0,2]]-data_normal[:,0,[0,2]]),axis=-1)
    hip2shoulder_r=np.linalg.norm((data_normal[:,1,[0,2]]-data_normal[:,12,[0,2]]),axis=-1)
    
    # bone_length=np.linalg.norm((data_normal[:,4]-data_normal[:,1]),axis=-1)
    # dist=data_normal[:,12,2]-data_normal[:,1,2]
    # hip_flex=angle_between(thigh_vec,torso_vec)
    trunk_rot_r=hip2shoulder_r**2/(hip_r*shoulder_r)-hip_r/shoulder_r-shoulder_r/hip_r
    trunk_rot_l=hip2shoulder_l**2/(hip_l*shoulder_l)-hip_l/shoulder_l-shoulder_l/hip_l
    A=[]
    B=[]
    # plt.plot(data_normal[:,2,0])
    # plt.show()
    # plt.plot(trunk_rot_l/2)
    # plt.show()
    # plt.plot(data_normal[:,4])
    # plt.show()
    # fig1,ax1 = plt.subplots(3,1,figsize = (10,7))
    # ax1[0].plot(data_normal[:,3,0])
    # ax1[1].plot(data_normal[:,3,1])
    # ax1[2].plot(data_normal[:,3,2])
    # plt.show()
    for ii in range(30,len(trunk_rot_r)):
        A.append(np.max(trunk_rot_r[ii-30:ii])-np.min(trunk_rot_r[ii-30:ii]))
        B.append(np.max(trunk_rot_l[ii-30:ii])-np.min(trunk_rot_l[ii-30:ii]))
        output_timeseries[subj][12].append(np.max(trunk_rot_r[ii-30:ii])-np.min(trunk_rot_r[ii-30:ii])) 
        output_timeseries[subj][13].append(np.max(trunk_rot_l[ii-30:ii])-np.min(trunk_rot_l[ii-30:ii]))   
    trunk_rot_r=np.asarray(A)
    trunk_rot_l=np.asarray(B)
    output[subj][12]=np.mean(trunk_rot_r)
    output[subj][13]=np.mean(trunk_rot_l)
    output_csv[idx,13]=np.mean(trunk_rot_r)
    output_csv[idx,14]=np.mean(trunk_rot_l)
    # output[subj][28]=np.std(trunk_rot_r)
    # output[subj][29]=np.std(trunk_rot_l)
    print(subj,'trunk_rot_l',np.mean(trunk_rot_l))
    print(subj,'trunk_rot_r',np.mean(trunk_rot_r))
 

    # ############# Gait Speed and variability  ##############
    # ########################################################
    # gait_speed=step_length/cadence
    # for ii in range(30,len(gait_speed)):
    #     A.append(np.max(gait_speed[ii-30:ii])-np.min(gait_speed[ii-30:ii]))
    #     B.append(np.max(gait_speed[ii-30:ii])-np.min(gait_speed[ii-30:ii]))
    # gait_speed=np.asarray(A)
    # gait_speed=np.asarray(B)
    # output[subj][15]=np.mean(gait_speed)
    # output[subj][16]=np.std(gait_speed)

    # print(subj,'gait speed',np.mean(trunk_rot_l))
    # print(subj,'gait speed variability',np.mean(trunk_rot_r))
 
    #############################
    ###### Plot #################
    
    # 
    if subj in ['S08','S20','S25','S26','S27']:
        marker='*'
    else:
        marker='o'
    c1=np.random.rand(1)[0]
    c2=np.random.rand(1)[0]
    
    line=ax[0,0].scatter(np.mean(hand_mov_n_r[:]),np.mean(hand_mov_n_l[:]),color=colors.to_rgb((c1,c2,num/len(subjects_All))),marker=marker,s=160) #:3900 
    line=ax[0,1].scatter(np.mean(step_length_r[:]),np.mean(step_length_l[:]),color=colors.to_rgb((c1,c2,num/len(subjects_All))),marker=marker,s=160) #:3900 
    line=ax[0,2].scatter(np.mean(foot_height_n_r[:]),np.mean(foot_height_n_l[:]),color=colors.to_rgb((c1,c2,num/len(subjects_All))),marker=marker,s=160)
    line=ax[1,0].scatter(np.mean(hip_flex_r[:]),np.mean(hip_flex_l[:]),color=colors.to_rgb((c1,c2,num/len(subjects_All))),marker=marker,s=160) #:3900 
    line=ax[1,1].scatter(np.mean(knee_flex_r[:]),np.mean(knee_flex_l[:]),color=colors.to_rgb((c1,c2,num/len(subjects_All))),marker=marker,s=160) #:3900 
    line=ax[1,2].scatter(np.mean(trunk_rot_r[:]),np.mean(trunk_rot_l[:]),color=colors.to_rgb((c1,c2,num/len(subjects_All))),marker=marker,s=160) #:3900 
    line=ax[2,0].scatter(np.mean(step_width[:]),np.mean(cadence[:]),color=colors.to_rgb((c1,c2,num/len(subjects_All))),marker=marker,s=160) #:3900 
    line=ax[2,1].scatter(np.mean(hand_mov_n_r[:]),np.mean(hand_mov_n_l[:]),color=colors.to_rgb((c1,c2,num/len(subjects_All))),marker=marker,s=160) #:3900
    line=ax[2,2].scatter(np.mean(gait_speed[:]),np.std(gait_speed[:]),color=colors.to_rgb((c1,c2,num/len(subjects_All))),marker=marker,s=160) #:3900

    num+=1
print('Average number of samples:',num_sample/num)
#save
ax[0,0].set_xlabel('Right Hand Range of Motion')
ax[0,0].set_ylabel('Left Hand Range of Motion')
# ax[0,0].plot([0, 0.55], [0, 0.55], ls="--", c=".3")
# while n<len(subjects)/2:
#     delta_x+0.05

#     ax[0,0].plot([0, 0.55], [0, 0.55], ls="--", c=".3")
# ax1.set_zlabel('Left Hand Range of Motion')
# plt.show() 

# plt.legend(subjects,loc="best")
ax[0,1].set_xlabel('Right Step Length')
ax[0,1].set_ylabel('Left Step Length')
# ax2.set_zlabel('Left Hand Range of Motion')

ax[0,2].set_xlabel('Right Foot Lifting')
ax[0,2].set_ylabel('Left Foot Lifting')


ax[1,0].set_xlabel('Right Hip Flexion')
ax[1,0].set_ylabel('Left Hip Flextion')

ax[1,1].set_xlabel('Right Knee Flexion')
ax[1,1].set_ylabel('Left Knee Flextion')

ax[1,2].set_xlabel('Right Trunk Rot')
ax[1,2].set_ylabel('Left Trunk Rot')

ax[2,0].set_xlabel('Step Width')
ax[2,0].set_ylabel('Cadence')

ax[2,1].set_xlabel('Right Hand Range of Motion')
ax[2,1].set_ylabel('Left Hand Range of Motion')

ax[2,2].set_xlabel('Gait Speed')
ax[2,2].set_ylabel('Gait Speed Variation')

fig.legend(subjects_All,loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.95)   
# ax[1,2].legend(subjects,loc="center left",bbox_to_anchor=(1.04,1),ncol=2)
plt.show() 

for subj in subjects_All:
    for ii in range(15):
        output_timeseries[subj][ii]=np.asarray(output_timeseries[subj][ii])
np.savez('output.npz',data=output)
np.savez('output_timeseries.npz',data=output_timeseries)

import csv
with open('output.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(output_csv_header)

    # write multiple rows
    writer.writerows(output_csv)

#load data
data=np.load('output_timeseries.npz',allow_pickle=True)
data=data['data'].item()

for subj in subjects_All:
    for ii in range(15):
        print(subj,',feature number', ii,':',data[subj][ii].shape)

data=np.load('output.npz',allow_pickle=True)
data=data['data'].item()
threshold_arm_sym_min=1
threshold_arm_sym_max=1
threshold_StepL_r=10
threshold_StepL_l=10
threshold_footlift_r=10
threshold_footlift_l=10
threshold_Hip_r=10
threshold_Hip_l=10
threshold_knee_r=10
threshold_knee_l=10
threshold_trunk_r=10
threshold_trunk_l=10
threshold_stepwidth=0
threshold_cadence=0
threshold_handROM_r=10
threshold_handROM_l=10
threshold_gait_speed=10
threshold_gait_speed_var=0
## setting thresholod

for subj in healthy_controls:
    if data[subj][14]<threshold_arm_sym_min:
        threshold_arm_sym_min=data[subj][14]
    if data[subj][14]>threshold_arm_sym_max:
        threshold_arm_sym_max=data[subj][14]
    if data[subj][1]<threshold_StepL_r:
        threshold_StepL_r=data[subj][1]
    if data[subj][2]<threshold_StepL_l:
        threshold_StepL_l=data[subj][2]
    if data[subj][4]<threshold_footlift_r:
        threshold_footlift_r=data[subj][4]
    if data[subj][5]<threshold_footlift_l:
        threshold_footlift_l=data[subj][5]
    if data[subj][8]<threshold_Hip_r:
        threshold_Hip_r=data[subj][8]
    if data[subj][9]<threshold_Hip_l:
        threshold_Hip_l=data[subj][9]
    if data[subj][10]<threshold_knee_r:
        threshold_knee_r=data[subj][10]
    if data[subj][11]<threshold_knee_l:
        threshold_knee_l=data[subj][11]
    if data[subj][12]<threshold_trunk_r:
        threshold_trunk_r=data[subj][12]
    if data[subj][13]<threshold_trunk_l:
        threshold_trunk_l=data[subj][13]
    if data[subj][0]>threshold_stepwidth:
        threshold_stepwidth=data[subj][0]
    if data[subj][3]>threshold_cadence:
        threshold_cadence=data[subj][3]
    if data[subj][6]<threshold_handROM_r:
        threshold_handROM_r=data[subj][6]
    if data[subj][7]<threshold_handROM_l:
        threshold_handROM_l=data[subj][7]
    if data[subj][15]<threshold_gait_speed:
        threshold_gait_speed=data[subj][15]
    if data[subj][16]>threshold_gait_speed_var:
        threshold_gait_speed_var=data[subj][16]
ii=0
data_=np.zeros((len(data.keys()),30))
for key in data.keys():
    data_[ii]=data[key]
    ii+=1
ii=0
label_matrix=np.zeros((len(subjects_PD),8))
for subj in subjects_PD:
    if data[subj][14]<threshold_arm_sym_min:
        label_matrix[ii,0]=1
    if data[subj][14]>threshold_arm_sym_max:
        label_matrix[ii,0]=1
    if data[subj][1]<threshold_StepL_r:
        label_matrix[ii,1]=1
    if data[subj][2]<threshold_StepL_l:
        label_matrix[ii,1]=1
    if data[subj][4]<threshold_footlift_r:
        label_matrix[ii,2]=1
    if data[subj][5]<threshold_footlift_l:
        label_matrix[ii,2]=1
    if data[subj][8]<threshold_Hip_r:
        label_matrix[ii,3]=1
    if data[subj][9]<threshold_Hip_l:
        label_matrix[ii,3]=1
    if data[subj][10]<threshold_knee_r:
        label_matrix[ii,4]=1
    if data[subj][11]<threshold_knee_l:
        label_matrix[ii,4]=1
    if data[subj][12]<threshold_trunk_r:
        label_matrix[ii,5]=1
    if data[subj][13]<threshold_trunk_l:
        label_matrix[ii,5]=1
    if data[subj][0]>threshold_stepwidth:
        label_matrix[ii,6]=1
    if data[subj][3]>threshold_cadence:
        label_matrix[ii,7]=1
    if data[subj][6]<threshold_handROM_r:
        label_matrix[ii,0]=1
    if data[subj][7]<threshold_handROM_l:
        label_matrix[ii,0]=1
    if data[subj][15]<threshold_gait_speed:
        label_matrix[ii,8]=1
    # if data[subj][16]>threshold_gait_speed_var:
    #     label_matrix[ii,8]=1
    ii+=1        
from snorkel.labeling.model import LabelModel

print(label_matrix)
print(np.sum(label_matrix,axis=1))
Y_true=np.asarray([1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,0,1])
Y_true=np.asarray(   [1,1,0,1,1,0,0,0,1,0   ,1,0,1,1,1,1,0,0,0,1,1,0,0,1,1,0,1,0,1])
Y_pred_Lu=np.asarray([0.95,0.90,0.15,0.14,0.93,0.11,0.16,0.89,0.97,0.57,0.8,0.10,0.70,0.8,0.90,0.79,0.16,0.75,0.18,0.69,0.95,0.1,0.96,0.86,0.96,0.61,0.62,0.55,0.85])
Y_pred=np.zeros(len(subjects_PD))
jj=0
data_X=np.zeros((len(subjects_PD),30))

for subj in subjects_PD:
    data_X[jj]=data[subj]
    jj+=1
accuracy=0
count=0
y_pred_tot=np.zeros((len(subjects_PD)))
  

# label_matrix_new=label_matrix[:,rand] 

for ii in range(len(subjects_PD)):
    # if ii in [8,19,24,25,26]:
    #     continue

    label_model = LabelModel(verbose=False)
    label_matrix_temp=np.delete(label_matrix,ii,0)
    label_model.fit(label_matrix_temp, seed=123)
    weights=label_model.get_weights()
    weights/=np.sum(weights)

    # print(weights)
    # print(np.sum(weights*label_matrix,axis=1))

    # weights=np.ones(8)/8
    data_Y=np.sum(weights*label_matrix,axis=1)
    # print(data_Y)

    data_Y[data_Y<=(2/8)]=0.0
    data_Y[data_Y>(2/8)]=1.0
    # print(data_Y)

    
    # data_Y_test= np.asarray([0.69177148 ,0.38927647, 0.,         0.,         0.  ,       0.10104691,
    #  0.    ,     0.      ,   0.14054997, 0.40401168, 0.24159688, 0.2068075 ,
    #  0.71717177, 0.28282823, 0.51294499, 0.71975249, 0.20998022, 0.1089333,
    #  0.22213471, 0.10576059, 0.4929041 , 0.81115197, 0.   ,      0.,
    #  0. ,        0.08609381, 0.14237392, 0.10576059, 0.44316162, 0.20998022,
    #  0.10576059, 0.08609381, 0.5050586 ])
    # # data_Y_test[[0,1,9,12,13,14,15,16,20,21,28,32]]=1
    
    data_Y_train=np.delete(Y_true,ii,0)
    # row=(data_Y%1!=0)
    # data_Y=Y_true
    # data_Y_train=np.concatenate((Y_true[:ii],data_Y[ii:ii+1],Y_true[ii+1:]),axis=0)
    # print(data_Y[ii])
    # data_Y_train[ii]=data_Y[ii]
    # data_Y_train[:ii]=Y_true[:ii]
    # data_Y_train[ii+1:]=Y_true[ii+1:]
    
    
    # print(data_Y_train[ii],np.copy(data_Y[ii]))
    data_Y_train=data_Y
    # row=(data_Y_train%1!=0)
    # data_Y_train=data_Y_train[row]
    # data_Y_train=data_Y
    # print(data_Y_train)
    data_Y_test=data_Y[ii:ii+1]
    data_X_train=np.delete(data_X[:,:15],ii,0)

    

    data_X_train=data_X[:,:15]
    # data_X_train=data_X
    # data_X_train=np.concatenate((label_matrix_temp,data_X_train),axis=1)
    # print(data_X_train.shape)
    
    # data_X_train=data_X
    data_X_test=data_X[ii:ii+1,:15]
    mu=np.mean(data_X_train,axis=0)
    std=np.std(data_X_train,axis=0)
    data_X_train=data_X_train-mu
    data_X_train=data_X_train/std
    data_X_test=data_X_test-mu
    data_X_test=data_X_test/std
    # data_X_train=data_X_test #[[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,31,32]]
    # data_Y_train=data_Y_test #[[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,31,32]]
    model = MLPClassifier(solver='adam', alpha=1e-5,
                         hidden_layer_sizes=(5), random_state=1)
    # model=RandomForestClassifier()
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(data_X_train)
    # model = svm.SVC(kernel='sigmoid')
    # model = keras.models.Sequential()
    # model.add(keras.Input(shape=(15,)))
    # model.add(keras.layers.Dense(30, activation='relu'))
    # # model.add(keras.layers.Dense(30, activation='relu'))
    # # model.add(keras.layers.Dense(10, activation='relu'))
    # model.add(keras.layers.Dense(1, activation='sigmoid'))
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(data_X_train,data_Y_train,batch_size=1,epochs=50,verbose=0)
    model.fit(data_X_train,data_Y_train)
    # # model.fit(data_X_train,data_Y_train)
    # y_pred=np.round(model.predict(data_X_test))
    y_pred=model.predict_proba(data_X_test)[0][1]
    # y_pred=np.round(kmeans.predict(data_X_test))
    # y_pred=data_Y[ii]
    y_true=Y_true[ii]
    print(subjects_PD[ii],'pred',y_pred,'label',y_true)
    y_pred_tot[ii]=y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_true,np.round(y_pred_tot))
# y_pred_tot[y_pred_tot>=0.5]=1
# y_pred_tot[y_pred_tot<0.5]=0
# accuracy = (Y_true == y_pred_tot).sum() 
print('accuracy',accuracy)
from sklearn.metrics import precision_recall_fscore_support
print('Pre Rec F1:',precision_recall_fscore_support(Y_true, np.round(y_pred_tot), average='macro'))
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
# RocCurveDisplay.from_predictions(Y_true, y_pred_tot)
# RocCurveDisplay.from_predictions(Y_true, Y_pred_Lu)
fpr, tpr, _ = metrics.roc_curve(Y_true, y_pred_tot)
auc = round(metrics.roc_auc_score(Y_true, y_pred_tot), 4)
plt.plot(fpr,tpr,label="Ours, AUC="+str(auc))
fpr, tpr, _ = metrics.roc_curve(Y_true, Y_pred_Lu)
auc = round(metrics.roc_auc_score(Y_true, Y_pred_Lu), 4)
plt.plot(fpr,tpr,label="Lu et al., AUC="+str(auc))
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.legend()
plt.show()
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(Y_true, y_pred_tot,normalize='true',cmap='Blues')
plt.show()
# print(data['data'].files)


# fig = plt.figure(figsize = (10, 7))
# ax1 = plt.axes()
# plt.rcParams['font.size'] = 8

# kmeans = KMeans(n_clusters=2, random_state=0).fit(data_train)
# pred=kmeans.predict(data_train)    
# label1=data_train[pred==0] 
# subjects_1=np.asarray(subjects)[pred==0]
# label2=data_train[pred==1]   
# subjects_2=np.asarray(subjects)[pred==1]
# print(subjects_1)
# for ii in range(len(label1)):
#     ax1.scatter(label1[ii,0],label1[ii,1],label=subjects_1[ii],marker='*')
# for jj in range(len(label2)):
#     ax1.scatter(label2[jj,0],label2[jj,1],label=subjects_2[jj],marker='o')
# plt.legend()
# plt.xlabel('Right Hand Range of Motion')
# plt.ylabel('Left Hand Range of Motion')
# plt.show()

    # fig = plt.figure(figsize = (10, 7))
    # ax = plt.axes(projection ="3d")
    # ax.scatter3D(step_length_n[:],hand_mov_n_l[:],foot_height_n_l[:])
    # ax.scatter3D(step_length_m[:],hand_mov_m_l[:],foot_height_m_l[:])
    # ax.scatter3D(step_length_s[:],hand_mov_s_l[:],foot_height_s_l[:])
    # plt.legend([n,m,s])
    # ax.set_xlabel('Step Length')
    # ax.set_ylabel('Right Hand Range of Motion')
    # ax.set_zlabel('Left Foot Clearance')
    # plt.show()

    # fig = plt.figure(figsize = (10, 7))
    # ax = plt.axes(projection ="3d")
    # ax.scatter3D(step_length_n[:],hand_mov_n_l[:]-hand_mov_n_r[:],foot_height_n_l[:]-foot_height_n_r[:])
    # ax.scatter3D(step_length_m[:],hand_mov_m_l[:]-hand_mov_m_r[:],foot_height_m_l[:]-foot_height_m_r[:])
    # ax.scatter3D(step_length_s[:],hand_mov_s_l[:]-hand_mov_s_r[:],foot_height_s_l[:]-foot_height_s_r[:])
    # plt.legend([n,m,s])
    # ax.set_xlabel('Step Length')
    # ax.set_ylabel('Hand Motion Symmetry')
    # ax.set_zlabel('Foot Clearance Symmetry')
    # plt.show()

    # plt.scatter(hand_mov_n_r[:3900],hand_mov_n_l[:3900])
    # plt.scatter(hand_mov_m_r[:100],hand_mov_m_l[:100])
    # plt.scatter(hand_mov_s_r[:300],hand_mov_s_l[:300])
    # plt.legend(['Normal','Moderate','Slight'])
    # plt.xlabel('Right Hand Range of Motion')
    # plt.ylabel('Left Hand Range of Motion')
    # plt.show()
    # # #20200127: Slight S37
    # # ### 20200227: Slight
    # # #20200123 : Moderate S35
    # # #20200214 :slight S43


# [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,11,12,13,14,15]
# [55,69,72,72,62,72,72,72,72,72,82,82,82,82,82]
# [- ,- ,51,55,69,69,69,65,69,72,72,69,76,82,82]

# [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ]
# [44,65,72,72,65,82,79,82,82]
# [- ,- ,65,62,76,89,86,86,86]
# accuracy 0.896551724137931
# Pre Rec F1: (0.8995098039215685, 0.8918269230769231, 0.8945454545454545, None)


