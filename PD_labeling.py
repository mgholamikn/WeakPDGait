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


subjects_PD=['S01','S02','S03','S04','S05','S06','S07','S09', 
'S10','S11','S12','S13','S14','S16','S17','S18','S19',
'S21','S22','S23','S24','S28','S29','S30','S31',
'S32','S33','S34','S35']
subjects_All=['S01','S02','S03','S04','S05','S06','S07','S08','S09',
'S10','S11','S12','S13','S14','S16','S17','S18','S19','S20',
'S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31',
'S32','S33','S34','S35']
subjects_All_date=['20210223','20191114','20191120','20191112','20191119','20200220','20191121',
'20191126','20191128','20191203','20191204','20200108','20200109','20200121','20200123',
'20200124','20200127','20200130','20200205','20200206_9339','20200207','20200213','20200214','20200218',
'26','20200221','20210706','20210804','20200206_9629','20210811','20191210','20191212','20191218','20200227']
healthy_controls=['S08','S20','S27','S25','S26']
Y_true=np.asarray([1,1,0,1,1,0,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,0,0,1,1,0,1,0,1])


plt.rcParams['font.size'] = 10
plt.rc('axes', labelsize=15) 
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15) 
fig,ax = plt.subplots(3,3,figsize = (12,7))

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

    num_sample+=len(data_normal)

    ############# Step Width #################
    ##########################################
    step_width=np.abs(data_normal[:,3,0]-data_normal[:,6,0])

    bone_length=np.linalg.norm((data_normal[:,1]-data_normal[:,4]),axis=-1)

    step_width=step_width/bone_length

    A=[]
    for ii in range(30,len(step_width)):
        A.append(np.mean(step_width[ii-30:ii]))
        output_timeseries[subj][0].append(np.mean(step_width[ii-30:ii]))
    step_width=np.asarray(A)


    output[subj][0]=np.mean(step_width)
    output_csv[idx,1]=np.mean(step_width)


    print(subj,'step_width',np.mean(step_width))    
    ############# Step Length R ##############
    ########################################
    step_length=np.linalg.norm((data_normal[:,3]-data_normal[:,6]),axis=-1)
    row_r=(data_normal[:,6,2]-data_normal[:,3,2]>0)
    row_l=(data_normal[:,3,2]-data_normal[:,6,2]>0)
    step_length_r=step_length[row_r]
    step_length_l=step_length[row_l]

    bone_length_r=np.linalg.norm((data_normal[:,5]-data_normal[:,6]),axis=-1)[row_r]
    bone_length_l=np.linalg.norm((data_normal[:,3]-data_normal[:,2]),axis=-1)[row_l]
    bone_length=np.linalg.norm((data_normal[:,5]-data_normal[:,6]),axis=-1)
    step_length=step_length /bone_length
    step_length_r=step_length_r/bone_length_r
    step_length_l=step_length_l /bone_length_l

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

    print(subj,'step_length_r',np.mean(step_length_r))
    print(subj,'step_length_l',np.mean(step_length_l))

    ############# Cadence ##############
    ########################################
    toe_traj=np.abs(data_normal[:,6,2]-data_normal[:,3,2])
    peaks, _ = find_peaks(toe_traj,distance=5,height=-0.2)
    ave=np.mean(toe_traj[peaks])-0.3
    peaks, _ = find_peaks(toe_traj,distance=5,height=ave)

    if subj in ['S01','S28','S29','S31']:
        cadence=60/((peaks[1:]-peaks[:-1])/30)
        gait_speed=toe_traj[peaks[1:]]*cadence
    else:
        cadence=60/((peaks[1:]-peaks[:-1])/15)
        gait_speed=toe_traj[peaks[1:]]*cadence

    output_timeseries[subj][3]=cadence
    output[subj][3]=np.mean(cadence)
    output_csv[idx,4]=np.mean(cadence)

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


    output[subj][4]=np.mean(foot_height_n_r)
    output[subj][5]=np.mean(foot_height_n_l)
    output_csv[idx,5]=np.mean(foot_height_n_r)
    output_csv[idx,6]=np.mean(foot_height_n_l)


    print(subj,'foot_height_n_r',np.mean(foot_height_n_r))
    print(subj,'foot_height_n_l',np.mean(foot_height_n_l))
    

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
    print(subj,'hand_mov_n_r',np.mean(hand_mov_n_r))

    ############# Hand Movement L ##############
    ########################################
    ave=np.mean(data_normal[:,11,:],axis=0,keepdims=True)
    dist=np.linalg.norm((data_normal[:,11]-data_normal[:,1]),axis=-1)
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

    if np.mean(hand_mov_n_r)<0.189 or np.mean(hand_mov_n_l)<0.2:
        subj_pd.append(subj)
    print(subj,'hand_mov_n_l',np.mean(hand_mov_n_l))

    ## data for clustering
    data_train[num,0]=np.mean(hand_mov_n_r[:])
    data_train[num,1]=np.mean(hand_mov_n_l[:])
    

    ############# Hip Flexion ##############
    ########################################
    dist_l=data_normal[:,1,2]-data_normal[:,2,2]
    bone_l=np.linalg.norm((data_normal[:,1]-data_normal[:,2]),axis=-1)
    dist_r=data_normal[:,4,2]-data_normal[:,5,2]
    bone_r=np.linalg.norm((data_normal[:,4]-data_normal[:,5]),axis=-1)

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

    print(subj,'hip_flex_l',np.mean(hip_flex_l))
    print(subj,'hip_flex_r',np.mean(hip_flex_r))


    ############# Knee Flexion ##############
    ########################################
    thigh_l=np.linalg.norm((data_normal[:,1]-data_normal[:,2]),axis=-1)
    shin_l=np.linalg.norm((data_normal[:,3]-data_normal[:,2]),axis=-1)
    leg_l=np.linalg.norm((data_normal[:,1]-data_normal[:,3]),axis=-1)
    thigh_r=np.linalg.norm((data_normal[:,5]-data_normal[:,4]),axis=-1)
    shin_r=np.linalg.norm((data_normal[:,5]-data_normal[:,6]),axis=-1)
    leg_r=np.linalg.norm((data_normal[:,4]-data_normal[:,6]),axis=-1)

    knee_flex_r=leg_r**2/(thigh_r*shin_r)-thigh_r/shin_r-shin_r/thigh_r
    knee_flex_l=leg_l**2/(thigh_l*shin_l)-thigh_l/shin_l-shin_l/thigh_l
    A=[]
    B=[]

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
    

    trunk_rot_r=hip2shoulder_r**2/(hip_r*shoulder_r)-hip_r/shoulder_r-shoulder_r/hip_r
    trunk_rot_l=hip2shoulder_l**2/(hip_l*shoulder_l)-hip_l/shoulder_l-shoulder_l/hip_l
    A=[]
    B=[]

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

    print(subj,'trunk_rot_l',np.mean(trunk_rot_l))
    print(subj,'trunk_rot_r',np.mean(trunk_rot_r))

    num+=1
print('Average number of samples:',num_sample/num)


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
threshold_StepL_r=1000
threshold_StepL_l=1000
threshold_footlift_r=1000
threshold_footlift_l=1000
threshold_Hip_r=1000
threshold_Hip_l=1000
threshold_knee_r=1000
threshold_knee_l=1000
threshold_trunk_r=1000
threshold_trunk_l=1000
threshold_stepwidth=-1000
threshold_cadence=-1000
threshold_handROM_r=1000
threshold_handROM_l=1000
threshold_gait_speed=1000
threshold_gait_speed_var=-1000
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

    ii+=1        
from snorkel.labeling.model import LabelModel

print(label_matrix)
print(np.sum(label_matrix,axis=1))

Y_pred=np.zeros(len(subjects_PD))
jj=0
data_X=np.zeros((len(subjects_PD),30))

for subj in subjects_PD:
    data_X[jj]=data[subj]
    jj+=1
accuracy=0
count=0
y_pred_tot=np.zeros((len(subjects_PD)))
  

for ii in range(len(subjects_PD)):

    label_model = LabelModel(verbose=False)
    label_matrix_temp=np.delete(label_matrix,ii,0)
    label_model.fit(label_matrix_temp, seed=123)
    weights=label_model.get_weights()
    weights/=np.sum(weights)

    data_Y=np.sum(weights*label_matrix,axis=1)
    
    # data_Y_train=np.delete(Y_true,ii,0)
    data_Y_test=data_Y[ii:ii+1]
    # data_X_train=np.delete(data_X[:,:15],ii,0)
    data_X_train=data_X[:,:15]
    data_X_test=data_X[ii:ii+1,:15]
    
    mu=np.mean(data_X_train,axis=0)
    std=np.std(data_X_train,axis=0)
    data_X_train=data_X_train-mu
    data_X_train=data_X_train/std
    data_X_test=data_X_test-mu
    data_X_test=data_X_test/std
    

    model = keras.models.Sequential()
    model.add(keras.Input(shape=(15,)))
    model.add(keras.layers.Dense(30, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(data_X_train,data_Y,batch_size=1,epochs=50,verbose=0)

    y_pred=model.predict(data_X_test)[0][1]
    y_true=Y_true[ii]
    
    print(subjects_PD[ii],'pred',y_pred,'label',y_true)
    y_pred_tot[ii]=y_pred

accuracy=accuracy_score(Y_true,np.round(y_pred_tot))
print('accuracy',accuracy)
# from sklearn.metrics import precision_recall_fscore_support
# print('Pre Rec F1:',precision_recall_fscore_support(Y_true, np.round(y_pred_tot), average='macro'))



