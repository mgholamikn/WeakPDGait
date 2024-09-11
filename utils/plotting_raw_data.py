from matplotlib import animation, colors
import numpy as np
from scipy.signal.wavelets import cascade
from plot import *
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
plt.rc('axes', labelsize=12) 
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12) 

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
def rotation(data_normal):
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

    return data_normal

for idx,(subj,subj_date) in enumerate(zip(subjects_All,subjects_All_date)):
    print(subj_date)
    fig,ax = plt.subplots(2,2,figsize = (12,7))
    output_csv[idx,0]=subj_date
    output[subj]=np.zeros(30)
    output_timeseries[subj]={}
    data_normal_pt_f=np.load('outputs_finetuned/Predictions_'+'S03'+'.npy')
    data_normal_pt_h=np.load('outputs_finetuned/Predictions_'+'S07'+'.npy')
    for feature in range(30):
        output_timeseries[subj][feature]=[]
    data_normal_pt_f=rotation(data_normal_pt_f)
    data_normal_pt_h=rotation(data_normal_pt_h)
    for ii in range(15):
        for jj in range(3):
            data_normal_pt_f[:,ii,jj]=savgol_filter(data_normal_pt_f[:,ii,jj],11,3)
            data_normal_pt_h[:,ii,jj]=savgol_filter(data_normal_pt_h[:,ii,jj],11,3)
    rows=np.concatenate((np.arange(0,600),np.arange(1000,2600),np.arange(2800,3900),np.arange(4000,4200)))
    time=np.arange(300)/15
    time1=np.arange(200)/15
    # fig1,ax1 = plt.subplots(1,1)
    ax[0,0].plot(time,data_normal_pt_f[0:300,3,0],'deepskyblue',linewidth=3)
    ax[0,0].plot(time,data_normal_pt_f[0:300,3,1],'yellowgreen',linewidth=3)
    ax[0,0].plot(time,data_normal_pt_f[0:300,3,2],'crimson',linewidth=3)
    ax[0,0].set_title('Fine-tuned 3D Pose Estimator (trajectory of left foot)')
    ax[0,0].legend(['x','y','z'],loc='right')
    ax[0,0].set_xlabel('Time(s)')
    ax[0,0].set_ylabel('Trajectory')

    ax[0,1].plot(time1,data_normal_pt_h[0:300,14,0],'deepskyblue',linewidth=3)
    ax[0,1].plot(time1,data_normal_pt_h[0:300,14,1],'yellowgreen',linewidth=3)
    ax[0,1].plot(time1,data_normal_pt_h[0:300,14,2],'crimson',linewidth=3)
    ax[0,1].set_title('Fine-tuned 3D Pose Estimator (trajectory of left hand)')
    ax[0,1].legend(['x','y','z'],loc='right')
    ax[0,1].set_xlabel('Time(s)')
    ax[0,1].set_ylabel('Trajectory')

    data_normal_ft_f=np.load('outputs_pretrained/Predictions_'+'S03'+'.npy')
    data_normal_ft_h=np.load('outputs_pretrained/Predictions_'+'S07'+'.npy')
    for feature in range(30):
        output_timeseries[subj][feature]=[]
    data_normal_ft_f=rotation(data_normal_ft_f)
    data_normal_ft_h=rotation(data_normal_ft_h)
    for ii in range(15):
        for jj in range(3):
            data_normal_ft_f[:,ii,jj]=savgol_filter(data_normal_ft_f[:,ii,jj],11,3)
            data_normal_ft_h[:,ii,jj]=savgol_filter(data_normal_ft_h[:,ii,jj],11,3)

    # fig1,ax1 = plt.subplots(1,1)
    ax[1,0].plot(time,data_normal_ft_f[0:300,3,0],'deepskyblue',linewidth=3)
    ax[1,0].plot(time,data_normal_ft_f[0:300,3,1],'yellowgreen',linewidth=3)
    ax[1,0].plot(time,data_normal_ft_f[0:300,3,2],'crimson',linewidth=3)
    ax[1,0].set_title('Pretrained 3D Pose Estimator (trajectory of left foot)')
    # ax1[1].plot(data_normal[:,6,:])
    ax[1,0].legend(['x','y','z'],loc='right')
    ax[1,0].set_xlabel('Time(s)')
    ax[1,0].set_ylabel('Trajectory')


    ax[1,1].plot(time1,data_normal_ft_h[0:300,14,0],'deepskyblue',linewidth=3)
    ax[1,1].plot(time1,data_normal_ft_h[0:300,14,1],'yellowgreen',linewidth=3)
    ax[1,1].plot(time1,data_normal_ft_h[0:300,14,2],'crimson',linewidth=3)
    
    # ax1[1].plot(data_normal[:,6,:])
    ax[1,1].legend(['x','y','z'],loc='right')
    ax[1,1].set_title('Pretrained 3D Pose Estimator (trajectory of left hand)')
    ax[1,1].set_xlabel('Time(s)')
    ax[1,1].set_ylabel('Trajectory')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.show()