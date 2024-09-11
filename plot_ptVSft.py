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
subjects_PD=['S01','S02','S03','S04','S05','S06','S07','S09',
'S10','S11','S12','S13','S14','S16','S17','S18','S19',
'S21','S22','S23','S24','S28','S29','S30','S31',
'S32','S33','S34','S35']
subjects_All=['S01','S02','S03','S04','S05','S06','S07','S08','S09',
'S10','S11','S12','S13','S14','S16','S17','S18','S19','S20',
'S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31',
'S32','S33','S34','S35']
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
subj_pd=[]
for subj in subjects_All:
    output[subj]=np.zeros(30)
    #################################################################
    data_normal0=np.load('outputs_finetuned/Predictions_'+subj+'.npy')
    
    for ii in range(15):
        for jj in range(3):
            data_normal0[:,ii,jj]=savgol_filter(data_normal0[:,ii,jj],11,3)
    rows=np.concatenate((np.arange(0,600),np.arange(1000,2600),np.arange(2800,3900),np.arange(4000,4200)))
    x_vec=data_normal0[:,1]-data_normal0[:,4]
    y_vec=data_normal0[:,7]-data_normal0[:,0]
    x_vec/=np.linalg.norm(x_vec,keepdims=True,axis=-1)
    y_vec/=np.linalg.norm(y_vec,keepdims=True,axis=-1)
    z_vec=np.cross(x_vec,y_vec)
    rotation_matrix=np.ones((len(x_vec),3,3))
    rotation_matrix[:,:,0]=x_vec
    rotation_matrix[:,:,1]=y_vec
    rotation_matrix[:,:,2]=z_vec
    data_normal0=np.matmul(data_normal0,rotation_matrix)
    scale0=np.linalg.norm(data_normal0[:,0]-data_normal0[:,7])
    #################################################################
    data_normal=np.load('outputs_pretrained/Predictions_'+subj+'.npy')
    
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
    scale=np.linalg.norm(data_normal[:,0]-data_normal[:,7])
    data_normal*=scale0/scale

    
    

    ###############################################################33
    print(subj)
    plot15j_PD(np.concatenate((data_normal0[20:21],data_normal[20:21]),axis=0),show_animation=False,color=['g','b'])
    