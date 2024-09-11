from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error, accuracy_score
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew, kurtosis
from sklearn.utils import shuffle

##################################### Start of Unsupervised Clustering #####################################
def plot17j_2d(poses, show_animation=False):
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.animation as anim

    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if not show_animation:
        plot_idx = 1
# 
        frames = np.linspace(start=0, stop=poses.shape[0], num=poses.shape[0]).astype(int)
        
        for i in range(poses.shape[0]):
            # ax = fig.add_subplot(1, poses.shape[0], plot_idx, projection='3d')

            # pose = poses[i,1:]
            # pose = pose[[2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 12, 11, 10], :]
            # pose=np.reshape(pose,(16*3,))
            
            # x = pose[0:16]
            # y = pose[16:32]
            # z = pose[32:48]

            x = poses[i,:,0]
            y = poses[i,:,1]
        
            ax.scatter(x, y)

            kkkkk=5

            ax.scatter(poses[i,kkkkk,0],poses[i,kkkkk,1])

            linewidth=3
            ax.plot(poses[i,[0,1],0], poses[i,[0,1],1],linewidth=linewidth)
            ax.plot(poses[i,[1,2],0], poses[i,[1,2],1],linewidth=linewidth)
            ax.plot(poses[i,[2,3],0], poses[i,[2,3],1],linewidth=linewidth)
            ax.plot(poses[i,[0,4],0], poses[i,[0,4],1],linewidth=linewidth)
            ax.plot(poses[i,[4,5],0], poses[i,[4,5],1],linewidth=linewidth)
            ax.plot(poses[i,[5,6],0], poses[i,[5,6],1],linewidth=linewidth)
            ax.plot(poses[i,[0,7],0], poses[i,[0,7],1],linewidth=linewidth)
            ax.plot(poses[i,[7,8],0], poses[i,[7,8],1],linewidth=linewidth)
            ax.plot(poses[i,[8,9],0], poses[i,[8,9],1],linewidth=linewidth)
            ax.plot(poses[i,[9,10],0], poses[i,[9,10],1],linewidth=linewidth)
            ax.plot(poses[i,[8,11],0], poses[i,[8,11],1],linewidth=linewidth)
            ax.plot(poses[i,[11,12],0], poses[i,[11,12],1],linewidth=linewidth)
            ax.plot(poses[i,[12,13],0], poses[i,[12,13],1],linewidth=linewidth)
            ax.plot(poses[i,[8,14],0], poses[i,[8,14],1],linewidth=linewidth)
            ax.plot(poses[i,[14,15],0], poses[i,[14,15],1],linewidth=linewidth)
            ax.plot(poses[i,[15,16],0], poses[i,[15,16],1],linewidth=linewidth)

            # Create cubic bounding box to simulate equal aspect ratio
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

    else:
        def update(i):

            ax.clear()

            pose = poses[i]

            x = pose[0:16]
            y = pose[16:32]
            z = pose[32:48]
            ax.scatter(x, y, z)

            ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])])
            ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])])
            ax.plot(x[([3, 4])], y[([3, 4])], z[([3, 4])])
            ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])])
            ax.plot(x[([0, 6])], y[([0, 6])], z[([0, 6])])
            ax.plot(x[([3, 6])], y[([3, 6])], z[([3, 6])])
            ax.plot(x[([6, 7])], y[([6, 7])], z[([6, 7])])
            ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])])
            ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])])
            ax.plot(x[([7, 10])], y[([7, 10])], z[([7, 10])])
            ax.plot(x[([10, 11])], y[([10, 11])], z[([10, 11])])
            ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])])
            ax.plot(x[([7, 13])], y[([7, 13])], z[([7, 13])])
            ax.plot(x[([13, 14])], y[([13, 14])], z[([13, 14])])
            ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])])

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

            plt.axis('equal')

        a = anim.FuncAnimation(fig, update, frames=poses.shape[0], repeat=False)
        plt.show()
    
    return


###################################################
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter requirements.
order = 6
fs = 30      # sample rate, Hz
cutoff = 1  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)
########################################################################3

data={}
data['pos']=[]
data['label']=[]
time=[]
data_train=[]
data_test=[]
label_train=[]
label_test=[]
files=['20210223','20200227','20200221','20200218']
fs=[30,15,15,15]
label=[1,1,0,0]
start_stop=[[500,2400],[0,-1],[120,-1],[100,-1]]
for ii in range(4):
    data_2d=np.load(files[ii]+'.npz',allow_pickle=True)
    data_2d=data_2d['positions_2d'].item()
    data_2d=data_2d['output.mp4']['custom'][0]
    # data_2d=data[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16]]
    print(data_2d.shape)
    if ii==1:
        row=np.where(data_2d[:,16,0]>700)
    else:
        row=np.where(data_2d[:,10,1]<1000)
    data_2d=data_2d[row][start_stop[ii][0]:start_stop[ii][1]]
    data_2d-=data_2d[:,:1]
    row=np.where(data_2d[:,13,1]>300)
    data_2d=data_2d[row]
    data_2d/=np.linalg.norm(data_2d,axis=-1,keepdims=True)+0.00001
    row=np.where(data_2d[:,16,1]>0.9)
    data_2d=data_2d[row]
    row=np.where(data_2d[:,13,1]>0.9)
    data_2d=data_2d[row]
    time.append(np.array(range(len(data_2d)))/fs[ii])
    data['pos'].append(data_2d)
    data['label'].append(np.ones((len(data_2d)))*label[ii])
    train_length=round(0.8*len(data_2d))
    test_length=round(0.2*len(data_2d))
    data_train.append(data_2d[0:train_length])
    label_train.append(np.ones((train_length))*label[ii])
    data_test.append(data_2d[-test_length:])
    label_test.append(np.ones((test_length))*label[ii])
    
# y = butter_lowpass_filter(data_2d[:,10,1], cutoff, fs, order)


# for jj in range(20):
#     fft=np.fft.fft(data['pos'][0][jj*100:jj*100+100,:,1])
#     freq=np.fft.fftfreq(len(data['pos'][0][jj*100:jj*100+100]),1/15)
#     mask=freq>0
#     fft=2*np.abs(fft/len(data['pos'][0]))
#     plt.plot(freq[mask],fft[mask])
#     plt.show()

fig,ax=plt.subplots(4)

ax[0].plot(data['pos'][0][:,3,1],'b');ax[0].plot(data['pos'][0][:,6,1],'r') #-data[0][:,13,0]
ax[1].plot(data['pos'][1][:,16,1],'b');ax[1].plot(data['pos'][1][:,13,1],'r') #-data[1][:,13,0]
ax[2].plot(data['pos'][2][:,16,1],'b');ax[2].plot(data['pos'][2][:,13,1],'r') #-data[2][:,13,0]
ax[3].plot(data['pos'][3][:,16,1],'b');ax[3].plot(data['pos'][3][:,13,1],'r') #-data[3][:,13,0]

ax[0].set_title('slight')
ax[1].set_title('slight')
ax[2].set_title('normal')
ax[3].set_title('normal')

plt.show()

def data_preprocess(data,pad):
    # X=np.zeros((len(data),2*pad,17,2))
    n_features=8
    X=np.zeros((len(data),n_features,17,2))
    for ii in range(pad,len(data)-pad):
        # X[ii]=np.reshape(data[ii-pad:ii+pad,:],(1,2*pad,17,2))
        x_time=data[ii-pad:ii+pad]
        x_freq=np.fft.fft(x_time)
        X[ii,0]=np.reshape(np.mean(x_time,axis=0),(1,17,2))
        X[ii,1]=np.reshape(np.min(x_time,axis=0),(1,17,2))
        X[ii,2]=np.reshape(np.max(x_time,axis=0),(1,17,2))
        X[ii,3]=np.reshape(np.std(x_time,axis=0),(1,17,2))
        # X[ii,4]=np.reshape(skew(x_time,axis=0),(1,17,2))
        # X[ii,5]=np.reshape(kurtosis(x_time,axis=0),(1,17,2))
        X[ii,4]=np.reshape(np.mean(x_freq.real,axis=0),(1,17,2))
        X[ii,5]=np.reshape(np.min(x_freq.real,axis=0),(1,17,2))
        X[ii,6]=np.reshape(np.max(x_freq.real,axis=0),(1,17,2))
        X[ii,7]=np.reshape(np.std(x_freq.real,axis=0),(1,17,2))
        # X[ii,10]=np.reshape(skew(x_freq,axis=0),(1,17,2))
        # X[ii,11]=np.reshape(kurtosis(x_freq,axis=0),(1,17,2))
        # X[ii,4]=np.reshape(np.min(data[ii-pad:ii+pad],axis=0),(1,17,2))
        
    return X

label={}
for ii in range(4):
    pad=50
    n_features=8
    # data['pos'][ii]=data_preprocess(data['pos'][ii],pad)
    data_train[ii]=data_preprocess(data_train[ii],pad)
    data_test[ii]=data_preprocess(data_test[ii],pad)
    # data['pos'][ii]=np.reshape(data['pos'][ii],(-1,2*pad*2*2))
    # data['pos'][ii]=np.reshape(data['pos'][ii],(-1,n_features*17*2))
    data_train[ii]=np.reshape(data_train[ii],(-1,n_features*17*2))
    data_test[ii]=np.reshape(data_test[ii],(-1,n_features*17*2))
    # print(data['pos'][ii].shape)
    # print(data['label'][ii].shape)
    # print('Naaan',np.sum(np.isnan(data['pos'][ii])))
    print(data_train[ii].shape)
    print(label_train[ii].shape)
    print(data_test[ii].shape)
    print(label_test[ii].shape)
    print('Naaan',np.sum(np.isnan(data_train[ii])))
    
    
# data['pos']=np.concatenate(data['pos'])
# data['label']=np.concatenate(data['label'])
# print(data['pos'].shape)
# print(data['label'].shape)

data_train=np.concatenate(data_train)
label_train=np.concatenate(label_train)
data_test=np.concatenate(data_test)
label_test=np.concatenate(label_test)
# print(data['pos'].shape)
# print(data['label'].shape)
print(data_train.shape)
print(label_train.shape)
print(data_test.shape)
print(label_test.shape)

# print('########',np.arange(10).shape)
# train_length=round(0.8*len(data['pos']))
# train_data=data['pos'][:train_length]
# train_label=data['label'][:train_length]
# test_data=data['pos'][train_length:]
# test_label=data['label'][train_length:]
shuffled_rows=np.arange(len(data_train))
np.random.shuffle(shuffled_rows)
# print(data['label'].shape)
# print('Naaan',np.sum(np.isnan(data['pos'])))
data_train=data_train[shuffled_rows]
label_train=label_train[shuffled_rows]


# print(data['pos'].shape)
# print(data['label'].shape)
# print('Naaan',np.sum(np.isnan(data['pos'])))
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf.fit(data_train,label_train)
# pred=clf.predict(data_test)
kmeans = KMeans(n_clusters=2, random_state=0).fit(data_train)
pred=kmeans.predict(data_test)
print(accuracy_score(label_test,pred))
# print(pred[100])
for ii in range(len(label_test)):
    if label_test[ii]==0:
        label_test[ii]=1
    elif label_test[ii]==1:
        label_test[ii]=0

print(accuracy_score(label_test,pred))
# label1=np.array(np.where(data['label'][train_length:]>0))
# label0=np.array(np.where(data['label'][train_length:]<1))
# print(label1)
# plt.scatter(data['pos'][train_length+label1,15], data['pos'][train_length+label1,10], marker='^', label='slight')
# plt.scatter(data['pos'][train_length+label0,15], data['pos'][train_length+label0,10], marker='o', label='normal')
# plt.plot(kmeans.labels_,'b')

plt.plot(label_test,'b')
plt.plot(pred,'r')


# plt.xlabel('feature1')
# plt.ylabel('feature2')
plt.show()
# plt.plot(label_train,'b')
# plt.show()

###################### End of Unsupervised Clustering ###############################################

# #####################   Test Sensor Data ##############################################################
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy import signal
# data_sensor=pd.read_csv('sensor_data.csv')
# data_sensor=np.asarray(data_sensor)
# data_sensor_ds=signal.resample(data_sensor[922:round(3528/30*128)],3528)


# import json
# for ii in range(1):
#     with open('alphapose-results-ch7.json') as f:
#         data=json.load(f)
# frame_size=np.zeros(4046)
# kpts=np.zeros((4046,15,2))
# for jj in range(0,len(data)):
#     for frame_idx in range(0,4046):
#         if int(data[jj]['image_id'][6:12])==(frame_idx+1):
#             if frame_size[frame_idx]<abs(data[jj]['box'][2]-data[jj]['box'][0]):
#                 frame_size[frame_idx]=abs(data[jj]['box'][2]-data[jj]['box'][0])
#                 x=np.asarray(data[jj]['keypoints'])[[19*3,11*3,13*3,15*3,12*3,14*3,16*3,18*3,17*3,6*3,8*3,10*3,5*3,7*3,9*3]]
#                 y=np.asarray(data[jj]['keypoints'])[[19*3+1,11*3+1,13*3+1,15*3+1,12*3+1,14*3+1,16*3+1,18*3+1,17*3+1,6*3+1,8*3+1,10*3+1,5*3+1,7*3+1,9*3+1]]
#                 z=np.asarray(data[jj]['keypoints'])[[19*3+2,11*3+2,13*3+2,15*3+2,12*3+2,14*3+2,16*3+2,18*3+2,17*3+2,6*3+2,8*3+2,10*3+2,5*3+2,7*3+2,9*3+2]]
#                 xy=np.concatenate((np.expand_dims(x,axis=-1),np.expand_dims(y,axis=-1)),axis=-1)
#                 kpts[frame_idx]=xy
# # kpts-=kpts[:,:1]
# kpts=kpts/np.mean(np.linalg.norm(kpts[:,:,:],axis=-1,keepdims=True),axis=-2,keepdims=True)
# plt.plot(kpts[517:,14,:])
# plt.plot(data_sensor_ds[:,8])
# plt.show()