import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import axes3d, Axes3D
def plot_16j(poses,show_animation=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_idx = 1
        frames = np.linspace(start=0, stop=poses.shape[0]-1, num=2).astype(int)
        if not show_animation:
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
                    ax.plot(poses[i,[8,9],0], poses[i,[8,9],1],linewidth=width,color='green')
                    ax.plot(poses[i,[8,10],0], poses[i,[8,10],1],linewidth=width,color='blue')
                    ax.plot(poses[i,[10,11],0], poses[i,[10,11],1],linewidth=width,color='blue')
                    ax.plot(poses[i,[11,12],0], poses[i,[11,12],1],linewidth=width,color='blue')
                    ax.plot(poses[i,[8,13],0], poses[i,[8,13],1],linewidth=width,color='red')
                    ax.plot(poses[i,[13,14],0], poses[i,[13,14],1],linewidth=width,color='red')
                    ax.plot(poses[i,[14,15],0], poses[i,[14,15],1],linewidth=width,color='red')

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
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            # plt.grid(True)
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

def process_data(data,channel,camera,len_data):
    if channel==1 and camera=='new':
        crop=2800
    elif channel==2 and camera=='new':
        crop=2000
    elif channel==1 and camera=='old':
        crop=2700   
    elif channel==2 and camera=='old':
        crop=1950    
    print(crop)  
    frame_size=np.zeros(len_data)
    bbx_=np.zeros(len_data)
    kpt=np.zeros((len_data,16,3))
    for jj in range(0,len(data)):
        for frame_idx in range(0,len_data):
            if int(data[jj]['image_id'][6:12])==(frame_idx+1):
                conf=np.mean(np.asarray(data[jj]['keypoints'])[[19*3+2,11*3+2,13*3+2,15*3+2,12*3+2,14*3+2,16*3+2,18*3+2,17*3+2,6*3+2,8*3+2,10*3+2,5*3+2,7*3+2,9*3+2]])
                bbx=abs(data[jj]['box'][2]-data[jj]['box'][0])
                if data[jj]['keypoints'][19*3]<crop and bbx_[frame_idx]<bbx and conf>0.6: #frame_size[frame_idx]<conf
                    frame_size[frame_idx]=conf 
                    bbx_[frame_idx]=bbx
                    xy=np.zeros((16,3))
                    joints=[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15]
                    xy[joints,0]=np.asarray(data[jj]['keypoints'])[[19*3,11*3,13*3,15*3,12*3,14*3,16*3,18*3,17*3,6*3,8*3,10*3,5*3,7*3,9*3]]
                    xy[joints,1]=np.asarray(data[jj]['keypoints'])[[19*3+1,11*3+1,13*3+1,15*3+1,12*3+1,14*3+1,16*3+1,18*3+1,17*3+1,6*3+1,8*3+1,10*3+1,5*3+1,7*3+1,9*3+1]]
                    xy[joints,2]=np.asarray(data[jj]['keypoints'])[[19*3+2,11*3+2,13*3+2,15*3+2,12*3+2,14*3+2,16*3+2,18*3+2,17*3+2,6*3+2,8*3+2,10*3+2,5*3+2,7*3+2,9*3+2]]
                    xy[7,:]=(xy[0,:]+xy[8,:])/2
                    kpt[frame_idx]=xy
          
    return kpt

with open('data/20210223/Walking_Oval/alphapose-results-ch1.json') as f:
    data1=json.load(f)
    kpt1=process_data(data1,channel=1,camera='new',len_data=4046)
    # print(len(kpt1))

with open('data/20210223/Walking_Oval/alphapose-results-ch2.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='new',len_data=4046)
    # print(len(kpt2))

# 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790
## for training
# row=np.concatenate((np.arange(650,960),np.arange(1130,1375),np.arange(1545,1805),np.arange(1970,2430),np.arange(2596,2870),np.arange(3050,3320),np.arange(3500,3790))) ## for training
## for test
plot_16j(kpt1[700:701])
plot_16j(kpt2[700:701])
row=np.concatenate((np.arange(840,960),np.arange(1250,1375),np.arange(1675,1805),np.arange(2300,2430)))   ## for test
kpt1,kpt2=kpt1[row],kpt2[row]
# row=(kpt1[:,0,1]>100)*(kpt2[:,0,1]>100)
# kpt1,kpt2=kpt1[row],kpt2[row]
plt.plot(kpt1[:,0])
plt.plot(kpt2[:,0])
plt.show()
data={}
data['S01']={}
data['S01']['WalkingOval']={}
data['S01']['WalkingOval']['pos']={}
data['S01']['WalkingOval']['conf']={}
data['S01']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
data['S01']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S01']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
data['S01']['WalkingOval']['conf'][1]=kpt2[:,:,2:]
print('S01',len(kpt1))

# # np.savez_compressed('data_PD',positions_2d=data)
# # for ii in range(0,2000,100):
# #     plot_16j(np.concatenate((kpt1[ii:ii+1]*-1,kpt2[ii:ii+1]*-1),axis=0))
# # print(kpt1.shape)
# # plt.plot(kpt1[:,0])
# # plt.show()

# with open('data/20210223/Walking_Random/alphapose-results_CH1.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=1,camera='new',len_data=4360)
#     # print(len(kpt1))

# with open('data/20210223/Walking_Random/alphapose-results_CH2.json') as f:
#     data2=json.load(f)
#     kpt2=process_data(data2,channel=2,camera='new',len_data=4360)
#     # print(len(kpt2))

# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row=np.concatenate((np.arange(710,987),np.arange(1080,1692),np.arange(1820,2175),np.arange(2240,2540),np.arange(2630,2930),np.arange(2990,3200),np.arange(3300,3580),np.arange(3650,3840),np.arange(3950,4350)))
# kpt1,kpt2=kpt1[row],kpt2[row]
# row=(kpt1[:,0,1]>100)*(kpt2[:,0,1]>100)
# kpt1,kpt2=kpt1[row],kpt2[row]
# plt.plot(kpt1[:,0])
# plt.plot(kpt2[:,0])
# plt.show()
# data['S01']['WalkingRandom']={}
# data['S01']['WalkingRandom']['pos']={}
# data['S01']['WalkingRandom']['conf']={}
# data['S01']['WalkingRandom']['pos'][0]=kpt1[:,:,:2]
# data['S01']['WalkingRandom']['pos'][1]=kpt2[:,:,:2]
# data['S01']['WalkingRandom']['conf'][0]=kpt1[:,:,2:]
# data['S01']['WalkingRandom']['conf'][1]=kpt2[:,:,2:]


# # np.savez_compressed('data_PD',positions_2d=data)
# # for ii in range(0,2000,100):
# #     plot_16j(np.concatenate((kpt1[ii:ii+1]*-1,kpt2[ii:ii+1]*-1),axis=0))
# # print(kpt1.shape)
# # plt.plot(kpt1[:,0])
# # plt.show()


############## 20191114 ########################
################################################

with open('data/20191114/Walking_Oval/alphapose-results_CH1.json') as f:
    data1=json.load(f)
    kpt1=process_data(data1,channel=1,camera='old',len_data=4000)
    # print(len(kpt1))

with open('data/20191114/Walking_Oval/alphapose-results_CH2.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='old',len_data=4000)
    # print(len(kpt2))

# 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

### for training
# row1=np.concatenate((np.arange(570,675),np.arange(775,900),np.arange(1010,1165),np.arange(1290,1455),np.arange(1563,1700),np.arange(1840,1980),np.arange(2080,2210),np.arange(2460,2585),np.arange(2680,2790)
#       ,np.arange(3128,3248),np.arange(3345,3463),np.arange(3563,3685),np.arange(3783,3900))) #,np.arange(2900,3130),
# row2=np.concatenate((np.arange(560,665),np.arange(765,890),np.arange(1000,1155),np.arange(1280,1445),np.arange(1553,1690),np.arange(1830,1970),np.arange(2070,2200),np.arange(2446,2571),np.arange(2667,2777)
#      ,np.arange(3115,3235),np.arange(3332,3450),np.arange(3553,3675),np.arange(3772,3889))) # ,np.arange(2887,3117)  

### for test

row1=np.concatenate((np.arange(615,675),np.arange(840,900),np.arange(1105,1165),np.arange(1395,1455),np.arange(1640,1700))) #,np.arange(2900,3130),
row2=np.concatenate((np.arange(605,665),np.arange(830,890),np.arange(1095,1155),np.arange(1385,1445),np.arange(1630,1690))) # ,np.arange(2887,3117)  

kpt1,kpt2=kpt1[row1],kpt2[row2]
row=(kpt1[:,0,1]>10)*(kpt2[:,0,1]>10)
kpt1,kpt2=kpt1[row],kpt2[row]
# print(kpt1.shape)
# print(kpt2.shape)
# plot_16j(kpt1[:])
# plot_16j(kpt2[:])
plt.plot(kpt1[:,0])
plt.plot(kpt2[:,0])
plt.show()
data['S02']={}
data['S02']['WalkingOval']={}
data['S02']['WalkingOval']['pos']={}
data['S02']['WalkingOval']['conf']={}
data['S02']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
data['S02']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S02']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
data['S02']['WalkingOval']['conf'][1]=kpt2[:,:,2:]
print('S02',len(kpt1))

# ############## 20191120 ########################
# ################################################

# with open('data/20191120/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(220,280),np.arange(450,510),np.arange(685,755),np.arange(920,985),np.arange(1160,1230))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()
# # for ii in range(0,len(row1),10):
# # plot_16j(kpt1[0:1])
# # plt.plot(kpt1[:,0])
# # plt.show()
# # row=(kpt1[:,0,1]>500)*(kpt1[:,0,1]<1300)
# # kpt1,kpt2=kpt1[row],kpt2[row]
# data['S03']={}
# data['S03']['WalkingOval']={}
# data['S03']['WalkingOval']['pos']={}
# data['S03']['WalkingOval']['conf']={}
# data['S03']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S03']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S03']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S03']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S03',len(kpt1))

# ############## 20191112 ########################
# ################################################
# with open('data/20191112/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(200,250),np.arange(410,460),np.arange(600,670),np.arange(830,880),np.arange(1040,1100))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()
# # for ii in range(0,len(row1),10):
# #     plot_16j(kpt1[ii:ii+1])
# # plt.plot(kpt1[:,0])
# # plt.show()
# # row=(kpt1[:,0,1]>500)*(kpt1[:,0,1]<1300)
# # kpt1,kpt2=kpt1[row],kpt2[row]
# data['S04']={}
# data['S04']['WalkingOval']={}
# data['S04']['WalkingOval']['pos']={}
# data['S04']['WalkingOval']['conf']={}
# data['S04']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S04']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S04']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S04']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S04',len(kpt1))

# ############## 20191119  ########################
# ################################################
# with open('data/20191119/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(160,240),np.arange(480,560),np.arange(770,850),np.arange(1040,1140))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# # for ii in range(0,len(row1),10):
# #     plot_16j(kpt1[ii:ii+1])
# # plt.plot(kpt1[:,0])
# # plt.show()
# # row=(kpt1[:,0,1]>500)*(kpt1[:,0,1]<1300)
# # kpt1,kpt2=kpt1[row],kpt2[row]
# data['S05']={}
# data['S05']['WalkingOval']={}
# data['S05']['WalkingOval']['pos']={}
# data['S05']['WalkingOval']['conf']={}
# data['S05']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S05']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S05']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S05']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S05',len(kpt1))

# ############## 20200220  ########################
# ################################################
# with open('data/20200220/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(230,280),np.arange(430,480),np.arange(630,680),np.arange(820,870))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[:])

# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# data['S06']={}
# data['S06']['WalkingOval']={}
# data['S06']['WalkingOval']['pos']={}
# data['S06']['WalkingOval']['conf']={}
# data['S06']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S06']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S06']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S06']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S06',len(kpt1))


# ############## 20191121  ########################
# ################################################
# with open('data/20191121/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(155,195),np.arange(330,370),np.arange(500,540),np.arange(670,710),np.arange(840,880))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# # for ii in range(0,len(row1),10):
# #     plot_16j(kpt1[ii:ii+1])
# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()
# # row=(kpt1[:,0,1]>500)*(kpt1[:,0,1]<1300)
# # kpt1,kpt2=kpt1[row],kpt2[row]
# data['S07']={}
# data['S07']['WalkingOval']={}
# data['S07']['WalkingOval']['pos']={}
# data['S07']['WalkingOval']['conf']={}
# data['S07']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S07']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S07']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S07']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S07',len(kpt1))

# ############## 20191126  ########################
# ################################################
# with open('data/20191126/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(130,185),np.arange(360,440),np.arange(650,700))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# # for ii in range(0,len(row1),10):
# #     plot_16j(kpt1[ii:ii+1])
# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()
# # row=(kpt1[:,0,1]>500)*(kpt1[:,0,1]<1300)
# # kpt1,kpt2=kpt1[row],kpt2[row]
# data['S08']={}
# data['S08']['WalkingOval']={}
# data['S08']['WalkingOval']['pos']={}
# data['S08']['WalkingOval']['conf']={}
# data['S08']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S08']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S08']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S08']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S08',len(kpt1))


# ############## 20191128  ########################
# ################################################
# with open('data/20191128/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(295,345),np.arange(510,570),np.arange(750,815))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# # for ii in range(0,len(row1),10):
# #     plot_16j(kpt1[ii:ii+1])
# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()
# # row=(kpt1[:,0,1]>500)*(kpt1[:,0,1]<1300)
# # kpt1,kpt2=kpt1[row],kpt2[row]
# data['S09']={}
# data['S09']['WalkingOval']={}
# data['S09']['WalkingOval']['pos']={}
# data['S09']['WalkingOval']['conf']={}
# data['S09']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S09']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S09']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S09']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S09',len(kpt1))


# ############## 20191203  ########################
# ################################################
# with open('data/20191203/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(235,300),np.arange(500,580),np.arange(750,830))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# # for ii in range(0,len(row1),10):
# #     plot_16j(kpt1[ii:ii+1])
# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()
# # row=(kpt1[:,0,1]>500)*(kpt1[:,0,1]<1300)
# # kpt1,kpt2=kpt1[row],kpt2[row]
# data['S10']={}
# data['S10']['WalkingOval']={}
# data['S10']['WalkingOval']['pos']={}
# data['S10']['WalkingOval']['conf']={}
# data['S10']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S10']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S10']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S10']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S10',len(kpt1))


# ############## 20191204  ########################
# ################################################
# with open('data/20191204/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(230,290),np.arange(480,540),np.arange(730,810))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# # for ii in range(0,len(row1),10):
# #     plot_16j(kpt1[ii:ii+1])
# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()
# # row=(kpt1[:,0,1]>500)*(kpt1[:,0,1]<1300)
# # kpt1,kpt2=kpt1[row],kpt2[row]
# data['S11']={}
# data['S11']['WalkingOval']={}
# data['S11']['WalkingOval']['pos']={}
# data['S11']['WalkingOval']['conf']={}
# data['S11']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S11']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S11']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S11']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S11',len(kpt1))



# ############## 20200108  ########################
# ################################################
# with open('data/20200108/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(265,330),np.arange(505,570),np.arange(790,840))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# # for ii in range(0,len(row1),10):
# #     plot_16j(kpt1[ii:ii+1])
# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()
# # row=(kpt1[:,0,1]>500)*(kpt1[:,0,1]<1300)
# # kpt1,kpt2=kpt1[row],kpt2[row]
# data['S12']={}
# data['S12']['WalkingOval']={}
# data['S12']['WalkingOval']['pos']={}
# data['S12']['WalkingOval']['conf']={}
# data['S12']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S12']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S12']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S12']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S12',len(kpt1))

# ############## 20200109  ########################
# ################################################
# with open('data/20200109/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# row1=np.concatenate((np.arange(170,210),np.arange(345,380),np.arange(520,565),np.arange(695,740))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])


# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# data['S13']={}
# data['S13']['WalkingOval']={}
# data['S13']['WalkingOval']['pos']={}
# data['S13']['WalkingOval']['conf']={}
# data['S13']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S13']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S13']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S13']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S13',len(kpt1))


# ############## 20200121  ########################
# ################################################
# with open('data/20200121/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# row1=np.concatenate((np.arange(260,370),np.arange(600,700))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# data['S14']={}
# data['S14']['WalkingOval']={}
# data['S14']['WalkingOval']['pos']={}
# data['S14']['WalkingOval']['conf']={}
# data['S14']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S14']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S14']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S14']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S14',len(kpt1))

# ############## 20200122  ########################
# ################################################
# with open('data/20200122/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# row1=np.concatenate((np.arange(160,210),np.arange(390,435),np.arange(620,670),np.arange(840,900))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])


# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# data['S15']={}
# data['S15']['WalkingOval']={}
# data['S15']['WalkingOval']['pos']={}
# data['S15']['WalkingOval']['conf']={}
# data['S15']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S15']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S15']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S15']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S15',len(kpt1))

# ############## 20200123  ########################
# ################################################
# with open('data/20200123/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# row1=np.concatenate((np.arange(180,240),np.arange(430,500),np.arange(710,765))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# data['S16']={}
# data['S16']['WalkingOval']={}
# data['S16']['WalkingOval']['pos']={}
# data['S16']['WalkingOval']['conf']={}
# data['S16']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S16']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S16']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S16']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S16',len(kpt1))

# ############## 20200124  ########################
# ################################################
# with open('data/20200124/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))
 

# row1=np.concatenate((np.arange(170,220),np.arange(400,460),np.arange(650,700))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# data['S17']={}
# data['S17']['WalkingOval']={}
# data['S17']['WalkingOval']['pos']={}
# data['S17']['WalkingOval']['conf']={}
# data['S17']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S17']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S17']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S17']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S17',len(kpt1))

# ############## 20200127  ########################
# ################################################
# with open('data/20200127/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# row1=np.concatenate((np.arange(290,340),np.arange(510,580),np.arange(780,830))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# data['S18']={}
# data['S18']['WalkingOval']={}
# data['S18']['WalkingOval']['pos']={}
# data['S18']['WalkingOval']['conf']={}
# data['S18']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S18']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S18']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S18']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S18',len(kpt1))

# ############## 20200130 ########################
# ################################################
# with open('data/20200130/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))



# row1=np.concatenate((np.arange(290,340),np.arange(500,550),np.arange(710,760))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# data['S19']={}
# data['S19']['WalkingOval']={}
# data['S19']['WalkingOval']['pos']={}
# data['S19']['WalkingOval']['conf']={}
# data['S19']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S19']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S19']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S19']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S19',len(kpt1))

# ############## 20200205 ########################
# ################################################
# with open('data/20200205/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(260,320),np.arange(520,590),np.arange(790,870))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# data['S20']={}
# data['S20']['WalkingOval']={}
# data['S20']['WalkingOval']['pos']={}
# data['S20']['WalkingOval']['conf']={}
# data['S20']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S20']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S20']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S20']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S20',len(kpt1))

# ############## 20200206 ########################
# ################################################
# with open('data/20200206/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(240,290),np.arange(450,495),np.arange(640,680),np.arange(820,870))) #,np.arange(2900,3130),

# # print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# data['S21']={}
# data['S21']['WalkingOval']={}
# data['S21']['WalkingOval']['pos']={}
# data['S21']['WalkingOval']['conf']={}
# data['S21']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S21']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S21']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S21']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S21',len(kpt1))

# ############## 20200207 ########################
# ################################################
# with open('data/20200207/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))



# row1=np.concatenate((np.arange(230,280),np.arange(410,460),np.arange(760,810))) #,np.arange(2900,3130),


# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# data['S22']={}
# data['S22']['WalkingOval']={}
# data['S22']['WalkingOval']['pos']={}
# data['S22']['WalkingOval']['conf']={}
# data['S22']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S22']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S22']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S22']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S22',len(kpt1))

# ############## 20200213 ########################
# ################################################
# with open('data/20200213/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))



# row1=np.concatenate((np.arange(210,280),np.arange(480,550),np.arange(780,850))) #,np.arange(2900,3130),


# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()
# data['S23']={}
# data['S23']['WalkingOval']={}
# data['S23']['WalkingOval']['pos']={}
# data['S23']['WalkingOval']['conf']={}
# data['S23']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S23']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S23']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S23']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S23',len(kpt1))

# ############## 20200214 ########################
# ################################################
# with open('data/20200214/WalkingOval/CH2/alphapose-results.json') as f:
#     data1=json.load(f)
#     kpt1=process_data(data1,channel=2,camera='old',len_data=4000)
#     # print(len(kpt1))


# # 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# row1=np.concatenate((np.arange(220,270),np.arange(420,470),np.arange(620,690))) #,np.arange(2900,3130),

# print(row1.shape)

# kpt1=kpt1[row1]
# row=(kpt1[:,0,1]>100)
# kpt1=kpt1[row]
# # plot_16j(kpt1[0:1])

# plt.plot(kpt1[:,0])
# # plt.plot(kpt2[:,0])
# plt.show()

# data['S24']={}
# data['S24']['WalkingOval']={}
# data['S24']['WalkingOval']['pos']={}
# data['S24']['WalkingOval']['conf']={}
# data['S24']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
# data['S24']['WalkingOval']['pos'][1]=kpt1[:,:,:2]
# data['S24']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
# data['S24']['WalkingOval']['conf'][1]=kpt1[:,:,2:]
# print('S24',len(kpt1))

############# 20200218 ########################
###############################################
with open('data/20200218/WalkingOval/CH1/alphapose-results.json') as f:
    data1=json.load(f)
    kpt1=process_data(data1,channel=1,camera='old',len_data=4000)
    # print(len(kpt1))

with open('data/20200218/WalkingOval/CH2/alphapose-results.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='old',len_data=4000)
    # print(len(kpt2))

# 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

# ## for train
# row1=np.concatenate((np.arange(160,260),np.arange(360,460),np.arange(560,670),np.arange(770,860))) #,np.arange(2900,3130),
# row2=np.concatenate((np.arange(154,254),np.arange(354,454),np.arange(554,664),np.arange(764,854))) #,np.arange(2900,3130),
## for test
row1=np.concatenate((np.arange(200,260),np.arange(400,460),np.arange(610,670),np.arange(800,860))) #,np.arange(2900,3130),
row2=np.concatenate((np.arange(194,254),np.arange(394,454),np.arange(604,664),np.arange(794,854))) #,np.arange(2900,3130),

print(row1.shape)

kpt1=kpt1[row1]
kpt2=kpt2[row2]
row=(kpt1[:,0,0]>10)
kpt1,kpt2=kpt1[row],kpt2[row]

# plot_16j(kpt1[:])
# plot_16j(kpt2[:])

plt.plot(kpt1[:,0])
plt.plot(kpt2[:,0])
plt.show()
#  
data['S25']={}
data['S25']['WalkingOval']={}
data['S25']['WalkingOval']['pos']={}
data['S25']['WalkingOval']['conf']={}
data['S25']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
data['S25']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S25']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
data['S25']['WalkingOval']['conf'][1]=kpt2[:,:,2:]

np.savez_compressed('data_PD',positions_2d=data)
print('S25',len(kpt1))

############## Test4 ########################
################################################
with open('data/Test4/WalkingOval/CH1/alphapose-results.json') as f:
    data1=json.load(f)
    kpt1=process_data(data1,channel=1,camera='old',len_data=4000)
    # print(len(kpt1))

with open('data/Test4/WalkingOval/CH2/alphapose-results.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='old',len_data=4000)
    # print(len(kpt2))

# 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790


## for training
# row1=np.concatenate((np.arange(200,310),np.arange(400,510),np.arange(600,710),np.arange(800,900),np.arange(1000,1100),np.arange(1200,1300))) #,np.arange(2900,3130),
# row2=np.concatenate((np.arange(196,306),np.arange(396,506),np.arange(596,706),np.arange(799,899),np.arange(999,1099),np.arange(1199,1299))) #,np.arange(2900,3130),
## for test
row1=np.concatenate((np.arange(250,310),np.arange(450,510),np.arange(650,710))) #,np.arange(2900,3130),
row2=np.concatenate((np.arange(246,306),np.arange(446,506),np.arange(646,706))) #,np.arange(2900,3130),

print(row1.shape)

kpt1=kpt1[row1]
kpt2=kpt2[row2]
row=(kpt1[:,0,1]>10)
kpt1,kpt2=kpt1[row],kpt2[row]
# plot_16j(kpt1[:])
# plot_16j(kpt2[:])
# for ii in range(0,len(row1),10):
#     plot_16j(kpt1[ii:ii+1])
plt.plot(kpt1[:,0])
plt.plot(kpt2[:,0])
plt.show()  
# row=(kpt1[:,0,1]>500)*(kpt1[:,0,1]<1300)
# kpt1,kpt2=kpt1[row],kpt2[row]
data['S26']={}
data['S26']['WalkingOval']={}
data['S26']['WalkingOval']['pos']={}
data['S26']['WalkingOval']['conf']={}
data['S26']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
data['S26']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S26']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
data['S26']['WalkingOval']['conf'][1]=kpt2[:,:,2:]

np.savez_compressed('data_PD',positions_2d=data)
print('S26',len(kpt1))


############## 20200221 ########################
################################################
with open('data/20200221/WalkingOval/CH1/alphapose-results.json') as f:
    data1=json.load(f)
    kpt1=process_data(data1,channel=1,camera='old',len_data=4000)
    # print(len(kpt1))

with open('data/20200221/WalkingOval/CH2/alphapose-results.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='old',len_data=4000)
    # print(len(kpt2))

# 650:960,1130:1375,1545:1805,1970:2430,2596:2870,3050:3320,3500:3790

## for training
# row1=np.concatenate((np.arange(190,290),np.arange(390,505),np.arange(610,730),np.arange(840,950),np.arange(1060,1160)
# ,np.arange(1280,1390),np.arange(1500,1620),np.arange(1740,1850))) #,np.arange(2900,3130),
# row2=np.concatenate((np.arange(185,285),np.arange(385,500),np.arange(605,725),np.arange(835,945),np.arange(1055,1155)
# ,np.arange(1275,1385),np.arange(1495,1615),np.arange(1735,1845)))#,np.arange(2900,3130),

## for testing
row1=np.concatenate((np.arange(230,290),np.arange(445,505),np.arange(670,730))) #,np.arange(2900,3130),
row2=np.concatenate((np.arange(225,285),np.arange(440,500),np.arange(665,725)))#,np.arange(2900,3130),

print(row1.shape)

kpt1=kpt1[row1]
kpt2=kpt2[row2]

row=(kpt1[:,0,1]>100)*(kpt2[:,0,1]>100)
kpt1,kpt2=kpt1[row],kpt2[row]
# plot_16j(kpt1[:])
# plot_16j(kpt2[:])
# for ii in range(0,len(row1),10):
#     plot_16j(kpt1[ii:ii+1])
plt.plot(kpt1[:,0])
plt.plot(kpt2[:,0])
plt.show()
# row=(kpt1[:,0,1]>500)*(kpt1[:,0,1]<1300)
# kpt1,kpt2=kpt1[row],kpt2[row]
data['S27']={}
data['S27']['WalkingOval']={}
data['S27']['WalkingOval']['pos']={}
data['S27']['WalkingOval']['conf']={}
data['S27']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
data['S27']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S27']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
data['S27']['WalkingOval']['conf'][1]=kpt2[:,:,2:]
print('S27',len(kpt2))

############## 20210706 ########################
################################################
with open('data/20210706/WalkingOval/CH1/alphapose-results.json') as f:
    data1=json.load(f)
    kpt1=process_data(data1,channel=1,camera='new',len_data=4000)
    # print(len(kpt1))

with open('data/20210706/WalkingOval/CH2/alphapose-results.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='new',len_data=4000)
    # print(len(kpt2))

# ## for training
# row=np.concatenate((np.arange(740,960),np.arange(1150,1400),np.arange(1590,1900),np.arange(2100,2400),np.arange(2600,2870),np.arange(3080,3390)
# ,np.arange(3590,3850))) #,np.arange(2900,3130),

## for testing
row=np.concatenate((np.arange(840,960),np.arange(1280,1400),np.arange(1780,1900),np.arange(2280,2400))) #,np.arange(2900,3130),

print(row.shape)

kpt1=kpt1[row]
kpt2=kpt2[row]
row=(kpt1[:,0,1]>100)*(kpt2[:,0,1]>100)
kpt1,kpt2=kpt1[row],kpt2[row]
# plot_16j(kpt1[:])
# plot_16j(kpt2[:])

plt.plot(kpt1[:,0])
plt.plot(kpt2[:,0])
plt.show()

data['S28']={}
data['S28']['WalkingOval']={}
data['S28']['WalkingOval']['pos']={}
data['S28']['WalkingOval']['conf']={}
data['S28']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
data['S28']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S28']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
data['S28']['WalkingOval']['conf'][1]=kpt2[:,:,2:]

print('S28',len(kpt2))
############## 20210804 ########################
################################################
with open('data/20210804/WalkingOval/CH1/alphapose-results.json') as f:
    data1=json.load(f)
    kpt1=process_data(data1,channel=1,camera='new',len_data=4000)
    # print(len(kpt1))

with open('data/20210804/WalkingOval/CH2/alphapose-results.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='new',len_data=4000)
    # print(len(kpt2))


## for training
# row=np.concatenate((np.arange(290,510),np.arange(690,900),np.arange(1070,1280),np.arange(1430,1650))) #,np.arange(2900,3130),
## for test
row=np.concatenate((np.arange(390,510),np.arange(780,900),np.arange(1160,1280))) #,np.arange(2900,3130),


print(row.shape)

kpt1=kpt1[row]
kpt2=kpt2[row]
row=(kpt1[:,0,1]>100)*(kpt2[:,0,1]>100)
kpt1,kpt2=kpt1[row],kpt2[row]
# plot_16j(kpt1[:])
# plot_16j(kpt2[:])

plt.plot(kpt1[:,0])
plt.plot(kpt2[:,0])
plt.show()

data['S29']={}
data['S29']['WalkingOval']={}
data['S29']['WalkingOval']['pos']={}
data['S29']['WalkingOval']['conf']={}
data['S29']['WalkingOval']['pos'][0]=kpt1[:,:,:2]
data['S29']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S29']['WalkingOval']['conf'][0]=kpt1[:,:,2:]
data['S29']['WalkingOval']['conf'][1]=kpt2[:,:,2:]

print('S29',len(kpt2))

############## 20200206_2 ########################
################################################

with open('data/20200206_2/WalkingOval/CH2/alphapose-results.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='old',len_data=4000)
    # print(len(kpt2))

row=np.concatenate((np.arange(220,270),np.arange(420,490),np.arange(650,695),np.arange(840,880))) #,np.arange(2900,3130),

print(row.shape)

kpt2=kpt2[row]
row=(kpt2[:,0,1]>100)
kpt2=kpt2[row]

# plot_16j(kpt2[:])
plt.plot(kpt2[:,0])
plt.show()

data['S30']={}
data['S30']['WalkingOval']={}
data['S30']['WalkingOval']['pos']={}
data['S30']['WalkingOval']['conf']={}
data['S30']['WalkingOval']['pos'][0]=kpt2[:,:,:2]
data['S30']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S30']['WalkingOval']['conf'][0]=kpt2[:,:,2:]
data['S30']['WalkingOval']['conf'][1]=kpt2[:,:,2:]
print('S30',len(kpt2))
############## 20210811 ########################
################################################

with open('data/20210811/WalkingOval/CH2/alphapose-results.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='new',len_data=4000)
    # print(len(kpt2))

row=np.concatenate((np.arange(1000,1190),np.arange(1570,1780))) #,np.arange(2900,3130),

print(row.shape)

kpt2=kpt2[row]
row=(kpt2[:,0,1]>100)
kpt2=kpt2[row]

# plot_16j(kpt2[:])
plt.plot(kpt2[:,0])
plt.show()

data['S31']={}
data['S31']['WalkingOval']={}
data['S31']['WalkingOval']['pos']={}
data['S31']['WalkingOval']['conf']={}
data['S31']['WalkingOval']['pos'][0]=kpt2[:,:,:2]
data['S31']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S31']['WalkingOval']['conf'][0]=kpt2[:,:,2:]
data['S31']['WalkingOval']['conf'][1]=kpt2[:,:,2:]

print('S31',len(kpt2))
############## 20191210 ########################
################################################

with open('data/20191210/WalkingOval/CH2/alphapose-results.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='old',len_data=4000)
    # print(len(kpt2))

row=np.concatenate((np.arange(230,270),np.arange(400,450),np.arange(590,640))) #,np.arange(2900,3130),

print(row.shape)

kpt2=kpt2[row]
row=(kpt2[:,0,1]>100)
kpt2=kpt2[row]

# plot_16j(kpt2[:])
plt.plot(kpt2[:,0])
plt.show()

data['S32']={}
data['S32']['WalkingOval']={}
data['S32']['WalkingOval']['pos']={}
data['S32']['WalkingOval']['conf']={}
data['S32']['WalkingOval']['pos'][0]=kpt2[:,:,:2]
data['S32']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S32']['WalkingOval']['conf'][0]=kpt2[:,:,2:]
data['S32']['WalkingOval']['conf'][1]=kpt2[:,:,2:]
print('S32',len(kpt2))
############## 20191212 ########################
################################################

with open('data/20191212/WalkingOval/CH2/alphapose-results.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='old',len_data=4000)
    # print(len(kpt2))

row=np.concatenate((np.arange(500,560),np.arange(700,770))) #,np.arange(2900,3130),

print(row.shape)

kpt2=kpt2[row]
row=(kpt2[:,0,1]>100)
kpt2=kpt2[row]

# plot_16j(kpt2[:])
plt.plot(kpt2[:,0])
plt.show()

data['S33']={}
data['S33']['WalkingOval']={}
data['S33']['WalkingOval']['pos']={}
data['S33']['WalkingOval']['conf']={}
data['S33']['WalkingOval']['pos'][0]=kpt2[:,:,:2]
data['S33']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S33']['WalkingOval']['conf'][0]=kpt2[:,:,2:]
data['S33']['WalkingOval']['conf'][1]=kpt2[:,:,2:]

print('S33',len(kpt2))
############## 20191218 ########################
################################################

with open('data/20191218/WalkingOval/CH2/alphapose-results.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='old',len_data=4000)
    # print(len(kpt2))

row=np.concatenate((np.arange(210,270),np.arange(430,490),np.arange(650,710))) #,np.arange(2900,3130),

print(row.shape)

kpt2=kpt2[row]
row=(kpt2[:,0,1]>100)
kpt2=kpt2[row]

# plot_16j(kpt2[0:1])
plt.plot(kpt2[:,0])
plt.show()

data['S34']={}
data['S34']['WalkingOval']={}
data['S34']['WalkingOval']['pos']={}
data['S34']['WalkingOval']['conf']={}
data['S34']['WalkingOval']['pos'][0]=kpt2[:,:,:2]
data['S34']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S34']['WalkingOval']['conf'][0]=kpt2[:,:,2:]
data['S34']['WalkingOval']['conf'][1]=kpt2[:,:,2:]
print('S34',len(kpt2))
############## 20200227 ########################
################################################

with open('data/20200227/WalkingOval/CH2/alphapose-results.json') as f:
    data2=json.load(f)
    kpt2=process_data(data2,channel=2,camera='old',len_data=4000)
    # print(len(kpt2))

row=np.concatenate((np.arange(210,260),np.arange(430,490),np.arange(690,740))) #,np.arange(2900,3130),

print(row.shape)

kpt2=kpt2[row]
row=(kpt2[:,0,1]>100)
kpt2=kpt2[row]

# plot_16j(kpt2[0:1])
plt.plot(kpt2[:,0])
plt.show()

data['S35']={}
data['S35']['WalkingOval']={}
data['S35']['WalkingOval']['pos']={}
data['S35']['WalkingOval']['conf']={}
data['S35']['WalkingOval']['pos'][0]=kpt2[:,:,:2]
data['S35']['WalkingOval']['pos'][1]=kpt2[:,:,:2]
data['S35']['WalkingOval']['conf'][0]=kpt2[:,:,2:]
data['S35']['WalkingOval']['conf'][1]=kpt2[:,:,2:]

print('S35',len(kpt2))

np.savez_compressed('data_PD_test',positions_2d=data)
