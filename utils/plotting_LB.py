import matplotlib.pyplot as plt
import numpy as np
a=[44,65,72,72,65,82,79,82,82]
# a=[55,69,72,72,62,72,72,72,72,72,82,82,82,82,82]
a=[55,69,72,72,62,72,72,72,72,72,82,72,72,79,79]
x1=[1,2,3,4,5,6,7 , 8, 9,10,11,12,13,14,15]
# b= [65,65,72,72,76,76,89,89,86,86,86,86,86]
b= [65,65,72,72,76,76,89,89,89,89,89,89,89]
x2=[3 ,4 ,5 ,6 ,7 , 8, 9,10,11,12,13,14,15]
fig = plt.figure(figsize = (10, 7))
plt.plot(x1,a,'navy')
plt.plot(x2,b,'crimson')
plt.legend(['Voting with labeling functions','Denoising Network'])
plt.ylabel('Accuracy(%)')
plt.xlabel('Labeling Functions')
plt.show()

fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
plt.rcParams['font.size'] = 15
plt.rc('axes', labelsize=15) 
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15) 
# X = np.arange(10)
# langs = ['Ours','Arm Swing Sym', 'Step Length', 'Foot Clearance', 'Hip Flex', 'Knee Flex', 'Trunk Rot','Step Width','Cadence','Arm Swing']
# Pre=[87,82,72,87,87,87,83,83,79,90]
# Rec=[85,82,73,85,85,85,82,82,79,89]
# F1= [86,83,72,86,86,86,82,82,79,89]
# acc=[86,82,72,86,86,86,82,82,79,89]
# y=np.ones(12)*86
# x=np.arange(-1,11)
# plt.plot(x,y,'r--')
# plt.bar(X-0.24,Pre,width=0.12,label='Precision')
# plt.bar(X-0.12,Rec,width=0.12,label='Recall')
# plt.bar(X+0,F1,width=0.12,label='F1')
# plt.bar(X+0.12,acc,width=0.12,label='Accuracy')
# plt.ylim([60,95])
# barWidth=0.12
# plt.xticks([r + barWidth for r in range(10)],
#         langs,rotation=60)
# plt.ylabel('Accuracy(%)')
# plt.legend(prop={"size":12})
# plt.show()

X = np.arange(20)
langs = ['Ours','ASS','RHM','LHM', 'RSL', 'LSL', 'RFC', 'LFC','FC', 'RHF','LHF','HF','RKF', 'LKF','KF','RTR','LTR','TR','SW','Cad']
acc=[89,82,89,89,82,86,89,89,86,89,89,86,89,89,86,89,89,86,82,79]
# acc=[89,82,89,89,82,86,89,89,86,89,89,86,89,89,86,89,89,86,82,79]
y=np.ones(19)*89
x=np.arange(-1,18)
# plt.plot(x,y,'r--')
plt.bar(X,acc,width=0.50,label='Accuracy',color='navy')
plt.ylim([60,95])
barWidth=0.12
plt.xticks([r for r in range(20)],
        langs)
plt.ylabel('Accuracy(%)')
plt.legend(prop={"size":12})
plt.show()

# 0 accuracy 0.8275862068965517
# Pre Rec F1: (0.8261904761904761, 0.8293269230769231, 0.8267622461170848, None)
# 1 accuracy 0.8275862068965517
# Pre Rec F1: (0.8261904761904761, 0.8293269230769231, 0.8267622461170848, None)
# 2 accuracy 0.8620689655172413
# Pre Rec F1: (0.8642857142857143, 0.8677884615384616, 0.8619047619047618, None)
# 3 accuracy 0.896551724137931
# Pre Rec F1: (0.8995098039215685, 0.8918269230769231, 0.8945454545454545, None)
# 4 accuracy 0.896551724137931
# Pre Rec F1: (0.8995098039215685, 0.8918269230769231, 0.8945454545454545, None)
# 5 accuracy 0.896551724137931
# Pre Rec F1: (0.8995098039215685, 0.8918269230769231, 0.8945454545454545, None)
# 6 accuracy 0.896551724137931
# Pre Rec F1: (0.8995098039215685, 0.8918269230769231, 0.8945454545454545, None)
# 7 accuracy 0.896551724137931
# Pre Rec F1: (0.8995098039215685, 0.8918269230769231, 0.8945454545454545, None)
# 8 accuracy 0.896551724137931
# Pre Rec F1: (0.8995098039215685, 0.8918269230769231, 0.8945454545454545, None)
# 9 accuracy 0.896551724137931
# Pre Rec F1: (0.8995098039215685, 0.8918269230769231, 0.8945454545454545, None)
# 10 accuracy 0.896551724137931
# Pre Rec F1: (0.8995098039215685, 0.8918269230769231, 0.8945454545454545, None)
# 11 accuracy 0.8275862068965517
# Pre Rec F1: (0.8261904761904761, 0.8293269230769231, 0.8267622461170848, None)
# 12 accuracy 0.7931034482758621
# Pre Rec F1: (0.7952380952380952, 0.7980769230769231, 0.7928571428571427, None)
# 13 accuracy 0.896551724137931
# Pre Rec F1: (0.8995098039215685, 0.8918269230769231, 0.8945454545454545, None)
# 14 accuracy 0.896551724137931
# Pre Rec F1: (0.8995098039215685, 0.8918269230769231, 0.8945454545454545, None)
# 15 accuracy 0.896551724137931
# Pre Rec F1: (0.8995098039215685, 0.8918269230769231, 0.8945454545454545, None)