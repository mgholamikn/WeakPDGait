# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import hashlib

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value


import numpy as np


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    Reimplementation of MATLAB's `procrustes` function to Numpy.
    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def cam_loss(matrix_k):
    # loss function to enforce a weak perspective camera as described in the paper
    batch_size, sequence_len, input_size= matrix_k.shape
    loss=0
    for ii in range(sequence_len):
        m = torch.reshape(matrix_k[:,ii,:], [-1, 2, 3])
        # print(m.shape)

        m_sq = torch.matmul(m, torch.transpose(m, 1, 2))
        B=torch.diagonal(m_sq, dim1=-2, dim2=-1).sum(-1) 
        A=2 / B
        loss_mat = torch.reshape(A, [-1, 1, 1])*m_sq - torch.eye(2).cuda()

        loss += torch.sum(torch.abs(loss_mat))/batch_size
        
    return loss/sequence_len


def reprojection_layer(K, pose3):
    
    # x = torch.from_numpy(x.astype('float32'))
    # pose3 = torch.reshape(tf.slice(x, [0, 0], [-1, 48]), [-1, 3, 16])
    
    #m = tf.reshape(tf.slice(x, [0, 48], [-1, 6]), [-1, 2, 3])
    ########################################################################################
    batch_size, sequence_len, input_size= K.shape
    K=torch.reshape(K,(batch_size,sequence_len,2,3))
    pose3=torch.transpose(pose3,2,3)
    pose2_rec = torch.reshape(torch.matmul(K, pose3), [-1, sequence_len , 17,2])

    return pose2_rec


def plot17j(poses, show_animation=False):
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.animation as anim

    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure()

    if not show_animation:
        plot_idx = 1
# 
        frames = np.linspace(start=0, stop=poses.shape[0]-1, num=10).astype(int)
        
        for i in frames:
            ax = fig.add_subplot(2, 5, plot_idx, projection='3d')

            # pose = poses[i,1:]
            # pose = pose[[2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 12, 11, 10], :]
            # pose=np.reshape(pose,(16*3,))
            
            # x = pose[0:16]
            # y = pose[16:32]
            # z = pose[32:48]
            x = poses[i,:,0]
            y = poses[i,:,1]
            z = poses[i,:,2]
            ax.scatter(x, y, z)

            ax.scatter(poses[i,15,0],poses[i,15,1],poses[i,15,2],'r', s=60)
            # ax.scatter(poses[i,1,0],poses[i,1,1],poses[i,1,2],'b', s=60)
            ax.scatter(poses[i,16,0],poses[i,16,1],poses[i,16,2],'g', s=60)
            # ax.scatter(poses[i,3,0],poses[i,3,1],poses[i,3,2],'c', s=60)
            # ax.scatter(poses[i,4,0],poses[i,4,1],poses[i,4,2],'m', s=60)
            # ax.scatter(poses[i,5,0],poses[i,5,1],poses[i,5,2],'y', s=60)
            # ax.scatter(poses[i,8,0],poses[i,8,1],poses[i,8,2],'black', s=60)

            ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])])
            ax.plot(x[([0, 4])], y[([0, 4])], z[([0, 4])])
            ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])])
            ax.plot(x[([2, 3])], y[([2, 3])], z[([2, 3])])
            ax.plot(x[([5, 6])], y[([5, 6])], z[([5, 6])])
            ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])])
            ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])])
            ax.plot(x[([8, 7])], y[([8, 7])], z[([8,7])])
            ax.plot(x[([0, 7])], y[([0, 7])], z[([0,7])])            
            ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])])
            ax.plot(x[([10, 9])], y[([10, 9])], z[([10, 9])])
            ax.plot(x[([11, 8])], y[([11, 8])], z[([11, 8])])
            ax.plot(x[([8, 14])], y[([8, 14])], z[([8, 14])])
            ax.plot(x[([15, 14])], y[([15, 14])], z[([15, 14])])
            ax.plot(x[([16, 15])], y[([16, 15])], z[([16, 15])])
            ax.plot(x[([13, 12])], y[([13, 12])], z[([13, 12])])
            ax.plot(x[([12, 11])], y[([12, 11])], z[([12, 11])])

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

            # ax.axis('equal')
            ax.axis('auto')
            ax.axis('off')

            ax.set_title('frame = ' + str(i))

            plot_idx += 1

        # this uses QT5Agg backend
        # you can identify the backend using plt.get_backend()
        # delete the following two lines and resize manually if it throws an error
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

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