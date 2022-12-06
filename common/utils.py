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

def procrustes_batch(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.

    """

    # assert predicted.shape == target.shape
    
    # muX = torch.mean(target, dim=1, keepdims=True)
    # muY = torch.mean(predicted, dim=1, keepdims=True)
    
    # X0 = target - muX
    # Y0 = predicted - muY

    # normX = torch.sqrt(torch.sum(X0**2, dim=(2, 3), keepdims=True))
    # normY = torch.sqrt(torch.sum(Y0**2, dim=(2, 3), keepdims=True))
    
    # X0 /= normX
    # Y0 /= normY

    # H = torch.matmul(X0.transpose(3, 2), Y0)
    # U, s, Vt =torch.svd(H).detach()
    # V = Vt.transpose(3, 2)
    # R = torch.matmul(V, U.transpose(3, 2))

    # # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    # sign_detR = torch.sign(torch.unsqueeze(torch.det(R), dim=1))
    # V[:, :, -1] *= sign_detR
    # s[:, -1] *= sign_detR.flatten()
    # R = torch.matmul(V, U.transpose(3, 2)) # Rotation

    # tr = torch.unsqueeze(torch.sum(s, dim=1, keepdims=True), dim=2)

    # a = tr * normX / normY # Scale
    # t = muX - a*torch.matmul(muY, R) # Translation
    
    # # Perform rigid transformation on the input
    # predicted_aligned = a*torch.matmul(predicted, R) + t

    
    muX = np.mean(target, axis=2, keepdims=True)
    muY = np.mean(predicted, axis=2, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(2, 3), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(2, 3), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 1, 3, 2), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 1,3, 2)
    R = np.matmul(V, U.transpose(0, 1, 3, 2))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=2))
    V[:, :, :, -1] *= sign_detR
    s[:, :, -1] *= sign_detR[:,:,0]
    R = np.matmul(V, U.transpose(0, 1, 3, 2)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    return R,a,t,predicted_aligned

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

    if reflection != 'best':

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
    ax = fig.add_subplot(111, projection='3d')
    if not show_animation:
        plot_idx = 1
      
        frames = np.linspace(start=0, stop=poses.shape[0]-1, num=poses.shape[0]).astype(int)
 
        for i in frames:
            # ax = fig.add_subplot(1, poses.shape[0], plot_idx, projection='3d')
            
            
            # pose = poses[1,[0,1],:]=0
            # pose = pose[[2, 1, 0, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 12, 11, 10], :]
            # pose=np.reshape(pose,(16*3,))
            
            # x = pose[0:16]
            # y = pose[16:32]
            # z = pose[32:48]
            x = poses[i,:,0]
            y = poses[i,:,1]
            z = poses[i,:,2]
            
            # ax.scatter(y, x, z)

            # kkkkk=15

            # ax.scatter(poses[i,kkkkk,0],poses[i,kkkkk,1],poses[i,kkkkk,2],'r')


            ax.plot(poses[i,[0,1],0], poses[i,[0,1],1], poses[i,[0,1],2])
            ax.plot(poses[i,[1,2],0], poses[i,[1,2],1], poses[i,[1,2],2])
            ax.plot(poses[i,[2,3],0], poses[i,[2,3],1], poses[i,[2,3],2])
            ax.plot(poses[i,[0,4],0], poses[i,[0,4],1], poses[i,[0,4],2])
            ax.plot(poses[i,[4,5],0], poses[i,[4,5],1], poses[i,[4,5],2])
            ax.plot(poses[i,[5,6],0], poses[i,[5,6],1], poses[i,[5,6],2])
            ax.plot(poses[i,[0,7],0], poses[i,[0,7],1], poses[i,[0,7],2])
            ax.plot(poses[i,[7,8],0], poses[i,[7,8],1], poses[i,[7,8],2])
            ax.plot(poses[i,[8,9],0], poses[i,[8,9],1], poses[i,[8,9],2])
            ax.plot(poses[i,[9,10],0], poses[i,[9,10],1], poses[i,[9,10],2])
            ax.plot(poses[i,[8,11],0], poses[i,[8,11],1], poses[i,[8,11],2])
            ax.plot(poses[i,[11,12],0], poses[i,[11,12],1], poses[i,[11,12],2])
            ax.plot(poses[i,[12,13],0], poses[i,[12,13],1], poses[i,[12,13],2])
            ax.plot(poses[i,[8,14],0], poses[i,[8,14],1], poses[i,[8,14],2])
            ax.plot(poses[i,[14,15],0], poses[i,[14,15],1], poses[i,[14,15],2])
            ax.plot(poses[i,[15,16],0], poses[i,[15,16],1], poses[i,[15,16],2])
            # ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])])
            # ax.plot(x[([3, 4])], y[([3, 4])], z[([3, 4])])
            # ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])])
            # ax.plot(x[([0, 6])], y[([0, 6])], z[([0, 6])])
            # ax.plot(x[([3, 6])], y[([3, 6])], z[([3, 6])])
            # ax.plot(x[([6, 7])], y[([6, 7])], z[([6, 7])])
            # ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])])
            # ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])])
            # ax.plot(x[([7, 10])], y[([7, 10])], z[([7, 10])])
            # ax.plot(x[([10, 11])], y[([10, 11])], z[([10, 11])])
            # ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])])
            # ax.plot(x[([7, 13])], y[([7, 13])], z[([7, 13])])
            # ax.plot(x[([13, 14])], y[([13, 14])], z[([13, 14])])
            # ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])])

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

            # ax.axis('equal')
            ax.axis('auto')
            ax.axis('on')
            ax.set_title('frame = ' +str(i))
            # ax.view_init(elev=10, azim=90)
            plot_idx += 1

            # ax.axes.xaxis.set_ticks([])
            # ax.axes.yaxis.set_ticks([])
            # ax.axes.zaxis.set_ticks([])
            
            
        # this uses QT5Agg backend
        # you can identify the backend using plt.get_backend()
        # delete the following two lines and resize manually if it throws an error
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.grid(True,'r')
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
            # ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])])
            # ax.plot(x[([3, 4])], y[([3, 4])], z[([3, 4])])
            # ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])])
            # ax.plot(x[([0, 6])], y[([0, 6])], z[([0, 6])])
            # ax.plot(x[([3, 6])], y[([3, 6])], z[([3, 6])])
            # ax.plot(x[([6, 7])], y[([6, 7])], z[([6, 7])])
            # ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])])
            # ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])])
            # ax.plot(x[([7, 10])], y[([7, 10])], z[([7, 10])])
            # ax.plot(x[([10, 11])], y[([10, 11])], z[([10, 11])])
            # ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])])
            # ax.plot(x[([7, 13])], y[([7, 13])], z[([7, 13])])
            # ax.plot(x[([13, 14])], y[([13, 14])], z[([13, 14])])
            # ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])])

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



def gradient_penalty(real_data, generated_data, gp_weight):
    
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(real_data)
    if torch.cuda.is_available():
        alpha = alpha.cuda()
    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated = Variable(interpolated, requires_grad=True)
    if torch.cuda.is_available():
        interpolated = interpolated.cuda()
    
    # Calculate probability of interpolated examples
    prob_interpolated = model_discriminator(interpolated)
    
    # Calculate gradients of probabilities with respect to examples
    with torch.backends.cudnn.flags(enabled=False):
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                                grad_outputs=torch.ones(prob_interpolated.size()).cuda() if torch.cuda.is_available() else torch.ones(
                                prob_interpolated.size()),
                                create_graph=True, retain_graph=True)[0]
    
    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    
    gradients=gradients.contiguous()
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()


def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu