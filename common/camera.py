# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch

from common.utils import wrap
from common.quaternion import qrot, qinverse

def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

def normalize_screen_coordinates_batch(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - torch.tensor([1, h/w]).cuda()

    
def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    
    # Reverse camera frame normalization
    return (X + [1, h/w])*w/2
    

def world_to_camera_batch(X, cam):
    R = cam[..., 0:4]
    t = cam[..., 4:7]
    Rt = qinverse(R) # Invert rotation
    Rt=torch.unsqueeze(torch.unsqueeze(Rt,1),1)
    Rt=Rt.repeat(1,X.shape[-3],X.shape[-2],1)
    t=torch.unsqueeze(torch.unsqueeze(t,1),1)
    t=t.repeat(1,X.shape[-3],X.shape[-2],1)
    return qrot(Rt, X - t) # Rotate and translate

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate
    
def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

def camera_to_world_batch(X, cam):
    R = cam[..., 0:4]
    t = cam[..., 4:7] 
    R=torch.unsqueeze(torch.unsqueeze(R,1),1)
    R=R.repeat(1,X.shape[-3],X.shape[-2],1)
    t=torch.unsqueeze(torch.unsqueeze(t,1),1)
    t=t.repeat(1,X.shape[-3],X.shape[-2],1)
    return qrot(R , X) + t

def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9) + extrinsic parametres (N, (4+3)*5=7*5) >> (N, 9+35=44)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:9]
    
    XX = torch.clamp(X[..., :2] / (X[..., 2:]), min=-1, max=1)

    if torch.sum(torch.isnan(XX))>0:
        print('Detected NAN')
        XX = torch.clamp(X[..., :2] / (X[..., 2:]), min=-1, max=1) 

    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2
    
    return f*XXX + c

def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)     
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
  

    return f*XX + c

def project_to_3d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 2
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = (X[..., :2] - c)*5/f

    return XX

def project_3d_points_to_image_plane_without_distortion(proj_matrix, points_3d, convert_back_to_euclidean=True):
    """Project 3D points to image plane not taking into account distortion
    Args:
        proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
        points_3d numpy array or torch tensor of shape (N, 3): 3D points
        convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                        NOTE: division by zero can be here if z = 0
    Returns:
        numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
    """
    print('1')
    if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.T
        # result = proj_matrix @ euclidean_to_homogeneous(points_3d).T
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")

def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean
    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M
    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")

def euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous
    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M
    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def project_point_radial(R,T,f,c,k,p, P):
    """
    Project points from 3d to 2d using camera parameters
    including radial and tangential distortion
    Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
    """

    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3
    f=f[0]
    # print(f)
    c=np.expand_dims(c,axis=1)
    k=np.expand_dims(k,axis=1)
    p=np.expand_dims(p,axis=1)

    N = P.shape[0]
    # R=R.T
    X = R.dot( P.T) + T  # rotate and translate
    XX = X[:2,:] / X[2,:]
    r2 = XX[0,:]**2 + XX[1,:]**2

    radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) )
    tan = p[0]*XX[1,:] + p[1]*XX[0,:]

    XXX = XX * np.tile(radial+tan,(2,1)) + \
            np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )

    Proj = (f * XXX) + c
    Proj = Proj.T

    D = X[2,]

    return Proj, D, radial, tan, r2


def world_to_camera_frame(P, R, T):
    """
    Convert points from world to camera coordinates
    Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    Returns
    X_cam: Nx3 3d points in camera coordinates
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3
    
    # X_cam = R.dot( P.T  T ) # rotate and translate
    X_cam = R.dot( P.T) + T  # rotate and translate

    return X_cam.T

def camera_to_world_frame(P, R, T):
  """Inverse of world_to_camera_frame
  Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 points in world coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.T.dot( P.T - T )# rotate and translate

  return X_cam.T

def Projection_Matrix(R,T,f,c):
    T=np.expand_dims(T,axis=1)
    # T=np.matmul(np.array(R), np.negative(T))
    P = np.concatenate((R,T),axis=1)
    K=[[f[0],0,c[0]],[0,f[1],c[1]],[0,0,1]]
    P=np.matmul(K, P)
    return P

def world_to_camera_batch2(P, Camera):
    """
    Convert points from world to camera coordinates
    Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    Returns
    X_cam: Nx3 3d points in camera coordinates
    """

    assert P.shape[-1] == 3
    assert Camera.shape[-1] == 9
    assert Camera.shape[0] == P.shape[0]

    R=Camera[:,0:9].view(-1,1,3,3)
    # R=R.repeat(1, P.shape[1], P.shape[2],1)
    
    # T= Camera[:,9:12].view(-1,1,3,1)
    
    # T=T.repeat(1, P.shape[1], P.shape[2],1)
    # X_cam = torch.matmul(R,(torch.transpose(P,-1,-2) - T ))# rotate and translate
    X_cam = torch.matmul(R,torch.transpose(P,-1,-2))  
    # X_cam = R.dot( P.T) + T  # rotate and translate

    return torch.transpose(X_cam, -1 , -2)

def camera_to_world_batch2(P, Camera):
    """Inverse of world_to_camera_frame
    Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    Returns
    X_cam: Nx3 points in world coordinates
    """
    assert P.shape[-1] == 3
    assert Camera.shape[-1] == 9
    assert Camera.shape[0] == P.shape[0]

    R=Camera[:,0:9].view(-1,1,3,3)
    
    # R=R.repeat(1, P.shape[1], 1,1)
    T= Camera[:,9:12].view(-1,1,3,1)
    
    # T=T.repeat(1, P.shape[1], 1,1)
    X_cam = torch.matmul(torch.transpose(R,-1,-2), (torch.transpose(P,-1,-2))) # rotate and translate

    return torch.transpose(X_cam, -1 , -2)

def check_rotation_matrix(Camera):
    "R: 3x3 Camera rotation matrix"

    assert Camera.shape[-1] == 9

    R=Camera[:,0:9].view(-1,1,3,3)

    return(torch.mean(torch.det(R)))


