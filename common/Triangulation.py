import numpy as np
import cv2
from torch import nn

'''
Code borrowed from: https://github.com/Eliasvan/Multiple-Quadrotor-SLAM
'''

def linear_eigen_triangulation(u1, P1, u2, P2, max_coordinate_value=1.e16):
	"""
	Linear Eigenvalue based (using SVD) triangulation.
	Wrapper to OpenCV's "triangulatePoints()" function.
	Relative speed: 1.0
	(u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
	(u2, P2) is the second pair.
	"max_coordinate_value" is a threshold to decide whether points are at infinity
	u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
	The status-vector is based on the assumption that all 3D points have finite coordinates.
	"""
	x = cv2.triangulatePoints(P1[0:3, 0:4], P2[0:3, 0:4], u1.T, u2.T)  # OpenCV's Linear-Eigen triangl

	x[0:3, :] /= x[3:4, :]  # normalize coordinates
	x_status = (np.max(abs(x[0:3, :]), axis=0) <= max_coordinate_value)  # NaN or Inf will receive status False

	return x[0:3, :].T.astype(output_dtype), x_status


# Initialize consts to be used in linear_LS_triangulation()
linear_LS_triangulation_C = -np.eye(2, 3)


def linear_LS_triangulation(u1, P1, u2, P2):
	"""
	Linear Least Squares based triangulation.
	Relative speed: 0.1
	(u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
	(u2, P2) is the second pair.
	u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
	The status-vector will be True for all points.
	"""
	A = np.zeros((4, 3))
	b = np.zeros((4, 1))

	# Create array of triangulated points
	x = np.zeros((3, len(u1)))

	# Initialize C matrices
	C1 = np.array(linear_LS_triangulation_C)
	C2 = np.array(linear_LS_triangulation_C)

	for i in range(len(u1)):
		# Derivation of matrices A and b:
		# for each camera following equations hold in case of perfect point matches:
		#     u.x * (P[2,:] * x)     =     P[0,:] * x
		#     u.y * (P[2,:] * x)     =     P[1,:] * x
		# and imposing the constraint:
		#     x = [x.x, x.y, x.z, 1]^T
		# yields:
		#     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
		#     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
		# and since we have to do this for 2 cameras, and since we imposed the constraint,
		# we have to solve 4 equations in 3 unknowns (in LS sense).

		# Build C matrices, to construct A and b in a concise way
		C1[:, 2] = u1[i, :]
		C2[:, 2] = u2[i, :]

		# Build A matrix:
		# [
		#     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
		#     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
		#     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
		#     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
		# ]
		A[0:2, :] = C1.dot(P1[0:3, 0:3])  # C1 * R1
		A[2:4, :] = C2.dot(P2[0:3, 0:3])  # C2 * R2

		# Build b vector:
		# [
		#     [ -(u1.x * P1[2,3] - P1[0,3]) ],
		#     [ -(u1.y * P1[2,3] - P1[1,3]) ],
		#     [ -(u2.x * P2[2,3] - P2[0,3]) ],
		#     [ -(u2.y * P2[2,3] - P2[1,3]) ]
		# ]
		b[0:2, :] = C1.dot(P1[0:3, 3:4])  # C1 * t1
		b[2:4, :] = C2.dot(P2[0:3, 3:4])  # C2 * t2
		b *= -1

		# Solve for x vector
		cv2.solve(A, b, x[:, i:i + 1], cv2.DECOMP_SVD)

	return x.T.astype(output_dtype), np.ones(len(u1), dtype=bool)


# Initialize consts to be used in iterative_LS_triangulation()
iterative_LS_triangulation_C = -np.eye(2, 3)


def iterative_LS_triangulation(u1, P1, u2, P2, tolerance=3.e-5):
	"""
	Iterative (Linear) Least Squares based triangulation.
	From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
	Relative speed: 0.025
	(u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
	(u2, P2) is the second pair.
	"tolerance" is the depth convergence tolerance.
	Additionally returns a status-vector to indicate outliers:
		1: inlier, and in front of both cameras
		0: outlier, but in front of both cameras
		-1: only in front of second camera
		-2: only in front of first camera
		-3: not in front of any camera
	Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).
	u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
	"""
	A = np.zeros((4, 3))
	b = np.zeros((4, 1))

	# Create array of triangulated points
	x = np.empty((4, len(u1)))
	x[3, :].fill(1)  # create empty array of homogenous 3D coordinates
	x_status = np.empty(len(u1), dtype=int)

	# Initialize C matrices
	C1 = np.array(iterative_LS_triangulation_C)
	C2 = np.array(iterative_LS_triangulation_C)

	for xi in range(len(u1)):
		# Build C matrices, to construct A and b in a concise way
		C1[:, 2] = u1[xi, :]
		C2[:, 2] = u2[xi, :]

		# Build A matrix
		A[0:2, :] = C1.dot(P1[0:3, 0:3])  # C1 * R1
		A[2:4, :] = C2.dot(P2[0:3, 0:3])  # C2 * R2

		# Build b vector
		b[0:2, :] = C1.dot(P1[0:3, 3:4])  # C1 * t1
		b[2:4, :] = C2.dot(P2[0:3, 3:4])  # C2 * t2
		b *= -1

		# Init depths
		d1 = d2 = 1.

		for i in range(10):  # Hartley suggests 10 iterations at most
			# Solve for x vector
			cv2.solve(A, b, x[0:3, xi:xi + 1], cv2.DECOMP_SVD)

			# Calculate new depths
			d1_new = P1[2, :].dot(x[:, xi])
			d2_new = P2[2, :].dot(x[:, xi])

			if abs(d1_new - d1) <= tolerance and \
							abs(d2_new - d2) <= tolerance:
				break

			# Re-weight A matrix and b vector with the new depths
			A[0:2, :] *= 1 / d1_new
			A[2:4, :] *= 1 / d2_new
			b[0:2, :] *= 1 / d1_new
			b[2:4, :] *= 1 / d2_new

			# Update depths
			d1 = d1_new
			d2 = d2_new

		# Set status
		x_status[xi] = (i < 10 and  # points should have converged by now
		                (d1_new > 0 and d2_new > 0))  # points should be in front of both cameras
		if d1_new <= 0: x_status[xi] -= 1
		if d2_new <= 0: x_status[xi] -= 2

	return x[0:3, :].T.astype(output_dtype), x_status


def polynomial_triangulation(u1, P1, u2, P2):
	"""
	Polynomial (Optimal) triangulation.
	Uses Linear-Eigen for final triangulation.
	Relative speed: 0.1
	(u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
	(u2, P2) is the second pair.
	u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
	The status-vector is based on the assumption that all 3D points have finite coordinates.
	"""
	P1_full = np.eye(4)
	P1_full[0:3, :] = P1[0:3, :]  # convert to 4x4
	P2_full = np.eye(4)
	P2_full[0:3, :] = P2[0:3, :]  # convert to 4x4
	P_canon = P2_full.dot(cv2.invert(P1_full)[1])  # find canonical P which satisfies P2 = P_canon * P1

	# "F = [t]_cross * R" [HZ 9.2.4]; transpose is needed for numpy
	F = np.cross(P_canon[0:3, 3], P_canon[0:3, 0:3], axisb=0).T

	# Other way of calculating "F" [HZ (9.2)]
	# op1 = (P2[0:3, 3:4] - P2[0:3, 0:3] .dot (cv2.invert(P1[0:3, 0:3])[1]) .dot (P1[0:3, 3:4]))
	# op2 = P2[0:3, 0:4] .dot (cv2.invert(P1_full)[1][0:4, 0:3])
	# F = np.cross(op1.reshape(-1), op2, axisb=0).T

	# Project 2D matches to closest pair of epipolar lines
	u1_new, u2_new = cv2.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))

	# For a purely sideways trajectory of 2nd cam, correctMatches() returns NaN for all possible points!
	if np.isnan(u1_new).all() or np.isnan(u2_new).all():
		F = cv2.findFundamentalMat(u1, u2, cv2.FM_8POINT)[0]  # so use a noisy version of the fund mat
		u1_new, u2_new = cv2.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))

	# Triangulate using the refined image points
	return linear_eigen_triangulation(u1_new[0], P1, u2_new[0], P2)


output_dtype = float


def set_triangl_output_dtype(output_dtype_):
	"""
	Set the datatype of the triangulated 3D point positions.
	(Default is set to "float")
	"""
	global output_dtype
	output_dtype = output_dtype_

import torch


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

def triangulate_batch_of_points(proj_matricies_batch, points_batch, confidences_batch=None):
    n_views, n_joints = points_batch.shape[:2]
    point_3d_batch = torch.zeros(n_joints, 3, dtype=torch.float32, device=points_batch.device)
    print(n_joints)

    for joint_i in range(0,17):
        points = points_batch[ :, joint_i, :]
        confidences = confidences_batch[:, joint_i] if confidences_batch is not None else None
        point_3d = triangulate_point_from_multiple_views_linear_torch(proj_matricies_batch, points, confidences=confidences)
        point_3d_batch[joint_i] = point_3d

    return point_3d_batch


def triangulate_point_from_multiple_views_linear(proj_matricies, points, confidences):
    """Triangulates one point from multiple (N) views using direct linear transformation (DLT).
    For more information look at "Multiple view geometry in computer vision",
    Richard Hartley and Andrew Zisserman, 12.2 (p. 312).
    Args:
        proj_matricies numpy array of shape (N, 3, 4): sequence of projection matricies (3x4)
        points numpy array of shape (N, 2): sequence of points' coordinates
    Returns:
        point_3d numpy array of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)
    A = np.zeros((n_views,2, 4))
    
    for j in range(len(proj_matricies)):
        A[j, 0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
        A[j, 1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]
    confidences=np.reshape(confidences, (n_views,1,1))
    A*=confidences
    A=np.reshape(A,(-1,4))
    u, s, vh =  np.linalg.svd(A, full_matrices=False)
    point_3d_homo = vh[3, :]

    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d


def triangulate_point_from_multiple_views_linear_torch(proj_matricies, points, confidences=None):
    """Similar as triangulate_point_from_multiple_views_linear() but for PyTorch.
    For more information see its documentation.
    Args:
        proj_matricies torch tensor of shape (N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (N, 2): sequence of points' coordinates
        confidences None or torch tensor of shape (N,): confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0
    Returns:
        point_3d numpy torch tensor of shape (3,): triangulated point
    """

    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)

    if confidences is None:
        confidences = torch.ones(n_views, dtype=torch.float32, device=points.device)

    A = proj_matricies[:, 2:3].expand(n_views, 2, 4) * points.view(n_views, 2, 1)
    A -= proj_matricies[:, :2]
    A *= confidences.view(-1, 1, 1)

    u, s, vh = torch.svd(A.view(-1, 4))

    point_3d_homo = -vh[:, 3]
    point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]

    return point_3d

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



class RANSACTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        
        self.direct_optimization = config.model.direct_optimization

    def forward(self, images, proj_matricies, batch):
        batch_size, n_views = images.shape[:2]

        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])

        # forward backbone and integrate
        heatmaps, _, _, _ = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])

        # calcualte shapes
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])

        # keypoints 2d
        _, max_indicies = torch.max(heatmaps.view(batch_size, n_views, n_joints, -1), dim=-1)
        keypoints_2d = torch.stack([max_indicies % heatmap_shape[1], max_indicies // heatmap_shape[1]], dim=-1).to(images.device)

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed

        # triangulate (cpu)
        keypoints_2d_np = keypoints_2d.detach().cpu().numpy()
        proj_matricies_np = proj_matricies.detach().cpu().numpy()

        keypoints_3d = np.zeros((batch_size, n_joints, 3))
        confidences = np.zeros((batch_size, n_views, n_joints))  # plug
        for batch_i in range(batch_size):
            for joint_i in range(n_joints):
                current_proj_matricies = proj_matricies_np[batch_i]
                points = keypoints_2d_np[batch_i, :, joint_i]
                keypoint_3d, _ = self.triangulate_ransac(current_proj_matricies, points, direct_optimization=self.direct_optimization)
                keypoints_3d[batch_i, joint_i] = keypoint_3d

        keypoints_3d = torch.from_numpy(keypoints_3d).type(torch.float).to(images.device)
        confidences = torch.from_numpy(confidences).type(torch.float).to(images.device)

        return keypoints_3d, keypoints_2d, heatmaps, confidences

    def triangulate_ransac(self, proj_matricies, points, n_iters=10, reprojection_error_epsilon=15, direct_optimization=True):
        assert len(proj_matricies) == len(points)
        assert len(points) >= 2

        proj_matricies = np.array(proj_matricies)
        points = np.array(points)

        n_views = len(points)

        # determine inliers
        view_set = set(range(n_views))
        inlier_set = set()
        for i in range(n_iters):
            sampled_views = sorted(random.sample(view_set, 2))

            keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(proj_matricies[sampled_views], points[sampled_views])
			
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), points, proj_matricies)[0]

            new_inlier_set = set(sampled_views)
            for view in view_set:
                current_reprojection_error = reprojection_error_vector[view]
                if current_reprojection_error < reprojection_error_epsilon:
                    new_inlier_set.add(view)

            if len(new_inlier_set) > len(inlier_set):
                inlier_set = new_inlier_set

        # triangulate using inlier_set
        if len(inlier_set) == 0:
            inlier_set = view_set.copy()

        inlier_list = np.array(sorted(inlier_set))
        inlier_proj_matricies = proj_matricies[inlier_list]
        inlier_points = points[inlier_list]

        keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(inlier_proj_matricies, inlier_points)
        reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
        reprojection_error_mean = np.mean(reprojection_error_vector)

        keypoint_3d_in_base_camera_before_direct_optimization = keypoint_3d_in_base_camera
        reprojection_error_before_direct_optimization = reprojection_error_mean

        # direct reprojection error minimization
        if direct_optimization:
            def residual_function(x):
                reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([x]), inlier_points, inlier_proj_matricies)[0]
                residuals = reprojection_error_vector
                return residuals

            x_0 = np.array(keypoint_3d_in_base_camera)
            res = least_squares(residual_function, x_0, loss='huber', method='trf')

            keypoint_3d_in_base_camera = res.x
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
            reprojection_error_mean = np.mean(reprojection_error_vector)

        return keypoint_3d_in_base_camera, inlier_list


class AlgebraicTriangulationNet(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()

        # self.use_confidences = config.model.use_confidences

        # config.model.backbone.alg_confidences = False
        # config.model.backbone.vol_confidences = False

        # if self.use_confidences:
        #     config.model.backbone.alg_confidences = True

        # self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        # self.heatmap_softmax = config.model.heatmap_softmax
        # self.heatmap_multiplier = config.model.heatmap_multiplier


    def forward(self, points, confidences, proj_matricies, batch):
        # device = images.device
        # batch_size, n_views = points.shape[:2]

        # # reshape n_views dimension to batch dimension
        # images = images.view(-1, *images.shape[2:])

        # # forward backbone and integral
        # if self.use_confidences:
        #     heatmaps, _, alg_confidences, _ = self.backbone(images)
        # else:
        #     heatmaps, _, _, _ = self.backbone(images)
        #     alg_confidences = torch.ones(batch_size * n_views, heatmaps.shape[1]).type(torch.float).to(device)

        # heatmaps_before_softmax = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        # keypoints_2d, heatmaps = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)

        # # reshape back
        # images = images.view(batch_size, n_views, *images.shape[1:])
        # heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        # keypoints_2d = keypoints_2d.view(batch_size, n_views, *keypoints_2d.shape[1:])
        # alg_confidences = alg_confidences.view(batch_size, n_views, *alg_confidences.shape[1:])

        # # norm confidences
        # alg_confidences = alg_confidences / alg_confidences.sum(dim=1, keepdim=True)
        # alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # # calcualte shapes
        # image_shape = tuple(images.shape[3:])
        # batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])

        # # upscale keypoints_2d, because image shape != heatmap shape
        # keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        # keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        # keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        # keypoints_2d = keypoints_2d_transformed
        
        # triangulate
        try:
            keypoints_3d = triangulate_batch_of_points(
                proj_matricies, keypoints_2d,
                confidences_batch=alg_confidences
            )
        except RuntimeError as e:
            print("Error: ", e)

            print("confidences =", confidences_batch_pred)
            print("proj_matricies = ", proj_matricies)
            print("keypoints_2d_batch_pred =", keypoints_2d_batch_pred)
            exit()

        return keypoints_3d #, keypoints_2d, heatmaps, alg_confidences


class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side

        self.kind = config.model.kind
        self.use_gt_pelvis = config.model.use_gt_pelvis

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        for p in self.backbone.final_layer.parameters():
            p.requires_grad = False

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)


    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        heatmaps, features, _, vol_confidences = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            if self.use_gt_pelvis:
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                keypoints_3d = batch['pred_keypoints_3d'][batch_i]

            if self.kind == "coco":
                base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
            elif self.kind == "mpii":
                base_point = keypoints_3d[6, :3]

            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + center

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        # integral 3d
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, features, volumes, vol_confidences, cuboids, coord_volumes, base_points