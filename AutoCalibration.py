import numpy as np
import os
import torch
import torch.nn as nn
from torch_geometric.nn import knn
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
import pandas as pd
import scipy

from hugeica import *
from hugeica import Adam_Lie
from Mine import *
from Crop import crop

def euler_diff(A, B):
    """
    Computes the difference in Euler angles between two rotation matrices A and B.
    """
    initR = A
    estR = B

    err = estR @ initR.T

    err = scipy.spatial.transform.Rotation.from_matrix(err)
    err = err.as_euler("xyz", degrees=True)
    return err

def quat_diff(A, B):
    """
    Computes the difference in Quaternion angles between two rotation matrices A and B.
    """
    
    def quaternion_multiply(quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def inv(quat):
        return np.hstack([quat[0:1], -quat[1:4] ])

    initR = A
    estR = B

    qdist = Rot.from_matrix(initR).as_quat()
    qestR = inv(Rot.from_matrix(estR).as_quat())
    
    d = quaternion_multiply(qdist, qestR)
    
    vec = np.linalg.norm(d[1:])
    mag = np.abs(d[0])
    
    rad = 2 * np.arctan2(vec, mag)

    deg = 180*np.abs(rad)/np.pi
    
    return deg


def randomRT(range_t, range_R):
    """
    Generates two random triplets for the given range.
    
    Return
        R (float, float, float), Euler angles
        T (float, float, float), trnaslation vector
    """
    
    T = np.random.uniform(-range_t, range_t, (3,))
    R = np.random.uniform(-range_R, range_R, (3,))
    
    if quat_diff(Rot.from_euler('xyz', R, degrees=True).as_matrix(), np.eye(3)) >= range_R or np.linalg.norm(T) >= range_t:
        return randomRT(range_t, range_R)
    
    if quat_diff(Rot.from_euler('xyz', R, degrees=True).as_matrix(), np.eye(3)) < range_R * 0.3 or np.linalg.norm(T) > range_t < range_R * 0.3:
        return randomRT(range_t, range_R)
    
    return R, T

def nn_sklearn(points, grid):
    """
    1-nearest neighbor using scikit-learn.
    """
    nrs = NearestNeighbors(n_neighbors=1)
    nrs.fit(points.detach().cpu().numpy())# + np.random.normal(0, 3, points.shape))
    neigh_dist, idx_neighbors = nrs.kneighbors(grid.cpu().numpy(), return_distance=True)
    neigh_dist, idx_neighbors = neigh_dist[:, 0], idx_neighbors[:, 0]
    return torch.from_numpy(neigh_dist), torch.from_numpy(idx_neighbors)

def nn_torch(points, grid):
    """
    1-nearest neighbor using torch geometric.
    """
    # Finds for each element in y the k nearest points in x.
    assign_index = knn(points, grid, 1)
    idx_src, target_idx = assign_index
    neigh_dist = torch.norm(grid[idx_src] - points[target_idx], dim=1)
    return neigh_dist, target_idx


class AutoCalibration(nn.Module):
    """
    Computes extrinisic calibration parameters R, T for a given sequence of Lidar and camera frames.
    """

    
    def __init__(self, RT, P_rect, patch_size=96, 
                 M=24, Mg=5, mi_batches=1, hidden_size=64, max_z=20, 
                 unique=True, max_dist_to_grid=300, clip_floor_3d = -1, 
                 mode="euclidean", rgb_channels=1, pc_batch=5000):
        """
        Parameters:
            RT (matrix 4x4), initial guess.
            P_rect (matrix 4x4), camera matrix
            patch_size (int, optional), the used patch size for optimiztaion
            M (int, optional), mini batch size
            Mg (int, optional), The distance between grid points
            mi_batches (int, optional), The number of frames for mutual information estimation        
        """
        super().__init__()
        
        # Hypers
        self.patch_size = patch_size
        self.pc_batch_size = pc_batch
        self.mi_batch_size = M
        self.mi_batches = mi_batches
        self.max_z = max_z
        self.max_dist_to_grid = max_dist_to_grid
        self.clip_floor_3d = clip_floor_3d # 3d z-coord
        self.clip_floor_2d = 100           # image pixels
        self.P_rect = torch.from_numpy(P_rect).float().clone()
        self.initRT = torch.from_numpy(RT).float().clone()
        self.contrast = "random"
        self.Mg = Mg
        self.unique = unique
        self.mode = mode
        self.rgb_channels = rgb_channels
        self.lidar_channels = 2 if mode == "euclidean+reflectance" else 1
        self.patience = 40
        self.n_steps = 10
        
        
        # PARAMS
        self.R = SO_Layer(n_dims=3)
        self.T = nn.Linear(3,3)
        self.reset_calibration()
        
        self.MINE =  Mine(fmaps=hidden_size, hidden_size=hidden_size, 
                          dimA=(self.rgb_channels, self.patch_size, self.patch_size), 
                          dimB=(self.lidar_channels, self.patch_size, self.patch_size))
        
        
        # STATE
        self.history = []
        self.history_test = []
        self.histor_rt = []
        self.best_mi = -100
        self.bu_max_z =  self.max_z
        self.l2_old  = 0
        self.init_checkpoint = None
        self.c_steps = self.n_steps
        
        # OPTIMIZER
        self.optim = None
        self.update_se_and_mine()
        
        # TEMP
        self.err_count = 0
        self.iter = 0
                
        
    def update_se_and_mine(self, lr_mi=1e-3, lr_R=1e-3, lr_T=1e-2):
        if lr_mi is None:
            lr_R, lr_T, lr_mi = [param["lr"] for param in self.optim.param_groups]
        self.optim = Adam_Lie([{'params': self.R.parameters(), 'lr': lr_R},
                       {'params': self.T.parameters(), 'lr': lr_T},
                       {'params': self.MINE.parameters(), 'lr': lr_mi}])

    def update_se(self, lr_R=1e-3, lr_T=1e-2):
        self.optim = Adam_Lie([{'params': self.R.parameters(), 'lr': lr_R},
                       {'params': self.T.parameters(), 'lr': lr_T}])

    def update_mine(self, lr=1e-3):
        self.optim = Adam_Lie([{'params': self.MINE.parameters(), 'lr': lr}])
        
        
    def reduce_lr(self, r=0.1):
        for param in self.optim.param_groups:
            param["lr"] = param["lr"] * r
     
    def increase_lr(self, r=10):
        for param in self.optim.param_groups:
            param["lr"] = param["lr"] * r
        
    def RT(self):
        RT = np.zeros((4,4))
        RT[3,  3] = 1
        RT[:3,:3] = self.R.weight.data[:3, :3].cpu().numpy()
        RT[:3, 3] = self.T.bias.data.cpu().numpy()
        return RT
    
    def distort_calibration(self, T=[0.01, 0.01, 0.01], R=[2, 2, 2]):
        device = self.R.weight.data.device

        self.T.bias.data = self.T.bias.data + torch.from_numpy(np.asarray(T)).to(device).view(self.T.bias.data.shape).float()

        r = Rot.from_euler('xyz', R, degrees=True)
        
        r = r.as_matrix()
        error = torch.from_numpy(r).to(device).float()
        
        
        self.R.weight.data = error @ self.R.weight.data
        
    def load_and_reset(self, checkpoint):
        self.init_checkpoint = checkpoint
        self.load_state_dict(torch.load(checkpoint))
        self.reset_calibration()
        
    def reset_calibration(self, RT=None):
        if RT is None:
            RT = self.initRT
    
        RT = torch.from_numpy(RT).float() if torch.is_tensor(RT) == False else RT
        device = self.R.weight.data.device
        
        self.T.weight.data *= 0 
        self.R.weight.data = RT[:3, :3].to(device).clone()
        self.T.bias.data = RT[:3, 3].to(device).clone()
        
    def to(self, device):
        self.P_rect = self.P_rect.to(device)
        self.initRT = self.initRT.to(device)
        return super().to(device)
    
    def transform_points(self, point_cloud, image, clipping=True, RT=None, P=None, grad_T=True, grad_R=True):
        """
        Projects the points cloud onto the image plane and clips invalid points outside the view frustum.
        """
        
        device = next(self.parameters()).device
                
        if clipping == False:
            bu_max_z = self.max_z
            bu_clip_floor_3d = self.clip_floor_3d
            bu_clip_floor_2d = self.clip_floor_2d
            self.max_z = np.iinfo(np.int32).max
            self.clip_floor_3d = np.iinfo(np.int32).min 
            self.clip_floor_2d = 0
        
        point_cloud, I = torch.from_numpy(point_cloud[:, :3]).float().to(device), torch.from_numpy(point_cloud[:, 3:]).float().to(device)
        image =  torch.from_numpy(np.asarray(image)/255.).float().transpose(1,2).transpose(0,1).unsqueeze(0).to(device)
        
        #variables
        max_x, max_y, max_z = image.shape[3], image.shape[2], self.max_z

        # transform
        I = I[point_cloud[:, 2] > self.clip_floor_3d]
        point_cloud = point_cloud[point_cloud[:, 2] > self.clip_floor_3d]
        if RT is None:
            R = self.R(point_cloud) if grad_R else self.R(point_cloud).detach()
            T = self.T.bias if grad_T else self.T.bias.detach()
        else:
            R = (RT[:3, :3] @ point_cloud.T).T
            T = RT[:3, 3]

        T_pointcloud = R + T
        T_pointcloud_h = torch.cat([T_pointcloud, (T_pointcloud[:, :1] * 0) + 1], axis=1)
        
        # project
        if P is None:
            proj_pointcloud =  (self.P_rect @ T_pointcloud_h.T).T
        else:
            proj_pointcloud =  (P @ T_pointcloud_h.T).T
            

        # clip points behind camera
        idx = np.arange(len(proj_pointcloud))
        mask = proj_pointcloud[:, 2].detach().cpu().numpy() > 1.
        mask = np.logical_and(mask, proj_pointcloud[:, 2].detach().cpu().numpy() < max_z)

        # z-normalize
        z = proj_pointcloud[:, 2:].clone()
        proj_pointcloud /= z
        proj_pointcloud = proj_pointcloud

        # image frustum
        mask = np.logical_and(mask, proj_pointcloud[:, 0].detach().cpu().numpy() < max_x - self.patch_size/2)
        mask = np.logical_and(mask, proj_pointcloud[:, 1].detach().cpu().numpy() < max_y - self.patch_size/2)
        mask = np.logical_and(mask, proj_pointcloud[:, 0].detach().cpu().numpy() > 0 + self.patch_size/2)
        mask = np.logical_and(mask, proj_pointcloud[:, 1].detach().cpu().numpy() > 0 + self.patch_size/2)
        mask = torch.from_numpy(mask).bool()

        # visible lidar points
        idx = idx[mask]
        valid_proj_pointcloud = proj_pointcloud[mask]
        x, y, ones = valid_proj_pointcloud.T

        # Project the points
        x, y, z, I = x.long().detach(), y.long().detach(), z.detach()[mask,0], I.detach()[mask, 0]
        
        if not clipping:
            self.max_z = bu_max_z
            self.clip_floor_3d = bu_clip_floor_3d 
            self.clip_floor_2d = bu_clip_floor_2d 
        
        return image, x, y, z, I, mask, max_x, max_y, max_z, valid_proj_pointcloud[: , :2]
    
    def get_projections(self, frames, min_idx, max_idx, RT=None, P=None, grad_T=True, grad_R=True):
        """
        Samples projections from the given dataset
        """
        if self.err_count > 100:
            raise ValueError("Could not find valid projections!")

        device = next(self.parameters()).device
        
        valid_proj_pointcloud = []
        
        # Sample image
        idx = np.random.randint(min_idx, max_idx)
        point_cloud, image, image3, oxts = frames[idx]
        image, x, y, z, I, mask, max_x, max_y, max_z, valid_proj_pointcloud = self.transform_points(point_cloud, image, RT=RT, P=P, grad_T=grad_T, grad_R=grad_R)
        # print(len(valid_proj_pointcloud), RT, P)
            
        if len(valid_proj_pointcloud) == 0:
            print("We need to reproject!", self.err_count)            
            self.err_count += 1
            return self.get_projections(frames, min_idx, max_idx, RT=RT, P=P, grad_T=grad_T, grad_R=grad_R)
        else:
            self.err_count = 0.
            
        if self.mode == "euclidean":
            black_image = torch.ones((1, 1, max_y, max_x)).to(device) * 0.
            std = z.detach().std()
            std = 1 if torch.isnan(std) else std
            black_image[0, 0, y, x] = z.detach() / std
        elif self.mode == "reflectance":
            black_image = torch.ones((1, 1, max_y, max_x)).to(device) * 0.
            std = I.detach().std()
            std = 1 if torch.isnan(std) else std
            black_image[0, 0, y, x] = I.detach() / std
        elif self.mode == "euclidean+reflectance":
            black_image = torch.ones((1, 2, max_y, max_x)).to(device) * 0.
            std = z.detach().std()
            std = 1 if torch.isnan(std) else std
            black_image[0, 0, y, x] = z.detach() / std
            std = I.detach().std()
            std = 1 if torch.isnan(std) else std
            black_image[0, 1, y, x] = I.detach() / std
        else:
            print("No valid 'mode' given", self.mode)
          
    
        # Subsample uniform grid for subsampling the point cloud
        height = max_y - self.patch_size/2 - self.clip_floor_2d
        width  = max_x - self.patch_size/2
        
        x = np.linspace(0, width, int(width/self.Mg ) )
        y = np.linspace(0, height, int(height/self.Mg ))
        grid_2d = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        
        device = valid_proj_pointcloud.device
        valid_proj_pointcloud = valid_proj_pointcloud[:(np.min([self.pc_batch_size, len(valid_proj_pointcloud)]))]
        neigh_dist, idx_neighbors = nn_torch(valid_proj_pointcloud, torch.from_numpy(grid_2d).to(device).float())
        # neigh_dist, idx_neighbors = nn_sklearn(valid_proj_pointcloud, torch.from_numpy(grid_2d))
        
        idx_neighbors = idx_neighbors[neigh_dist < self.max_dist_to_grid]
        if self.unique:
            idx_neighbors = torch.unique(idx_neighbors)
        
        valid_proj_pointcloud = valid_proj_pointcloud[idx_neighbors]
        
        if len(valid_proj_pointcloud) < self.mi_batch_size:
            self.err_count += 1
            if self.err_count > 5:
                print("We had to duplicate points as there where too few projected.", self.err_count)
                less = int(self.mi_batch_size / len(valid_proj_pointcloud)) + 1
                valid_proj_pointcloud = np.concatenate([valid_proj_pointcloud] * less)
            else:
                return self.get_projections(frames, min_idx, max_idx, RT=RT, P=P, grad_T=grad_T, grad_R=grad_R)
        else:
            self.err_count = 0.
        
        return image, black_image, valid_proj_pointcloud
    
    def get_joint(self, valid_proj_pointcloud, image, black_image, batch_size, rng=None, std_xy=1.25):
        
        if rng is None:
            rng = np.random
            
         # Differentialble Sampling
        index_joint = rng.choice(len(valid_proj_pointcloud), size=batch_size, replace=False)
        crops_image_joint = crop(image.repeat(batch_size, 1, 1, 1),       
                                 valid_proj_pointcloud[index_joint], 
                                 crop_size=(self.patch_size,self.patch_size))
        
        # Unplug the gradient
        valid_proj_pointcloud = valid_proj_pointcloud.detach() 
        
        crops_black_joint = crop(black_image.repeat(batch_size, 1, 1, 1), 
                                 valid_proj_pointcloud.detach()[index_joint], 
                                 crop_size=(self.patch_size,self.patch_size), std_x=std_xy, std_y=std_xy)
        
        joint = torch.cat([crops_image_joint[:, :self.rgb_channels], crops_black_joint], dim=1)
            
        return joint, crops_image_joint, crops_black_joint
    
    
    def get_marginal(self, valid_proj_pointcloud, image, black_image, batch_size, rng=None, std_xy=1.25):
        
        if rng is None:
            rng = np.random
        
        # Unplug the gradient
        valid_proj_pointcloud = valid_proj_pointcloud.detach() 
        
        index_joint = rng.choice(len(valid_proj_pointcloud), size=batch_size, replace=False)
        crops_image_margi = crop(image.repeat(batch_size, 1, 1, 1), 
                                 valid_proj_pointcloud[index_joint], 
                                 crop_size=(self.patch_size,self.patch_size))
        
        if self.contrast == "shift36":
            index_marginal = index_joint
            valid_proj_pointcloud = valid_proj_pointcloud + torch.distributions.Uniform(3, 6).sample(valid_proj_pointcloud.shape).to(device)
            crops_black_margi = crop(black_image.repeat(batch_size, 1, 1, 1), 
                                     valid_proj_pointcloud[index_marginal], 
                                     crop_size=(self.patch_size,self.patch_size), std_x=std_xy, std_y=std_xy)
        elif self.contrast == "random":
            index_marginal = rng.choice(len(valid_proj_pointcloud), size=batch_size, replace=False)
        else:
            raise ValueError(f"Contrast type {self.contrast} unknown.")
        
        crops_black_margi = crop(black_image.repeat(batch_size, 1, 1, 1), 
                                 valid_proj_pointcloud.detach()[index_marginal], 
                                 crop_size=(self.patch_size,self.patch_size), std_x=std_xy, std_y=std_xy)
        marginal = torch.cat([crops_image_margi[:, :self.rgb_channels], crops_black_margi], dim=1)
        
        return marginal, crops_image_margi, crops_black_margi
        
    
    def step(self, frames, min_idx, max_idx, batch_size, grad_se=False, viz="none", RT=None, P=None, grad_T=True, grad_R=True, grad_mine=True, rng=None):
        device = next(self.parameters()).device

        # First sample some projected points
        image, black_image, valid_proj_pointcloud = self.get_projections(frames, min_idx, max_idx, RT=RT, P=P, grad_T=grad_T, grad_R=grad_R)
        
        if not grad_se:
            valid_proj_pointcloud = valid_proj_pointcloud.detach()
        
        res = []
        for _ in range(self.mi_batches):
            
            if grad_se or grad_mine:
                self.optim.zero_grad()
            
            
            # Construct the contrastive datasets
            joint, crops_image_joint, crops_black_joint = self.get_joint(valid_proj_pointcloud, image, black_image, batch_size, rng)
            marginal, crops_image_margi, crops_black_margi = self.get_marginal(valid_proj_pointcloud, image, black_image, batch_size, rng)

            # Optimize MI
            loss_, mi_lb_, acc_, pred = self.MINE.mutual_information(joint, marginal)
            
            if grad_se or grad_mine:
                loss_.backward(retain_graph=True)
                
                #
                # Null the MINE if needed
                if not grad_mine:
                    for p in self.MINE.parameters():
                        p.grad *= 0.
                        
                #
                # TAKE A STEP
                self.optim.step()
                
                #
                # CLEAN UP
                self.optim.zero_grad()

            res.append( torch.stack([loss_.detach().clone(), mi_lb_.detach().clone(), acc_.detach().clone()]) )
            
        
        loss, mi_lb, acc = torch.stack(res, dim=0).mean(0)
        
  
        if viz == "joint":
            plt.rcParams["figure.figsize"] = (5,5)
            pred_j, pred_m = pred.view(2, -1)
            plt.title(pred_j[0])
            plt.subplot(1,2, 1)
            plt.imshow(crops_black_joint[0][0].detach().cpu().numpy(), cmap="gray", interpolation="none")
            plt.axis("off")
            plt.subplot(1,2, 2)
            mix = np.clip(crops_black_joint[0][0].detach().cpu().numpy()[:, :, None] + crops_image_joint[0].detach().cpu().numpy().transpose(1,2,0), 0, 1)      
            plt.imshow(mix, interpolation="none")
            plt.axis("off")
            plt.show()
            
        if viz == "jointn":
            image, black_image, valid_proj_pointcloud = self.get_projections(frames, 0, 10, 
                                                                             RT=RT, P=P, grad_T=grad_T, grad_R=grad_R)
            valid_proj_pointcloud = valid_proj_pointcloud.detach()
            joint, crops_image_joint, crops_black_joint = self.get_joint(valid_proj_pointcloud, image, black_image, batch_size, rng)
            marginal, crops_image_margi, crops_black_margi = self.get_marginal(valid_proj_pointcloud, image, black_image, batch_size, rng)
            black = np.tile(crops_black_joint.detach().cpu().numpy(), (1, 3, 1, 1))
            black = black - black.min()
            black = black / black.max()
            black[:, 0] *= 0 # R
            black[:, 2] *= 0 # B
            image = crops_image_joint.detach().cpu().numpy()
            plt.rcParams["figure.figsize"] = (7,5)
            pred_j, pred_m = pred.view(2, -1)
            plt.title(pred_j[0])
            plt.subplot(1,3, 1)
            plt.imsave("jointn_black.png", black[0].transpose(1,2,0))
            plt.imshow(black[0].transpose(1,2,0), interpolation="none")
            plt.axis("off")
            plt.subplot(1,3, 2)
            mix = np.clip(image[0].transpose(1,2,0), 0, 1)      
            plt.imsave("jointn_image.png", mix)
            plt.imshow(mix, interpolation="none")
            plt.axis("off")
            plt.subplot(1,3, 3)
            mix = np.clip(black[0].transpose(1,2,0) + image[0].transpose(1,2,0), 0, 1)      
            plt.imsave("jointn_mix.png", mix)
            plt.imshow(mix, interpolation="none")
            plt.axis("off")
            plt.show()
            
        if viz == "marginaln":
            image, black_image, valid_proj_pointcloud = self.get_projections(frames, 0, 10, 
                                                                             RT=RT, P=P, grad_T=grad_T, grad_R=grad_R)
            valid_proj_pointcloud = valid_proj_pointcloud.detach()
            joint, crops_image_joint, crops_black_joint = self.get_joint(valid_proj_pointcloud, image, black_image, batch_size, rng)
            marginal, crops_image_margi, crops_black_margi = self.get_marginal(valid_proj_pointcloud, image, black_image, batch_size, rng)
            black = np.tile(crops_black_margi.detach().cpu().numpy(), (1, 3, 1, 1))
            black = black - black.min()
            black = black / black.max()
            black[:, 0] *= 0 # R
            black[:, 2] *= 0 # B
            image = crops_image_margi.detach().cpu().numpy()
            plt.rcParams["figure.figsize"] = (7,5)
            pred_j, pred_m = pred.view(2, -1)
            plt.title(pred_j[0])
            plt.subplot(1,3, 1)
            plt.imsave("marginaln_black.png", black[0].transpose(1,2,0))
            plt.imshow(black[0].transpose(1,2,0), interpolation="none")
            plt.axis("off")
            plt.subplot(1,3, 2)
            mix = np.clip(image[0].transpose(1,2,0), 0, 1)      
            plt.imsave("marginaln_image.png", mix)
            plt.imshow(mix, interpolation="none")
            plt.axis("off")
            plt.subplot(1,3, 3)
            mix = np.clip(black[0].transpose(1,2,0) + image[0].transpose(1,2,0), 0, 1)      
            plt.imsave("marginaln_mix.png", mix)
            plt.imshow(mix, interpolation="none")
            plt.axis("off")
            plt.show()
            
            
        if viz == "marginal":
            #plt.rcParams["figure.figsize"] = (20,5)
            #plt.subplot(3,1, 1)
            #plt.imshow(black_image[0].detach().cpu().numpy().transpose(1,2,0), cmap="gray", interpolation="none")
            #plt.show()
            plt.rcParams["figure.figsize"] = (5,5)
            pred_j, pred_m = pred.view(2, -1)
            plt.title(pred_m[0])
            plt.subplot(1,2, 1)
            plt.imshow(crops_black_margi[0][0].detach().cpu().numpy(), cmap="gray", interpolation="none")
            plt.axis("off")
            plt.subplot(1,2, 2)
            mix = np.clip(crops_black_margi[0][0].detach().cpu().numpy()[:, :, None] + crops_image_margi[0].detach().cpu().numpy().transpose(1,2,0),0,1)
            plt.imshow(mix, cmap="gray", interpolation="none")
            plt.axis("off")
            plt.show()
        
        return loss, mi_lb, acc, pred
    
    
    def predict(self, frames, min_idx=0, max_idx=100):
        pred_j, pred_m = self.step(frames, min_idx, max_idx)[3].view(2, -1)
        return pred_j, pred_m
    
    
    
    def test(self, frames, iters=10, log_interval=5, min_idx=100, max_idx=150, viz="none", RT=None, P=None, 
             rng = np.random.RandomState(2020)):
        
        history = []
        min_idx = 0
        
        for i in range(iters):
            
            with torch.no_grad():
                loss, mi_lb, acc, pred = self.step(frames, min_idx+i, min_idx+i+1, batch_size=50, grad_se=False, grad_mine=False, viz=viz, RT=RT, P=P, rng=rng)
            
            history.append((loss.detach().item(), mi_lb.detach().item(), acc.detach().item()))
            
            if i % log_interval == 0 and log_interval > 0:
                print(f"MI: {np.asarray(history[-20:]).mean():.2f}, {mi_lb.detach().item():.3f}")
                
        
        return np.asarray(history)
        
    def fit(self, frames_list, iters=1000, log_interval=5, viz=False, 
            grad_SE =False, grad_mine=False, grad_T=True, grad_R=True, 
            log_avg_window=20, n_test_batches=10, valid_ratio=0.1, eps=1e-8,
            record_weight_pool=False, weight_pool_root="checkpoints/mi_model_p96_small", sample_weight_pool=False,
            checkpoints_log_path = None):
        
        device = next(self.parameters()).device
        
        if sample_weight_pool:
            if record_weight_pool:
                raise ValueError("Cannot sample and record weight pool at the same time.")
            pool = os.listdir(weight_pool_root)
            print("MINE weight pool size is", len(pool))
            
        for i in range(iters):
            
            self.iter += 1
            
            grad_se_, grad_mine_, frames = grad_SE, grad_mine, frames_list[np.random.randint(len(frames_list))]
            P, RT = None, None 
            self.max_z = self.bu_max_z

            if valid_ratio > 0:
                min_idx, max_idx = 0, int(len(frames) * (1-valid_ratio))
            else:
                min_idx, max_idx = 0, int(len(frames))
                
            #
            # GRADIENT DESCENT
            loss, mi_lb, acc, pred = self.step(frames, 
                                               min_idx, np.min([max_idx, len(frames)]), self.mi_batch_size,
                                               grad_se=grad_se_, grad_T=grad_T, grad_R=grad_R, grad_mine=grad_mine_, RT=RT, P=P)
            
            #
            # WEIGHT POOL OF MINE
            if sample_weight_pool:
                # self.MINE.load_state_dict(torch.load(weight_pool_root + "/" + pool[np.random.randint(len(pool))]))
                self.MINE.load_state_dict(torch.load(weight_pool_root + "/" + pool[i % len(pool)]))
            
            
            if record_weight_pool and grad_mine_:
                if not os.path.exists(weight_pool_root):
                    os.makedirs(weight_pool_root)
                torch.save(self.MINE.state_dict(), weight_pool_root + "/MINE_" + str(i) + ".pth.tar")
            
            
            if i % log_interval == 0 and log_interval > 0 and RT == None:
                
                # CHOOSE frames
                frames = frames_list[i % len(frames_list)]
                
                avg_RT = self.RT()
                if len(self.histor_rt) > log_avg_window:
                    for l in range(1, log_avg_window):
                        avg_RT += self.histor_rt[-l]
                    avg_RT = avg_RT / log_avg_window
                
                # METRICS
                l2 = np.linalg.norm(self.initRT.cpu().numpy() - avg_RT)
                t_l2 = np.linalg.norm(avg_RT[:3, 3] - self.initRT.cpu().numpy()[:3, 3])
                R_xyz = list(np.round(euler_diff(A=self.initRT.cpu().numpy()[:3, :3], B=avg_RT[:3, :3]), 2))
                Q_cos = quat_diff(A=self.initRT.cpu().numpy()[:3, :3], B=avg_RT[:3, :3])
                eps_ = np.abs(self.l2_old - l2)
                
                #
                # LOG THE PROGRESS
                self.history.append((i, loss.detach().item(), mi_lb.detach().item(), acc.detach().item(), 
                                l2, t_l2, Q_cos, R_xyz[0], R_xyz[1], R_xyz[2]))
                
                
                self.histor_rt.append(self.RT())
                
                
                tst_loss = self.test(frames, iters=n_test_batches, viz=viz, log_interval=-1, 
                                     min_idx=np.min([max_idx, len(frames)-1]), max_idx=len(frames), 
                                     RT=RT, P=P)
                
                
                self.history_test.append(np.hstack([np.array([i]).reshape(1, -1), 
                                               tst_loss.mean(0, keepdims=True), 
                                               np.array([l2 ,t_l2, Q_cos ]).reshape(1,-1), 
                                               np.asarray(R_xyz).reshape(1,-1) ]))
                
                history_test = np.concatenate(self.history_test, axis=0)
                _, train_loss, train_mi, train_acc, _, _, _, _, _, _ = np.asarray(self.history)[-log_avg_window:].mean(0)
                _, val_loss,   val_mi,   val_acc, _, _, _, _, _, _   = history_test[-log_avg_window:].mean(0)
              
                
                print(f"#{i:4}: nll: {train_loss:.2f}/{val_loss:.2f}, mi: {train_mi: .3f}/{val_mi: .3f}, acc: {train_acc*100:.1f}/{val_acc*100:.1f}, eps: {eps_:.5f}, R={R_xyz}, t={t_l2:.2f}, Q_cos={Q_cos:.2f}")
                
                
                
                if self.best_mi < val_mi:
                    fname = f"checkpoints/checkpoint_best_mi.pth.tar"
                    torch.save(self.state_dict(), fname)
                    print("Saved checkpoint.")                    
                    self.best_mi = val_mi
                    
                if checkpoints_log_path:
                    path = f"checkpoints/{checkpoints_log_path}"
                    if not os.path.exists(path):
                        os.makedirs(path)
                    fname = f"{path}/checkpoint_{self.iter}.pth.tar"
                    torch.save(self.state_dict(), fname)
                    
                if eps_ < eps:
                    print(f"Fitting has converged { eps_} < {eps}")
                    break
                    
                if history_test[-self.patience:].max(0)[2] < self.best_mi:
                    print("Reload checkpoint and flush.")
                    self.load_state_dict(torch.load("checkpoints/checkpoint_best_mi.pth.tar"))
                    self.update_se_and_mine(None)
                    self.R.weight.data = self.R.weight.data + torch.normal(0, 0.00001, self.R.weight.data.shape).to(device)
                    self.T.bias.data = self.T.bias.data + torch.normal(0, 0.00001, self.T.bias.data.shape).to(device)
                    self.history = self.history[:-self.patience]
                    self.history_test = self.history_test[:-self.patience]
                    # return None
                    if self.c_steps > 0:
                        self.c_steps -= 1
                    else:
                        self.c_steps = self.n_steps
                        break

                   
                    
                self.l2_old = l2
                
                                    
        
        names_trn = ["iter", "loss", "mi", "acc", "l2", "t_l2", "Q_cos", "Rx", "Ry", "Rz"]
        names_tst = ["iter", "loss", "mi", "acc", "l2", "t_l2", "Q_cos", "Rx", "Ry", "Rz"]
        df_history_train = pd.DataFrame(np.asarray(self.history), columns=names_trn)
        df_history_test = pd.DataFrame(np.concatenate(self.history_test, axis=0), columns=names_tst)
        
        torch.save(self.state_dict(), "checkpoints/checkpoint_last.pth.tar")
        
        return df_history_train, df_history_test
    
    
