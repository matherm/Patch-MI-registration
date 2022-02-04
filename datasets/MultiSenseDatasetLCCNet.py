import imageio
from PIL import Image
import glob
import h5py
import numpy as np
from PIL import Image
from matplotlib import cm
import cv2
from collections import namedtuple

import torchvision.transforms.functional as TTF
import torch
from torchvision import transforms
import os
import csv



def dot_size(coords, img_shape, size=3):
   half_size = size // 2
   mask = np.all([coords[:,0]-half_size > 0, coords[:,1]-half_size > 0, coords[:,1]+half_size < img_shape[0], coords[:,0]+half_size < img_shape[1]],axis=0)
   range_coords = coords[mask]

   a = np.linspace(-half_size, half_size, size)
   b = np.linspace(-half_size, half_size, size)
   x, y = np.meshgrid(a, b)
   kernel = np.array([x.flatten(), y.flatten()]).T

   range_coords = range_coords.reshape(range_coords.shape[0], 1, 2)
   range_coords = np.repeat(range_coords, [size*size], axis=0).reshape(range_coords.shape[0], size*size, 2) + kernel
   return range_coords.reshape(range_coords.shape[0] * range_coords.shape[1], 2).astype('int')

class MultiSenseData():
    
    def __init__(self, base_path, rectified=True, crop=[0, 0, 600, 960], full=False, max_r = 2.0, max_t = 0.001, gt_RT=None):
    
        self.rectified = rectified
        self.crop = crop
        
        self.base_path = base_path
        self.filenames = sorted(glob.glob(self.base_path + "/*.hdf5"))
        
        # Switch return values of get_item()
        self.full = full
        
        # Get calibration data
        self.calib =  {}
        self.load_calib(h5py.File(self.filenames[0]))
        
        # LCCNet
        self.gt_RT = gt_RT
        self.max_r = max_r
        self.max_t = max_t
        amt_files = len(self)
        self.val_RT = []
        print("Generating a new test set")
        val_RT_file = os.path.join(self.base_path, f'val_RT_left_{self.max_r:.2f}_{self.max_t:.2f}.csv')
        val_RT_file = open(val_RT_file, 'w')
        val_RT_file = csv.writer(val_RT_file, delimiter=',')
        val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
        for i in range(5):
          rotz = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
          roty = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
          rotx = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
          transl_x = np.random.uniform(-self.max_t, self.max_t)
          transl_y = np.random.uniform(-self.max_t, self.max_t)
          transl_z = np.random.uniform(-self.max_t, self.max_t)
          # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
          for j in range(amt_files):
            val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                   rotx, roty, rotz])
            self.val_RT.append([float(i), transl_x, transl_y, transl_z,
                                 rotx, roty, rotz])

        assert len(self.val_RT) == amt_files*5, "Something wrong with test RTs"
        
        
                             
    def __repr__(self):
        return f"MultiSenseData(base_path={self.base_path})[{len(self)}]"
    
    def __len__(self):
        return 5 * len(self.filenames)
    
    # from LCCNet
    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
        rgb = color_transform(rgb)
        if flip:
            rgb = TTF.hflip(rgb)
        rgb = TTF.rotate(rgb, img_rotation)
        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb
    
    
    def load_calib(self, f):
        left_D = f['left_image/cam_info/D'][:]
        left_K = f['left_image/cam_info/K'][:].reshape(3,3)
        left_P = f['left_image/cam_info/P'][:].reshape(3,4)
        left_R = f['left_image/cam_info/R'][:].reshape(3,3)
        
        right_D = f['right_image/cam_info/D'][:]
        right_K = f['right_image/cam_info/K'][:].reshape(3,3)
        right_P = f['right_image/cam_info/P'][:].reshape(3,4)
        right_R = f['right_image/cam_info/R'][:].reshape(3,3)
        
        R_velo_cam = f["pointcloud/transforms/velodyne_to_camera_left"]["R"][:]
        T_velo_cam = f["pointcloud/transforms/velodyne_to_camera_left"]["t"][:]
        
        R_caml_camr = f["pointcloud/transforms/camera_left_to_camera_right"]["R"][:]
        T_caml_camr = f["pointcloud/transforms/camera_left_to_camera_right"]["t"][:]
        
        T_camleft_velo = np.zeros((4,4))
        T_camleft_velo[:3,:3] = R_velo_cam.T
        T_camleft_velo[:3, 3] = np.dot(-R_velo_cam.T, T_velo_cam)
        T_camleft_velo[3, 3]   = 1
        
        T_camleft_camright = np.zeros((4,4))
        T_camleft_camright[:3,:3] = R_caml_camr.T
        T_camleft_camright[:3, 3] = np.dot(-R_caml_camr.T, T_caml_camr)
        T_camleft_camright[3, 3]   = 1
        
        Calib = namedtuple("Calib", 'T_cam2_velo ' \
                                    'T_caml_camr ' \
                                    'P_rect_20 P_rect_30 ' \
                                    'R_rect_20 R_rect_30 ' \
                                    'K_rect_20 K_rect_30 ' \
                                    'D_rect_20 D_rect_30')
        
        self.calib = Calib(T_camleft_velo, 
                           T_camleft_camright,
                           left_P, right_P,
                           left_R, right_R,
                           left_K, right_K,
                           left_D, right_D)
        
    def project_to_left(self, pc, cam):
        point_cloud_h = np.hstack([point_cloud[:, :3], 
                                   np.ones((len(point_cloud), 1))])
        
        pc_in_image = (self.calib.P_rect_20 @ self.calib.T_cam2_velo @ point_cloud_h.T).T
        
        euclidean = np.true_divide(pc_in_image[:,:2], pc_in_image[:, -1:])
        img_coords = euclidean.astype(np.uint16)
        
        height = cam.shape[0]
        width = cam.shape[1]
        mask1 = np.all([img_coords[:,0] >= 0, img_coords[:,0] < width] , axis = 0)
        mask2 = np.all([img_coords[:,1] >= 0 ,img_coords[:,1] < height] , axis = 0)
        mask = np.all([mask1, mask2], axis=0)
        img_coords = img_coords[mask] 
        
        img_coords = dot_size(img_coords, cam.shape, size=3)
        img_coords = np.transpose(img_coords, (1,0))
        cam[img_coords[1], img_coords[0]] = [0,0,1]
        
        return cam
            
    def __getitem__(self, idx):
        
        idx = idx % len(self.filenames)
        #idx = len(self) - idx - 1
        
        f = h5py.File(self.filenames[idx])
        
        self.load_calib(f)
        
        cam_left = f['left_image/image'][:]
        cam_left = np.uint8(cam_left / np.iinfo("uint16").max * np.iinfo("uint8").max)
        
        
        cam_right = f['right_image/image'][:]
        cam_right = np.uint8(cam_right  / np.iinfo("uint16").max * np.iinfo("uint8").max)
        
        if self.rectified:
            map1_left, map2_left = cv2.initUndistortRectifyMap(self.calib.K_rect_20, 
                                                               self.calib.D_rect_20, 
                                                               self.calib.R_rect_20, 
                                                               self.calib.P_rect_20, 
                                                               (cam_left.shape[1], cam_left.shape[0]), 
                                                                cv2.CV_32FC1)
            cam_left = cv2.remap(cam_left, map1_left, map2_left, cv2.INTER_CUBIC).astype('uint8')

            map1_right, map2_right = cv2.initUndistortRectifyMap(self.calib.K_rect_30, 
                                                               self.calib.D_rect_30, 
                                                               self.calib.R_rect_30, 
                                                               self.calib.P_rect_30, 
                                                               (cam_left.shape[1], cam_left.shape[0]), 
                                                                cv2.CV_32FC1)
            cam_right = cv2.remap(cam_right, map1_right, map2_right, cv2.INTER_CUBIC).astype('uint8')
            
        else:
            cam_left = cv2.undistort(cam_left, self.calib.K_rect_20, self.calib.D_rect_20)
            cam_right = cv2.undistort(cam_right, self.calib.K_rect_30, self.calib.D_rect_30)
        
        
        point_cloud = f["pointcloud/points"][:]
        ring = point_cloud["ring"]
        point_cloud = np.stack([point_cloud["x"], 
                                point_cloud["y"], 
                                point_cloud["z"], 
                                point_cloud["intensity"]]).T
        
        # remove points from the boat and from behind
        point_cloud = point_cloud[point_cloud[:, 0] > 4] 
        
        tly, tlx, bry, brx = self.crop
        cam_left = cam_left[tly:bry, tlx:brx]
        cam_right = cam_right[tly:bry, tlx:brx]
        if not self.full:
            return point_cloud, Image.fromarray(cam_left), Image.fromarray(cam_right), None
        
        # LCCNet output from here
        else:
            # Import necessary stuff
            import mathutils
            import importlib.util
            spec = importlib.util.spec_from_file_location("module.name", "/home/ios/GitHub/LCCNet/utils.py")
            utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils)

            
            img = Image.fromarray(cam_left)
            try:
                img = self.custom_transform(img)
            except OSError:
                new_idx = np.random.randint(0, self.__len__())
                return self.__getitem__(new_idx)
            
            pc_org = torch.from_numpy(point_cloud.astype(np.float32))
            
            if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
                pc_org = pc_org.t()
            if pc_org.shape[0] == 3:
                homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
                pc_org = torch.cat((pc_org, homogeneous), 0)
            elif pc_org.shape[0] == 4:
                if not torch.all(pc_org[3, :] == 1.):
                    pc_org[3, :] = 1.
            else:
                raise TypeError("Wrong PointCloud shape")
            
            img_path = os.path.join(self.base_path, 'left_image', 'image', str(idx),'cam_left.jpg')
            pc_lidar = point_cloud.reshape((-1, 4)).copy()
            
                        
            rotz = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
            roty = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
            rotx = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            
            R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
            T = mathutils.Vector((transl_x, transl_y, transl_z))
            
            R, T = utils.invert_pose(R, T)
            R, T = torch.tensor(R), torch.tensor(T)
                                
            #initialRT = self.val_RT[idx]
            initialRT = self.calib.T_cam2_velo.copy().astype(np.int64)
            initialRT[:3, :3] = np.eye(3)
            initialRT = [0, initialRT[0, 3], initialRT[1, 3], initialRT[2, 3], 0, 0, 0]
            
            E_RT = self.calib.T_cam2_velo.copy().astype(np.float32)
            if self.gt_RT:
                E_RT = np.load(self.gt_RT)
            pc_rot = np.matmul(E_RT, pc_org.numpy())
            pc_rot = pc_rot.astype(np.float32).copy()
            pc_in = torch.from_numpy(pc_rot)
            
            calib = self.calib.P_rect_20[:3, :3]
            
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
                  'tr_error': T, 'rot_error': R, 'rgb_name': 'cam_left' + str(idx) +'.png', 'item': img,
                  'extrin': E_RT, 'initial_RT': initialRT, 'pc_lidar': pc_lidar}

            return sample