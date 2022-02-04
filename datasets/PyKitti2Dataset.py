import numpy as np
from collections import namedtuple
import glob, os
from pykitti.utils import load_oxts_packets_and_poses
import pykitti
from pykitti.tracking import KittiTrackingLabels
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
import cv2

def rt_to_se3(rvecs, tvecs):
    RT = np.zeros((4,4))
    RT[:3, :3] = cv2.Rodrigues(rvecs)[0]
    RT[:3, 3] = tvecs[:, 0]
    RT[3, 3] = 1
    return RT

def show_pair(volumes, patches, idx):
        im, pc = patches[idx], volumes[idx]
        print(im.shape)
        if len(im.flatten()) > 0:
            # meshplot is looking down negative z
            x, y, z = pc.T 
            pc = np.stack([x, z, -y]).T
            plot = meshplot.plot(pc, c=np.arange(len(pc)), shading={"point_size": 0.8, "width": 200, "height": 200})        
            plt.imshow(im)
            plt.axis("off")
            plt.show()   
        else:
            print("Image patch too small.")

def to_homo(pts):
    if pts.shape[1] == 3:
        return np.hstack([pts, np.ones((len(pts), 1))])
    else:
        return pts

def get_iis_keypoints(pc, salient_radius=0.03,non_max_radius=0.03, max_distance=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                                            salient_radius=salient_radius,
                                                            non_max_radius=non_max_radius,
                                                            gamma_21=0.5,
                                                            gamma_32=0.5)
    keypoints = np.asarray(keypoints.points)
    keypoints = keypoints[np.linalg.norm(keypoints - keypoints.mean(0), axis=1) <= max_distance] # y-coord
    return keypoints


def lookAt(target, center=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    f = (target - center); f = f/np.linalg.norm(f)
    s = np.cross(f, up); s = s/np.linalg.norm(s)
    u = np.cross(s, f); u = u/np.linalg.norm(u)

    m = np.zeros((3, 3))
    m[0, :] = s
    m[1, :] = f
    m[2, :] = u
    
    return m.T


def get_volumes(keypoints, pc, radius=1):
    rnr = NearestNeighbors(radius=radius)
    rnr.fit(pc)
    points, idx = rnr.radius_neighbors(keypoints)
    locs, volumes = [], []
    for i in range(len(idx)):
        points = pc[np.asarray(idx[i])]
        mean = keypoints[i]
        locs.append(mean)
        #R = lookAt(mean) 
        #points = (R @ points.T).T
        points -= mean
        volumes.append(points)
    return volumes, np.asarray(locs)

def get_sift_keypoints(im, sift=None, min_scale = 1, max_scale = 2):
    assert im.shape[2] == 3

    def unpackSIFTOctave(kpt):
        """unpackSIFTOctave(kpt)->(octave,layer,scale)
        @created by Silencer at 2018.01.23 11:12:30 CST
        @brief Unpack Sift Keypoint by Silencer
        @param kpt: cv2.KeyPoint (of SIFT)
        """
        _octave = kpt.octave
        octave = _octave&0xFF
        layer  = (_octave>>8)&0xFF
        if octave>=128:
            octave |= -128
        if octave>=0:
            scale = float(1/(1<<octave))
        else:
            scale = float(1<<-octave)
        return (octave, layer, scale)
    
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    kp = sift.detect(gray, None)
    kp = np.asarray([(*unpackSIFTOctave(kpt), int(kpt.pt[0]),int(kpt.pt[1]))  for kpt in kp])
    kp = kp[kp.T[2] >= 1]  # octave, layer, scale, x, y
    kp = kp[kp.T[2] <= 2]  # octave, layer, scale, x, y
    return kp
    

def get_patches(keypoints, IM, basic_shape=(256, 256)):
    locs, patches = [], []
    for kp in keypoints:
        octave, layer, scale, x, y = kp 
        x, y = int(x), int(y)
        locs.append((x, y))
        shp = int(basic_shape[0] / scale // 2)
        t,b,l,r = y-shp, y+shp, x-shp, x+shp
        t = np.clip(t, 0, IM.shape[0])
        b = np.clip(b, 0, IM.shape[0])
        l = np.clip(l, 0, IM.shape[1])
        r = np.clip(r, 0, IM.shape[1])
        patch = IM[t:b,l:r]    
        patches.append(patch)
    return patches, np.asarray(locs)

        
        
def RT_from_ICP(src, trg, threshold = 2.0):

    # remove floor
    src = src[src[:, 2] > 0.1]
    trg = trg[trg[:, 2] > 0.1]

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(trg)

    trans_init = np.zeros((4,4))
    trans_init[:3, :3] = np.eye(3)
    trans_init[:3, 3] = np.zeros(3)
    trans_init[3, 3] = 1

    reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # print(reg_p2p)
    return reg_p2p.transformation

class PyKitti2(pykitti.tracking):  
    
    """
    Sequences with same calibration data:
        * [0-13], 
        * [14-17], 
        * [18-19], 
        * [20]
    
    
    """
    
    SEQUENCES = [f"000{i}" for i in range(10)] + [f"00{i}" for i in range(10, 20)]
    
    def __init__(self, base_path, sequence, with_labels="none"):
        super().__init__(base_path, sequence=sequence)
        
        assert with_labels in ["none", "mask", "box"]
        
        self.with_labels = with_labels
        self.oxts_files = sorted(glob.glob(os.path.join(self.base_path, 'oxts', '{}.{}'.format(self.sequence, "txt"))))
        self.image_files = sorted(glob.glob(os.path.join(self.base_path, 'image_02/{}/*.png'.format(self.sequence))))
        self.pc_files = sorted(glob.glob(os.path.join(self.base_path, 'velodyne/{}/*.bin'.format(self.sequence))))
        self.oxts = load_oxts_packets_and_poses(self.oxts_files)
        # self.labels = KittiTrackingLabels(base_path + f"/label_02/{sequence}.txt")
        self.labels = KittiTrackingLabels(os.path.join(base_path,"label_02",str(sequence) + ".txt"))
        KittiTrackingLabels.classes += ["Person"]

        self._load_calib()
    
    def __repr__(self):
        return f"PyKitti2(base_path={self.base_path}, sequence={self.sequence})[{len(self)}]"
    
    def __len__(self):
        return np.min([len(self.oxts), len(self.image_files),len(self.pc_files), len(self.labels)])
    
    def __getitem__(self, idx):
        if self.with_labels == "box":
            return  (self.get_velo(idx), 
                     self.get_cam2(idx), 
                     self.get_cam3(idx), 
                     self.oxts[idx], 
                     self.labels.bbox[idx], 
                     self.labels.cls[idx])
        
        if self.with_labels == "mask":
            velo, cam2, cam3, oxts = self.get_velo(idx), self.get_cam2(idx), self.get_cam3(idx), self.oxts[idx]
            boxes = self.labels.bbox[idx]
            cls = self.labels.cls[idx]
            label_im = self.make_label_image(np.asarray(cam2).copy()[:,:,0] * 0, boxes, cls)
            label_pc = self.make_label_pc(label_im, velo)
            return (self.get_velo(idx), 
                    self.get_cam2(idx), 
                    self.get_cam3(idx), 
                    self.oxts[idx],
                    label_im,
                    label_pc)
        
        return self.get_velo(idx), self.get_cam2(idx), self.get_cam3(idx), self.oxts[idx]
    
    def make_label_pc(self, label_im, velo):
        x, y, z, i, coords, idxes = self.veloTocam2(velo)
        
        # some label images are smaller than the camera image
        mask_y = y < label_im.shape[0]
        idxes = idxes[mask_y]
        y     = y[mask_y]
        x     = x[mask_y]
        
        # some label images are smaller than the camera image
        mask_x = x < label_im.shape[1]
        idxes = idxes[mask_x]
        y     = y[mask_x]
        x     = x[mask_x]
   
        cls = label_im[y, x]
        label_pc = np.zeros(len(velo)) - 1
        label_pc[idxes] = cls
        return label_pc
            
        
    def make_label_image(self, image, boxes, cls):
        for i,box in enumerate(boxes):
            x1,y1, x2, y2 = box.astype(int)
            image[y1:y1+y2, x1:x1+x2] = KittiTrackingLabels.classes.index(cls[i])
        return image
    
    def build_point_cloud(self, idx_start=0, idx_end=5):
        """
        Returns a merged pointcloud xyz for the given indices.
        """

        the_world, intensities, batch, RTs = None, None, None, []
        init_points = 0

        for i in range(idx_start, idx_end):

            # Grab the data
            PC, R, T, RT = self[i][0], self[i][3].T_w_imu[:3, :3], self[i][3].T_w_imu[:3, 3], self[i][3].T_w_imu

            # replace intensity by homgeneous 1
            PC = to_homo(PC[:, :3])

            # GoTo orign
            #PC_origin = (R.T @ ((PC[:, :3].T).T  - T).T).T
            PC_origin = (np.linalg.inv(RT) @ PC.T).T
            
            # Build the world
            if the_world is None:
                the_world = PC_origin
                intensities = PC[:, 3]
                batch = np.ones(len(PC_origin)) * i
                init_points = len(the_world)
                RTs.append(RT)
            else:
                RT01 = RT_from_ICP(the_world[:init_points, :3], PC_origin[:, :3])
                PC_origin = (np.linalg.inv(RT01) @ PC_origin.T).T
                the_world = np.vstack([the_world, PC_origin])
                intensities = np.concatenate([intensities, PC[:, 3]])
                batch = np.concatenate([batch, np.ones(len(PC_origin)) * i])
                RTs.append(np.linalg.inv( np.linalg.inv(RT01) @ np.linalg.inv(RT)))
                

        return the_world[:, :3], intensities, batch, RTs
    
    def _load_calib(self):
            """Load and compute intrinsic and extrinsic calibration parameters."""
            # We'll build the calibration parameters as a dictionary, then
            # convert it to a namedtuple to prevent it from being modified later
            data = {}

            # Load the calibration file
            #calib_filepath = os.path.join(self.sequence_path + '.txt', 'calib.txt')
            #calib_filepath = self.base_path + "/calib/" + self.sequence + ".txt"
            calib_filepath = os.path.join(self.base_path, "calib", str(self.sequence) + ".txt")

            def read_calib_file(filepath):
                """Read in a calibration file and parse into a dictionary."""
                data = {}

                with open(filepath, 'r') as f:
                    for line in f.readlines():
                        try:
                            key, value = line.split(':', 1)
                        except:
                            key, value = line.split(' ', 1)                        
                        # The only non-float values in these files are dates, which
                        # we don't care about anyway
                        try:
                            data[key] = np.array([float(x) for x in value.split()])
                        except ValueError:
                            pass

                return data

            filedata = read_calib_file(calib_filepath)

            # Rectification
            data['R_rect'] = np.zeros((4,4))
            data['R_rect'][3,3] = 1
            data['R_rect'][:3,:3] = filedata['R_rect'].reshape(3,3)

            # Create 3x4 projection matrices
            P_rect_00 = np.reshape(filedata['P0'], (3, 4))
            P_rect_10 = np.reshape(filedata['P1'], (3, 4))
            P_rect_20 = np.reshape(filedata['P2'], (3, 4))
            P_rect_30 = np.reshape(filedata['P3'], (3, 4))

            data['P_rect_00'] = P_rect_00
            data['P_rect_10'] = P_rect_10
            data['P_rect_20'] = P_rect_20
            data['P_rect_30'] = P_rect_30

            # Compute the rectified extrinsics from cam0 to camN
            T1 = np.eye(4)
            T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
            T2 = np.eye(4)
            T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
            T3 = np.eye(4)
            T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

            # Compute the velodyne to rectified camera coordinate transforms
            data['T_cam0_velo'] = np.reshape(filedata['Tr_velo_cam'], (3, 4))

            # Adds rectification to the points in camera coords
            data['T_cam0_velo'] = data['R_rect'].dot(np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]]))

            # Skip the extrinsics and we are using the projection matrices directly
            #data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
            #data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
            #data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])
            data['T_cam1_velo'] = data['T_cam0_velo']
            data['T_cam2_velo'] = data['T_cam0_velo']
            data['T_cam3_velo'] = data['T_cam0_velo']

            # Velodyne to IMI
            data["Tr_imu_velo"] = np.reshape(filedata['Tr_imu_velo'], (3, 4))

            # Compute the camera intrinsics
            data['K_cam0'] = P_rect_00[0:3, 0:3]
            data['K_cam1'] = P_rect_10[0:3, 0:3]
            data['K_cam2'] = P_rect_20[0:3, 0:3]
            data['K_cam3'] = P_rect_30[0:3, 0:3]

            # Compute the stereo baselines in meters by projecting the origin of
            # each camera frame into the velodyne frame and computing the distances
            # between them
            p_cam = np.array([0, 0, 0, 1])
            p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
            p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
            p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
            p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

            data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
            data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline    

            self.calib = namedtuple('CalibData', data.keys())(*data.values())
            
            
    def veloToX(self, velo, T_cam_velo, P_rect, im_shape=(375, 1242, 3)):
            max_x, max_y = im_shape[1], im_shape[0]

            # homogenous coords
            velo = to_homo(velo)
            coords, intensities = velo[:, :3], velo[:, 3]
        
            # project to camera
            velo_in_cam = T_cam_velo.dot(to_homo(coords).T).T

            # project to image plane
            velo_in_cam_im = P_rect.dot(velo_in_cam.T).T
            
            # clip points behind camera
            idx = np.arange(len(velo_in_cam_im))
            mask_behind_cam = velo_in_cam_im[:, 2] > 1.
            velo_in_cam_im, z, i, coords, idx = (velo_in_cam_im[mask_behind_cam], 
                                                 velo_in_cam_im[:, 2][mask_behind_cam], 
                                                 intensities[mask_behind_cam],
                                                 coords[mask_behind_cam],
                                                 idx[mask_behind_cam])

            # z-normalize
            velo_in_cam_im /= z[:, None]
            velo_in_cam_px = velo_in_cam_im[:, :2].astype(np.int32)
                
            # image frustum
            mask_frustum_x = velo_in_cam_px[:, 0] < max_x
            velo_in_cam_px, z, i, coords, idx = (velo_in_cam_px[mask_frustum_x], 
                                     z[mask_frustum_x], 
                                     i[mask_frustum_x],
                                     coords[mask_frustum_x],
                                     idx[mask_frustum_x])
            mask_frustum_y = velo_in_cam_px[:, 1] < max_y
            velo_in_cam_px, z, i, coords, idx = (velo_in_cam_px[mask_frustum_y], 
                                     z[mask_frustum_y], 
                                     i[mask_frustum_y],
                                     coords[mask_frustum_y],
                                     idx[mask_frustum_y])
            mask_frustum_x0 = velo_in_cam_px[:, 0] > 0
            velo_in_cam_px, z, i, coords, idx = (velo_in_cam_px[mask_frustum_x0], 
                                     z[mask_frustum_x0], 
                                     i[mask_frustum_x0],
                                     coords[mask_frustum_x0],
                                     idx[mask_frustum_x0])
            mask_frustum_y0 = velo_in_cam_px[:, 1] > 0
            velo_in_cam_px, z, i, coords, idx = (velo_in_cam_px[mask_frustum_y0], 
                                     z[mask_frustum_y0], 
                                     i[mask_frustum_y0],
                                     coords[mask_frustum_y0],
                                     idx[mask_frustum_y0])

            # turn to integer pixels
            x, y = velo_in_cam_px[:, 0], velo_in_cam_px[:, 1]

            return x, y, z, i, coords, idx
        
    def veloTocam2(self, velo, im_shape=(375, 1242, 3)):            
            return self.veloToX(velo, self.calib.T_cam2_velo,  self.calib.P_rect_20)

    def veloTocam3(self, velo, im_shape=(375, 1242, 3)):            
            return self.veloToX(velo, self.calib.T_cam3_velo,  self.calib.P_rect_30)
             
    
    def _match_keypoints_image_pc(self, im_xy, pc_xy, radius_tresh=3):

        # Compute pairwise distances
        dists = euclidean_distances(im_xy, pc_xy)

        # Build index correspondences vectors
        trgt_idx = np.argmin(dists, axis=1) # (#sift, 1)
        src_idx = np.arange(len(dists))

        # Threshold minimum distances
        min_dists = dists[src_idx, trgt_idx] # (#sift, 1)
        src_idx = src_idx[min_dists <= radius_tresh]
        trgt_idx = trgt_idx[min_dists <= radius_tresh]

        # Return matches
        return np.stack([src_idx, trgt_idx])
    
    
    def compute_matches_cam2(self, idx_start=0, idx_end=5, px_tresh=3, radius=1, min_points=30, max_distance=20, min_occurences=3, debug=False):

        # Integrate the world
        the_world, intensities, batch, RTs = self.build_point_cloud(idx_start, idx_end)

        # Compute the 3d keypoints
        kps_3d = get_iis_keypoints(the_world, salient_radius=0.7, non_max_radius=1.5, max_distance=max_distance)
        kps_3d_h = to_homo(kps_3d)

        # track matches and keypoints
        matches, kps_2d, images  = [] , [], []

        # Iterate over the frames, project the keypoints and measure pixel distances
        for j,i in enumerate(range(idx_start, idx_end)):
            velo, cam2, cam3, oxts = self[i] 
            cam2 = np.asarray(cam2)
            RT = RTs[j]

            # Compute the keypoints
            kps_3d_in_cam = (RT @ kps_3d_h.T).T
            x, y, z, ii, kps_3d_, idx = self.veloTocam2(kps_3d_in_cam)
            pc_xy = np.stack([x, y]).T

            kps_2d_ = get_sift_keypoints(cam2)
            im_xy = kps_2d_[:, 3:5] # octave, layer, scale, x, y

            # Compute matches
            src, trgt = [], []
            if len(im_xy) > 0 and len(pc_xy) > 0:
                matches_ = self._match_keypoints_image_pc(im_xy, pc_xy, radius_tresh=px_tresh)
                src, trgt = matches_
                trgt = idx[trgt]

            # Update matches
            matches.append((src, trgt))
            kps_2d.append(kps_2d_)
            images.append(cam2)

        
        if debug:
            print(f"Found {len(np.concatenate(matches, axis=1).T)} matches.")
        
        # Filter out 3d keypoints with less than 'min_occurences' views
        srcs, trgts = np.concatenate(matches, axis=1).astype(np.int32)
        idx, counts = np.unique(trgts, return_counts=True) # matches are src->trgt, where src is the image
        valid_trgts = idx[counts >= min_occurences]
        
        if debug:
            print(f"Found {len(valid_trgts)} keypoints that occur in {min_occurences} patches.")
        
        volumes, patches, ilocs, pclocs, frame_idx, affinity = [], [], [], [], [], []
        if len(valid_trgts) > 0:

            # Filter out nearby 3d keypoints
            rnr = NearestNeighbors(radius=radius)
            rnr.fit(kps_3d[valid_trgts])
            points, idx = rnr.radius_neighbors(kps_3d[valid_trgts])
            mask = np.asarray([True] * len(valid_trgts))
            for i, neighbors in enumerate(idx):
                if mask[i]:
                    for neighbor in neighbors:
                        mask[neighbor] = False
                    mask[i] = True
            valid_trgts = valid_trgts[mask]


            if debug:
                print(f"Found {len(valid_trgts)} keypoints without nearby keypoints.")


            # Computing the volumes and patches for each frame
            for j, i in enumerate(range(idx_start, idx_end)):

                # matches
                src, trgt = matches[j]

                if len(trgt) > 0 and len(src) > 0:

                    # filter out unvalid keypoints 
                    mask = np.isin(trgt, valid_trgts)

                    trgt = trgt[mask]
                    src = src[mask]

                    subcloud = the_world[batch == i]

                    if len(subcloud) > 1 and len(trgt) > 0 and len(src) > 0:

                        # Compute volumes    
                        volumes_, pc_locs_ = get_volumes(kps_3d[trgt], subcloud, radius=radius)

                        # Compute patches
                        image, kps_2d_ =  images[j], kps_2d[j] 
                        patches_, ilocs_ = get_patches(kps_2d_[src], image, basic_shape=(256, 256))

                        # Update stack
                        volumes = volumes + volumes_
                        patches = patches + patches_
                        ilocs.append(ilocs_)
                        pclocs.append(pc_locs_)
                        frame_idx.append(np.ones(len(volumes)).astype(np.int32) * i)
                        affinity.append(np.stack([src, trgt]))


            # Preparing output
            pclocs = np.concatenate(pclocs)
            ilocs = np.concatenate(ilocs)
            frame_idx = np.concatenate(frame_idx)
            affinity = np.concatenate(affinity, 1).T

            # Filter out volumens with too few points as given by 'min_points'
            idx_valid_volumes = [i for i,v in enumerate(volumes) if len(v) > min_points]    
            volumes = [volumes[i] for i in idx_valid_volumes] 
            pclocs = np.asarray([pclocs[i] for i in idx_valid_volumes] )
            patches = [patches[i] for i in idx_valid_volumes] 
            ilocs = np.asarray([ilocs[i] for i in idx_valid_volumes] )
            frame_idx = np.asarray([frame_idx[i] for i in idx_valid_volumes])
            affinity = np.asarray([affinity[i] for i in idx_valid_volumes]).T

            if debug:
                print(f"Found {len(volumes)} matches with minimum {min_points} points.")

        return volumes, pclocs, patches, ilocs, RTs, frame_idx, affinity
