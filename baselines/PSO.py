import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage, spatial
from itertools import product
import open3d as o3d
from timeit import default_timer as timer


class PSO():
    def __init__(self, n_particles, point_clouds, cam2s, P, randRoll, randPitch, randYaw, randTranslation, RT_gt, bins=256, max_z=10, modified=True, method="normals", filter=False):
        super().__init__()
        '''
        Variable initializations
        '''
        # Parameters
        self.n_particles = n_particles                      # amount of particles for pso
        if not isinstance(cam2s, Image.Image):
            self.point_clouds = np.asarray(point_clouds)    # point clouds to match
            tmp_imgList = []
            for img in cam2s:
                tmp_imgList.append(np.asarray(img))
            self.cam2s = np.asarray(tmp_imgList)            # camera images to numpy array
        else:
            self.cam2s = np.array([np.asarray(cam2s)])      # single camera image to numpy array
            self.point_clouds = np.asarray([point_clouds])  # single point cloud to numpy array

        self.RT_gt = RT_gt                                  # initial guess for RT-matrix
        self.P = P                                          # projection matrix
        self.bins = bins                                    # number of bins for histogram
        self.max_z = max_z                                  # max depth value
        self.clip_floor = -1                                # floor height to clip

        # Parameters for random RT matrix generation
        self.randRoll = randRoll
        self.randPitch = randPitch
        self.randYaw = randYaw
        self.randTranslation = randTranslation

        # Method setup
        self.modified = modified              # whether to use standard or modified implementation
        self.method = method                  # whether to use reflactance instead of normals
        self.filter = filter                  # whether to use gauss filter before histogram calculation

        # Normal estimation
        self.normals = np.array([])

        # Only for image plotting
        self.imgScaleFactor = 2

        # Hyperparameters
        self.c1 = 0.1
        self.c2 = 0.1 # cognitive and social factor constants
        self.w = 0.8 # inertial factor (Traegheit)

       
    '''
    Method declarations
    '''
    # Returns correct roll, pitch, yaw and translation vector
    def getInitialGuess(self, euler=False):
        rot = self.RT_gt.copy()[:3, :3]
        transl = self.RT_gt.copy()[:3, 3]
        if not euler:
             return self.RT_gt.copy()    
        else:
            roll, pitch, yaw = spatial.transform.Rotation.from_matrix(rot).as_euler('xyz', degrees=True)
            return (roll, pitch, yaw), transl

    
    # Makes rotation matrix from roll, pitch, yaw angles and translations in x, y, and z direction
    def makeRT(self, roll, pitch, yaw, x, y, z):
        R = spatial.transform.Rotation.from_euler('xyz', [roll,pitch,yaw], degrees=True).as_matrix()
        E = np.zeros((4,4))
        E[:3,:3] = R
        E[:3, 3] = [x,y,z]
        E[3,3] = 1
        return E
    
    # Makes rotation matrix from roll, pitch, yaw angles and translations in x, y, and z direction (vectorized)
    def makeRTVectorized(self, rpy, xyz):
        R = spatial.transform.Rotation.from_euler('xyz', rpy, degrees=True).as_matrix() # N x 3 x 3
        E = np.zeros((len(rpy), 4, 4))
        E[:, :3, :3] = R
        E[:, :3, 3] = xyz
        E[:, 3, 3] = 1
        return E
    
    
    # Calculates NMI for two discrete sources
    def mutual_information_2d(self, x, y, normalized=True, sigma=5):
        EPS = np.finfo(float).eps  # machine epsilon
    
        jh = np.histogram2d(x, y, bins=self.bins)[0]

        
        # smooth the jh with a gaussian filter of given sigma
        if self.filter:
            ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                     output=jh)
        
        # compute marginal histograms
        jh = jh + EPS
        sh = np.sum(jh)
        jh = jh / sh
        s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
        s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))
        
        # Normalised Mutual Information of:
        # Studholme,  jhill & jhawkes (1998).
        # "A normalized entropy measure of 3-D medical image alignment".
        # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
        if normalized:
            mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                    / np.sum(jh * np.log(jh))) - 1
        else:
            mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
                   - np.sum(s2 * np.log(s2)))
    
        return mi


    
    # Transforms Lidar points to the image plane
    def transform_points(self, point_cloud_, image, RT=None, P=None):
        """
        Projects the points cloud onto the image plane and clips invalid points outside the view frustum.
        """
        # Hypers
        P = torch.from_numpy(P.copy()).float()
        RT = torch.from_numpy(RT.copy()).float()
    
        device = "cpu"
    
        point_cloud__ = torch.from_numpy(point_cloud_[:, :3]).float().to(device)
        intensities = torch.from_numpy(point_cloud_[:, 3]).float().to(device)
        image = torch.from_numpy(np.asarray(image)/255.).float().transpose(1,2).transpose(0,1).unsqueeze(0).to(device)
        
        #variables
        max_x, max_y, max_z = image.shape[3], image.shape[2], self.max_z
    
        # transform
        point_cloud = point_cloud__[point_cloud__[:, 2] > self.clip_floor]
        intensities = intensities[point_cloud__[:, 2] > self.clip_floor]
     
        T_pointcloud = (RT[:3, :3] @ point_cloud.T).T + RT[:3, 3]
    
        T_pointcloud_h = torch.cat([T_pointcloud, (T_pointcloud[:, :1] * 0) + 1], axis=1).float()
    
        # project
        proj_pointcloud = (P @ T_pointcloud_h.T).T
    
        # clip points behind camera
        mask = proj_pointcloud[:, 2].detach().cpu().numpy() > 1.
        mask = np.logical_and(mask, proj_pointcloud[:, 2].detach().cpu().numpy() < max_z)
    
        # z-normalize
        z = proj_pointcloud[:, 2:].clone()
        proj_pointcloud /= z
    
        # image frustum
        mask = np.logical_and(mask, proj_pointcloud[:, 0].detach().cpu().numpy() < max_x)
        mask = np.logical_and(mask, proj_pointcloud[:, 1].detach().cpu().numpy() < max_y)
        mask = np.logical_and(mask, proj_pointcloud[:, 0].detach().cpu().numpy() > 0)
        mask = np.logical_and(mask, proj_pointcloud[:, 1].detach().cpu().numpy() > 0)
        mask = torch.from_numpy(mask).bool()
        
        
        valid_proj_pointcloud = proj_pointcloud[mask]
        intensities = intensities[mask].T
        masked_pointcloud = point_cloud[mask]
        x, y, ones = valid_proj_pointcloud.T
        
        # Project the points
        x, y, z = x.long().detach(), y.long().detach(), z.detach()[mask,0]
    
        return image, x,y,z, intensities, mask, max_x, max_y, max_z, valid_proj_pointcloud[: , :2]
    

    def estimateNormals(self, point_clouds, visualizePointCloud=False):
        normalsList = []
        for i in range(len(point_clouds)):
            # Create open3D PointCloud data
            pcd = o3d.geometry.PointCloud()
            
            data = point_clouds[i][:,:3]

            pcd.points = o3d.utility.Vector3dVector(data)

            # Estimate normals from PointCloud using open3D
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=8))
            
            # invert normals
            pcd.orient_normals_to_align_with_direction([-1, 0, 0])
            normalsList.append(np.asarray(pcd.normals))
            # Visualize PointCloud
            if visualizePointCloud:
                o3d.visualization.draw_geometries([pcd], width=1024, height=768, point_show_normal=True)
        # return normals
        return np.asarray(normalsList)
        
    
    def createProjectedImages(self, RT):
        # Transform points
        cam2_projList = []
        intensitiesList = []
        xList = []
        yList = []
        for i in range(len(self.point_clouds)):
            image, x, y, z, lidar_intensities, mask = self.transform_points(self.point_clouds[i], self.cam2s[i], RT, self.P)[:6]

            x, y, z = x.float().cpu().detach().numpy(), y.float().cpu().detach().numpy(), z.cpu().detach().numpy()
            x = x[z < self.max_z]
            y = y[z < self.max_z]
            z = z[z < self.max_z]
            z = -z

            # Cast x and y to integers to use as indices
            x =  np.floor(x).astype(int)
            y = np.floor(y).astype(int)

            # Projected intensity values from point cloud
            intensities = np.zeros((self.cam2s[0].shape[0], self.cam2s[0].shape[1]), dtype="uint8")

            if self.method == "euclidean":
                # using euclidean distances
                distances = self.point_clouds[i].copy()[:, :3]
                distances = distances[self.point_clouds[i][:, 2] > self.clip_floor]
                distances = distances[mask]
                distances = np.sqrt(distances[:, 0]*distances[:, 0] + distances[:, 1]*distances[:, 1] + distances[:, 2]*distances[:, 2])
                std = np.std(distances)
                distances = distances/std
                distances = ( distances - distances.min()) / ( distances.max() - distances.min() + 1e-8) * 255
                intensities[y, x] = distances

            elif self.method == "reflectance":
                # Using reflectance
                reflectance = self.point_clouds[i].copy()[:, 3]
                reflectance = reflectance[self.point_clouds[i][:, 2] > self.clip_floor]
                reflectance = reflectance[mask]
                reflectance = ( reflectance - reflectance.min()) / ( reflectance.max() - reflectance.min() + 1e-8) * 255
                intensities[y, x] = reflectance
            else:
                # Using normals
                point_cloud = self.point_clouds[i].copy()[:, :3]
                normals = self.normals[i].copy()[point_cloud[:, 2] > self.clip_floor]
                normals = normals[mask]
                angles = self.getAngleBetweenVectors(normals, np.full((len(normals), 3), [0, 0, -1]))
                angles = ( angles - angles.min()) / ( angles.max() - angles.min() + 1e-8) * 255
                intensities[y, x] = angles
            
            # Camera grayscale image (only projected pixels from point cloud)
            cam2_proj = np.zeros((self.cam2s[i].shape[0], self.cam2s[i].shape[1]), dtype="uint8")
            dot = self.cam2s[i].copy()[..., :3] @ [.2989, .5870, .1140]
            cam2_proj[y, x] = dot[y, x]

            cam2_projList.append(cam2_proj)
            intensitiesList.append(intensities)
            xList.append(x)
            yList.append(y)

        return np.asarray(cam2_projList), np.asarray(intensitiesList), np.asarray(xList), np.asarray(yList)


    # Returns NMI for 1 particle with given RT-matrix
    def nmi(self, RT):
        # create projected camera and intensity images
        cam2_projList, intensitiesList, xList, yList = self.createProjectedImages(RT)

        # filter out zeros
        cam2_projList_nonzero = cam2_projList[intensitiesList > 0]
        intensitiesList_grayscale_nonzero = intensitiesList[intensitiesList > 0]
        
        # calculate entropies and NMI
        nmi = self.mutual_information_2d(cam2_projList_nonzero, intensitiesList_grayscale_nonzero)
        
        if not self.modified:
            return nmi
        return nmi * (len(intensitiesList_grayscale_nonzero) / (self.cam2s[0].shape[1] * self.cam2s[0].shape[0]))
    
    
    def render(self, RT):
        cam2_projList, intensitiesList, xList, yList = self.createProjectedImages(RT)
        
        # Color the points for plotting
        thick = 1
        cam2_points = self.cam2s[0].copy()
        intensities = intensitiesList[0]
        x = xList[0]
        y = yList[0]
        for i, (x_, y_) in enumerate(product(range(-thick, thick), range(-thick, thick))):
                current_intensity = intensities[y, x]
                #cam2_points[y+y_, x+x_] = np.array([current_intensity, current_intensity, current_intensity]).T
                cam2_points[yList[0]+y_, xList[0]+x_] = 255
        
        plt.figure(figsize=(25, 10))
        plt.imshow(Image.fromarray(cam2_points).resize((self.cam2s[0].shape[1] // self.imgScaleFactor, self.cam2s[0].shape[0] // self.imgScaleFactor)), cmap="gray")
        plt.axis('off')
        
        return cam2_points
        

    # Calculates the angle between two vectors
    def getAngleBetweenVectors(self, v1, v2):
        v1 = self.norm(v1)
        v2 = self.norm(v2)
        dot = np.einsum('ij, ij->i', v1, v2)
        return np.arccos(dot)
    
    # Normalize vector with small epsilon, such that no division by zero occurs.
    def norm(self, vec, axis=0):
        return vec / (np.linalg.norm(vec, axis=axis, keepdims=True) + 1e-8)
        
        
    # Runs PSO
    def run(self, max_iterations=50, log_iterations=True, render=True, log_results=True, max_converge_count=10):
        if (log_iterations):
            self.printSetup(max_iterations)
        iteration_count = 0
        
        # estimate normals
        if self.method != "reflectance" and self.method != "euclidean":
            self.normals = self.estimateNormals(self.point_clouds)

        # initializing
        initialGuess = np.zeros((1,6))
        R, T = self.getInitialGuess(euler=True)
        initialGuess[0,:3] = R
        initialGuess[0, 3:] = T
        
        rollPitchYaw_min = np.array([-self.randRoll, -self.randPitch, -self.randYaw])
        rollPitchYaw_max = np.array([self.randRoll, self.randPitch, self.randYaw])
        
        # Create random rotation distortion (shuffled)
        randRot = np.random.uniform(rollPitchYaw_min, rollPitchYaw_max, (self.n_particles, 3))
        np.random.shuffle(randRot)

        # Create random translation distortion (shuffled)
        randTransl = np.random.uniform(-self.randTranslation, self.randTranslation, (self.n_particles, 3))
        np.random.shuffle(randTransl)
        randRT = np.hstack([randRot, randTransl]) # N x 6
        
        X = initialGuess + randRT  # shape: N x 6        
        V = np.random.uniform(-.1, .1, (self.n_particles, 1))
        
        # Create random rotation distortion (shuffled)
        randRot = np.random.uniform(rollPitchYaw_min, rollPitchYaw_max, (self.n_particles, 3))
        np.random.shuffle(randRot)

        # Create random translation distortion (shuffled)
        randTransl = np.random.uniform(-self.randTranslation, self.randTranslation, (self.n_particles, 3))
        np.random.shuffle(randTransl)
        randRT = np.hstack([randRot, randTransl]) # N x 6
        
        pbest = X + randRT
        pbest_obj = np.zeros(self.n_particles)
        gbest = pbest[pbest_obj.argmax()]
        gbest_obj = 0.0
        gbest_obj_old = 0.0
        
        # converge counter
        converge_count = 0
        
        # Vectorized NMI function
        VecNMI = np.vectorize(self.nmi, otypes=[np.ndarray], signature='(n,n)->()')

        # Velocities (list of velocities for each iteration) -> for plotting
        velocities = []

        # NMIs (global best for each iteration)
        gbest_objs = []

        # RT-tuples (global best for each iteration)
        gbests = []

        # start timer
        pso_start = timer()
        
        while(iteration_count <= max_iterations):
            iteration_count += 1
            start = timer()

            r1, r2 = np.random.rand(2)
                
            # vectorizing
            X = X + V * self.norm(gbest - X, axis=0)
            X = X + V * self.norm(pbest - X, axis=0)
            X = X + V * np.random.uniform(-.1, .1, (self.n_particles, 6)) # N x 6

            V = self.w * V + self.c1*r1*np.linalg.norm(pbest - X, axis=0) + self.c2*r2*np.linalg.norm(pbest - X)
            
            E = self.makeRTVectorized(X[:, :3], X[:, 3:]) # euclidean matrices
            
            NMIs = VecNMI(E)

            pbest[(NMIs > pbest_obj), :] = X[(NMIs > pbest_obj), :]
            pbest_obj = np.array([NMIs, pbest_obj]).max(axis=0)
            gbest = pbest[pbest_obj.argmax(), :]
            gbest_obj_old = gbest_obj
            gbest_obj = pbest_obj.max()

            end = timer()

            velocities.append(V)
            gbest_objs.append(gbest_obj)
            gbests.append(gbest)
                   
            # Increment converge count if new optimum is old optimum
            # else: reset counter
            if gbest_obj == gbest_obj_old:
                converge_count += 1
            else:
                converge_count = 0

            if log_iterations:
                print("\rBest NMI for iteration  " + str(iteration_count) + "  with  " + str(self.n_particles) + "  particles:  " + "{:.8f}".format(gbest_obj) +  "  conv_count: " + str(converge_count) + "     eta: " + "{:.2f}".format(end-start) + "s", end="")
            
            # break loop if max converge count is reached
            # do not break, if max_converge_count == -1
            if max_converge_count != -1 and converge_count >= max_converge_count:
                print()
                break
        
        pso_end = timer()

        frame = np.zeros((self.cam2s[0].shape[0], self.cam2s[0].shape[1], 3), dtype="uint8")
        if render:
            frame = self.render(self.makeRT(*gbest))
        
        
        # Performance metrics
        frob = self.frobenius(self.makeRT(*gbest))
        deltas = self.calcDelta(self.makeRT(*gbest))
        quat = self.quat_diff(self.makeRT(*gbest))

        if log_results:
            print()
            print("DONE in ", (pso_end - pso_start), "s")
            print()
            print("NMI: ", gbest_obj)
            print("Delta x: ", deltas[0])
            print("Delta y: ", deltas[1])
            print("Delta z: ", deltas[2])
            print("Magnitude:", np.linalg.norm(np.array([deltas[0], deltas[1], deltas[2]])))
            print()
            print("Delta roll: ", deltas[3])
            print("Delta pitch: ", deltas[4])
            print("Delta yaw: ", deltas[5])
            print("Frobeniusnorm: " + str(frob))
            print("Quaternion Differenz: " + str(quat))
            print()
        
        if render:
            self.plotVelocities(velocities)
            self.plotVelocitiesOverIterations(velocities, iteration_count)
            self.plotNMIs(gbest_objs, iteration_count)

        return frame, gbest_obj, deltas, frob, quat


    # Plot 3D-Graph of Gradient
    def plotYawRollGradient(self, max_yaw=15, max_roll=3, num_values=5):
        print("Plotting Gradient ...")
        yaw = np.linspace(-max_yaw, max_yaw, num_values)
        roll = np.linspace(-max_roll, max_roll, num_values)
        yaw, roll = np.asarray(np.meshgrid(yaw, roll))
        
        RTs = np.array([self.makeRT(yaw[x][y], 0, roll[x][y], 0, 0, 0) @ self.RT_gt for x in range(num_values) for y in range(num_values)], dtype='float64')
        NMIs = np.array([self.nmi(RTs[x]) for x in range(num_values*num_values)])

        plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')
        dx = NMIs[1] - NMIs[0]
        dydx = np.gradient(NMIs, dx)
        ax.plot_trisurf(roll.flatten(), yaw.flatten(), dydx, cmap='jet', linewidth=0, antialiased=False)
        ax.view_init(20, 35)
        ax.set_xlabel('Yaw (degrees)')
        ax.set_ylabel('Roll (degrees)')
        ax.set_zlabel('Normalized Mutual Information')
        plt.show()



    # Plot 3D-Graph of NMI with changing roll and yaw angles (in deg)
    def plotYawRoll(self, max_yaw=15, max_roll=3, num_values=5):
        print("Plotting Yaw, Roll ...")
        yaw = np.linspace(-max_yaw, max_yaw, num_values)
        roll = np.linspace(-max_roll, max_roll, num_values)
        yaw, roll = np.asarray(np.meshgrid(yaw, roll))
        
        RTs = np.array([self.makeRT(yaw[x][y], 0, roll[x][y], 0, 0, 0) @ self.RT_gt for x in range(num_values) for y in range(num_values)], dtype='float64')
        NMIs = np.array([self.nmi(RTs[x]) for x in range(num_values*num_values)])

        #plt.style.use("ggplot")
        plt.rcParams.update(plt.rcParamsDefault)
        plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')
        surf = ax.plot_trisurf(roll.flatten(), yaw.flatten(), NMIs, cmap='jet', linewidth=0, antialiased=False)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
        ax.view_init(20, 35)
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
        ax.set_xlabel('Yaw (degrees)')
        ax.set_ylabel('Roll (degrees)')
        if self.modified:
            ax.set_zlabel('mNMI', rotation=90)
        else:
            ax.set_zlabel('NMI', rotation=90)
        plt.show()
        
    
    # 2D plot of NMI with changing roll and yaw angles (in deg)
    def plotYawRoll2D(self, max_yaw=15, max_roll=3, num_values=5):
        print("Plotting Yaw, Roll ...")
        yaw = np.linspace(-max_yaw, max_yaw, num_values)
        roll = np.linspace(-max_roll, max_roll, num_values)
        yaw, roll = np.asarray(np.meshgrid(yaw, roll))
        
        RTs = np.array([self.makeRT(yaw[x][y], 0, roll[x][y], 0, 0, 0) @ self.RT_gt for x in range(num_values) for y in range(num_values)], dtype='float64')
        NMIs = np.array([self.nmi(RTs[x]) for x in range(num_values*num_values)])

        plt.style.use("ggplot")
        ax = plt.gca()
        stepx = num_values // max_yaw
        xticks = np.linspace(0, num_values, stepx)
        ax.set_xticks(xticks)
        xlabels = ["%.2f" % number for number in np.linspace(-max_yaw, max_yaw, stepx)]
        ax.set_xticklabels(xlabels)
        
        stepy = num_values // max_roll
        yticks = np.linspace(0, num_values, stepy)
        ax.set_yticks(yticks)
        ylabels = ["%.2f" % number for number in np.linspace(-max_roll, max_roll, stepy)]
        ax.set_yticklabels(ylabels)

        ax.grid(linewidth=1)
        ax.set_xlabel('Yaw (degrees)')
        ax.set_ylabel('Roll (degrees)')
        im = ax.imshow(NMIs.reshape(num_values, num_values),cmap='jet', aspect='equal')
        cb=plt.colorbar(im)
        
        if self.modified:
            cb.set_label('mNMI')
        else:
            cb.set_label('NMI')
        plt.show()


    # Calculate and plot table with rotation and translation deltas
    def calcDelta(self, RT):
        translation_guess = self.RT_gt.copy()[:,3]
        translation_opt = RT[:,3]
        
        # Calculate deltas
        est_R = spatial.transform.Rotation.from_matrix(RT[:3, :3]).as_euler('zyx', degrees=True)
        true_R = spatial.transform.Rotation.from_matrix(self.RT_gt.copy()[:3, :3]).as_euler("zyx", degrees=True)

        rpy_err = est_R - true_R

        dx = np.abs(translation_guess[0] - translation_opt[0])
        dy = np.abs(translation_guess[1] - translation_opt[1])
        dz = np.abs(translation_guess[2] - translation_opt[2])
        
        labels = ("\u0394 x", "\u0394 y", "\u0394 z", "\u0394 roll", "\u0394 pitch", "\u0394 yaw")
        data = list(map("\u00B1 {:.3f}".format, [dx, dy, dz, rpy_err[2], rpy_err[1], rpy_err[0]]))
        
        return (dx, dy, dz, rpy_err[0], rpy_err[1], rpy_err[2])
        
        
    # Calculate frobenius norm of difference of real and estimated RT matrix
    def frobenius(self, RT):
        RT = RT[:3, :3] # neglect homogeneous property
        guess = self.RT_gt.copy()[:3, :3]
        diff = np.subtract(RT, guess) # calculate element wise difference between matrices
        return np.linalg.norm(diff, ord='fro') # calculte Frobenius norm
        
    
    # Calculate quaterion angle between rotation matrices
    def quat_diff(self, RT):
        def quaternion_multiply(quaternion1, quaternion0):
            w0, x0, y0, z0 = quaternion0
            w1, x1, y1, z1 = quaternion1
            return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                             x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                             -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                             x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    
        def inv(quat):
            return np.hstack([quat[0:1], -quat[1:4] ])
        
        initR = self.RT_gt.copy()[:3, :3]
        estR = RT.copy()[:3, :3]
    
        qdist = spatial.transform.Rotation.from_matrix(initR).as_quat()
        qestR = inv(spatial.transform.Rotation.from_matrix(estR).as_quat())
        
        d = quaternion_multiply(qdist, qestR)
        
        vec = np.linalg.norm(d[1:])
        mag = np.abs(d[0])
        
        rad = 2 * np.arctan2(vec, mag)
    
        deg = 180*np.abs(rad)/np.pi
        
        return deg

    
    # Calculates standard deviation of metrics between frames
    def std(self, NMIs, RT_deltas, frobenius, quaternion_differences):
        stdNMIs = np.std(NMIs)
        RT_deltas = np.asarray(RT_deltas)
        stddx = np.std(RT_deltas[:, 0])
        stddy = np.std(RT_deltas[:, 1])
        stddz = np.std(RT_deltas[:, 2])
        stdRoll = np.std(RT_deltas[:, 3])
        stdPitch = np.std(RT_deltas[:, 4])
        stdYaw = np.std(RT_deltas[:, 5])
        stdFrob = np.std(frobenius)
        stdQuat = np.std(quaternion_differences)
        
        print("--- Standardabweichungen ---")
        print("Std NMI:", stdNMIs)
        print("Std dx:", stddx)
        print("Std dy:", stddy)
        print("Std dz:", stddz)
        print("Std roll:", stdRoll)
        print("Std pitch:", stdPitch)
        print("Std yaw:", stdYaw)
        print("Std Frobenius:", stdFrob)
        print("Std Quaternion:",stdQuat)

    # Calculates mean of metrics between frames
    def mean(self, NMIs, RT_deltas, frobenius, quaternion_differences):
        stdNMIs = np.mean(NMIs)
        RT_deltas = np.asarray(RT_deltas)
        stddx = np.mean(RT_deltas[:, 0])
        stddy = np.mean(RT_deltas[:, 1])
        stddz = np.mean(RT_deltas[:, 2])
        stdRoll = np.mean(RT_deltas[:, 3])
        stdPitch = np.mean(RT_deltas[:, 4])
        stdYaw = np.mean(RT_deltas[:, 5])
        stdFrob = np.mean(frobenius)
        stdQuat = np.mean(quaternion_differences)
        
        print("--- Mittelwerte ---")
        print("Mean NMI:", stdNMIs)
        print("Mean dx:", stddx)
        print("Mean dy:", stddy)
        print("Mean dz:", stddz)
        print("Mean roll:", stdRoll)
        print("Mean pitch:", stdPitch)
        print("Mean yaw:", stdYaw)
        print("Mean Frobenius:", stdFrob)
        print("Mean Quaternion:",stdQuat)

    # Calculates mean of metrics between frames
    def median(self, NMIs, RT_deltas, frobenius, quaternion_differences):
        stdNMIs = np.median(NMIs)
        RT_deltas = np.asarray(RT_deltas)
        stddx = np.median(RT_deltas[:, 0])
        stddy = np.median(RT_deltas[:, 1])
        stddz = np.median(RT_deltas[:, 2])
        stdRoll = np.median(RT_deltas[:, 3])
        stdPitch = np.median(RT_deltas[:, 4])
        stdYaw = np.median(RT_deltas[:, 5])
        stdFrob = np.median(frobenius)
        stdQuat = np.median(quaternion_differences)
        
        print("--- Median ---")
        print("Median NMI:", stdNMIs)
        print("Median dx:", stddx)
        print("Median dy:", stddy)
        print("Median dz:", stddz)
        print("Median roll:", stdRoll)
        print("Median pitch:", stdPitch)
        print("Median yaw:", stdYaw)
        print("Median Frobenius:", stdFrob)
        print("Median Quaternion:",stdQuat)

    # Plot mean velocities for each particle
    def plotVelocities(self, velocities):
        velocities = np.asarray(velocities)
        mean = np.mean(velocities, axis=0)
        mean = [np.linalg.norm(x) for x in mean] 

        plt.figure()
        plt.bar(np.arange(self.n_particles), mean, label="x")
        plt.xlabel("particle Pi")
        plt.ylabel("velocity")
        plt.title("Mean velocity for each particle")
        plt.show()

    # Plot mean velocities per iteration
    def plotVelocitiesOverIterations(self, velocities, iterations):
        velocities = np.asarray(velocities)
        mean = np.mean(velocities, axis=1)
        print(mean.shape)
        mean = [np.linalg.norm(x) for x in mean] 

        plt.style.use('ggplot')
        plt.figure()
        plt.bar(np.arange(iterations), mean)
        plt.xlabel("iteration")
        plt.ylabel("velocity")
        
        import tikzplotlib
        tikzplotlib.save('velocities_kitti.tex')
        plt.show()
    
    def plotNMIs(self, NMIs, iterations):
        # get NMI for undistorted RT-matrix
        optimal_nmi = self.nmi(self.RT_gt)
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(iterations), NMIs, label="mNMI per iteration")
        plt.plot(np.arange(iterations), np.full(iterations, optimal_nmi), label="mNMI for ground truth")
        plt.legend()
        plt.xlabel("iteration")
        plt.ylabel("mNMI")
        import tikzplotlib
        tikzplotlib.save('NMI.tex')
        plt.show()


    def printSetup(self, max_iterations):
        print("===============================================")
        print("RUNNING PSO WITH THE FOLLOWING CONFIGURATION:")
        print("Particles amount: \t\t", self.n_particles)
        print("Method: \t\t\t", self.method)
        print("Gauss filter: \t\t\t", self.filter)
        print("Max. Iterations: \t\t", max_iterations)
        print("max_translation: \t\t", self.randTranslation, "[m]")
        print("max_roll: \t\t\t", self.randRoll, "[deg]")
        print("max_pitch: \t\t\t", self.randPitch, "[deg]")
        print("max_yaw: \t\t\t", self.randYaw, "[deg]")
        print("modified NMI calculation: \t", self.modified)
        print("===============================================")