import os
from PyKitti2Dataset import *
import glob 
import meshplot as mp
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, CenterCrop


class PyKittiPatches():
    
    def __init__(self, base_path, train=True, val_split=0.9, sequences=["0000"]):
        
        self.sequences = sequences
        self.base_path = base_path
        self.train = train
        self.val_split = val_split
        
        filenames_ = []
        for seq in sequences:
            filenames_ += sorted(glob.glob(os.path.join(self.base_path, 'patches', seq, '*', 'vol_*.npy')))
        
        self.filenames = []
        for i, f in enumerate(filenames_):
            if self.train and i % 100 < self.val_split * 100: # 0, 1, 2, 3, 89
                    self.filenames.append(f)
            if not self.train and i % 100 > self.val_split * 100: # 90, 91,..99
                    self.filenames.append(f)
                

    def get_keypoints(self, idx, local_coords=True):
        filename = self.filenames[idx].split("/")
        fname = filename[-1]
        frame_idx = int(fname.split("_")[1])
        pos = int(fname.split("_")[2])
        step  = filename[-2]
        seq   = filename[-3]
       
        keypoints2d = np.load(self.base_path + "/patches/" + seq + "/" + step + "/" + f"keypoints2d.npy")
        keypoints3d = np.load(self.base_path + "/patches/" + seq + "/" + step + "/" + f"keypoints3d.npy")
        RT = np.load(self.base_path + "/patches/" + seq + "/" + step + "/" + f"RT_{frame_idx}.npy")
        
        if local_coords:
            keypoints3d = (RT @ to_homo(keypoints3d).T).T[:, :3]
        
        return keypoints2d[pos], keypoints3d[pos]
        
    def __repr__(self):
        return f"PyKittiPatches(base_path={self.base_path}, train={self.train}, val_split={self.val_split}, sequences={self.sequences})[{len(self)}]"
    
    def __len__(self):
        return len(self.filenames)
                 
    @staticmethod
    def __preproc_pc__(pc):
        pc = torch.FloatTensor(pc).float()
        if len(pc) < 1024:
            padding = np.random.choice(len(pc), np.clip(1024 - len(pc), 0 , 1024))
            pc = torch.cat([pc, pc[padding]], dim=0)
        else:
            crop = np.random.choice(len(pc), 1024)
            pc = pc[crop]
        return pc
                 
    @staticmethod                 
    def __preproc_im__(im, transforms=Compose([ToPILImage(), Resize(128), CenterCrop(128), ToTensor()])):
        assert im.dtype == np.float32
        im = (im * 255).astype(np.uint8)
        im = transforms(im)
        im = im - im.mean()
        return im
           
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        filename_ = filename.split("/")
        fname = filename_[-1] # vol_1_13_.npy
        if len(fname.split("_")) == 4:
            frame_idx = int(fname.split("_")[1])
            pos = int(fname.split("_")[2])
        if len(fname.split("_")) == 3:    
            frame_idx = -1
            pos = int(fname.split("_")[1])
            
        step  = int(filename_[-2])
        seq   = int(filename_[-3])
        
        vol =  np.load(filename)
        patch =  plt.imread(filename.replace(".npy", ".png").replace("vol", "img"))[:, :, :3]
                
        vol = PyKittiPatches.__preproc_pc__(vol)
        patch = PyKittiPatches.__preproc_im__(patch)
        
        return patch, vol.T, seq, step, frame_idx, pos
    
    def show_pair(self, idx):
        im, pc, seq, step, frame_idx, pos = self[idx]
        im = im.numpy().transpose(1,2,0)
        # meshplot is looking down negative z
        x, y, z = pc 
        pc = np.stack([x, z, -y]).T
        mp.plot(pc, c=pc[:, 1], shading={"point_size": 0.3, "width": 300, "height": 300})
        print(im.shape, pc.shape)
        plt.imshow(im)
        plt.axis("off")
        plt.show()
        
         
    def build_dataset(self, stepsize = 3, max_distance=40, min_occurences=1, min_points=30, start=("0000", 0)):
        
        basedir = self.base_path
        os.mkdir(basedir + "/patches") if not os.path.exists(basedir + "/patches") else None
        has_started = False
        
        for seq in PyKitti2.SEQUENCES:
               
            os.mkdir(basedir + "/patches/" + seq) if not os.path.exists(basedir + "/patches/" + seq) else None
            kitti = PyKitti2(basedir, seq)    

            for f in range(0, len(kitti) - stepsize, stepsize): 
                
                if not has_started:
                    if seq == start[0] and f == start[1]:
                        has_started = True
                    else:
                        continue
                
                #try:
                
                os.mkdir(basedir + "/patches/" + seq + "/" + str(f)) if not os.path.exists(basedir + "/patches/" + seq + "/" + str(f)) else None

                volumes, pclocs, patches, ilocs, RTs, frame_idx, aff = kitti.compute_matches_cam2(f, f+stepsize, 
                                                                                       max_distance=30, 
                                                                                       min_occurences=1, 
                                                                                       min_points=30)
                for i,v in enumerate(volumes):
                    np.save(basedir + "/patches/" + seq + "/" + str(f) + "/" + f"vol_{frame_idx[i]}_{i}_.npy", v)

                for i,p in enumerate(patches):
                    plt.imsave(basedir + "/patches/" + seq + "/" + str(f) + "/" + f"img_{frame_idx[i]}_{i}_.png", p)   

                np.save(basedir + "/patches/" + seq + "/" + str(f) + "/" + f"keypoints3d.npy", pclocs)
                np.save(basedir + "/patches/" + seq + "/" + str(f) + "/" + f"keypoints2d.npy", ilocs)
                np.save(basedir + "/patches/" + seq + "/" + str(f) + "/" + f"frame_idx.npy", frame_idx)
                np.save(basedir + "/patches/" + seq + "/" + str(f) + "/" + f"match_idx.npy", aff)

                for i,rt in enumerate(RTs):
                    np.save(basedir + "/patches/" + seq + "/" + str(f) + "/" + f"RT_{i}.npy", rt)

                print(seq, f, len(volumes))
                        
                #except Exception as e: 
                #    print("Could not process", seq, f)
                #    print(e)
                    