from torchvision.transforms import Resize, ToTensor, Compose, ToPILImage
import glob, os, shutil
from imageio import imread
import open3d as o3d
import numpy as np
import meshplot as mp
import matplotlib.pyplot as plt
import zipfile
import torch
from iosdata import NasFile

class OxfordMatchDataset(torch.utils.data.Dataset):
    
    SIFT_PATH = "/3d-data/oxford/sift_zip/"
    ISS_PATH = "/3d-data/oxford/iss_zip/iss_volume/"
    
    
    def __init__(self, path, segments=["2014-12-12-10-45-15"], train=True, train_split=0.9):
        super().__init__()
        
        self.train = train
        self.train_split = train_split
        self.path = path
        self.segments = segments
        self.patch_file_list = []
        for segment in segments:
            l = sorted(glob.glob(path + "/" + OxfordMatchDataset.SIFT_PATH + "/" + segment + "/**/*.png"))
            if self.train:
                l = l[:int(len(l)*self.train_split)]
            else:
                l = l[int(len(l)*self.train_split):]
            self.patch_file_list = self.patch_file_list + l
            
    def __download__(self):
        # Download the Dataset from IOSNAS DS02
        iss = NasFile(path=self.path, host="iosds02", server_path="/3d-data/oxford/iss_volume.zip", download=True)
        sift1 = NasFile(path=self.path, host="iosds02", server_path="/3d-data/oxford/sift_patch_1.zip", download=True)
        sift2 = NasFile(path=self.path, host="iosds02", server_path="/3d-data/oxford/sift_patch_2.zip", download=True)

        iss_zip = zipfile.ZipFile(iss.file_path)
        iss_zip.extractall(path=self.path + "/3d-data/oxford/iss_zip/")
        #os.remove(iss.file_path)
        
        target_dir = self.path + "/3d-data/oxford/sift_zip/"
        
        sift1_zip = zipfile.ZipFile(sift1.file_path)
        sift1_zip.extractall(path=target_dir)
        #os.remove(sift1.file_path)
        
        sift2_zip = zipfile.ZipFile(sift2.file_path)
        sift2_zip.extractall(path=target_dir)
        #os.remove(sift2.file_path)
                        
        #source_dir = self.path + "/3d-data/oxford/sift_zip/sift_patch_1/"
        #file_names = os.listdir(source_dir)
        #for file_name in file_names:
        #    shutil.move(os.path.join(source_dir, file_name), target_dir)
        #os.removedirs(source_dir)
        
        #source_dir = self.path + "/3d-data/oxford/sift_zip/sift_patch_2/"
        #file_names = os.listdir(source_dir)
        #for file_name in file_names:
        #    shutil.move(os.path.join(source_dir, file_name), target_dir)
        #os.removedirs(source_dir)
            

    def __preproc_im__(self, im, transforms=Compose([ToPILImage(), Resize(128), ToTensor()])):
        im = transforms(im)
        im = im - im.mean()
        return im
    
    def __preproc_pc__(self, pc):
        pc = torch.FloatTensor(pc)
        if len(pc) < 1024:
            padding = np.random.choice(len(pc), np.clip(1024 - len(pc), 0 , 1024))
            pc = torch.cat([pc, pc[padding]], dim=0)
        else:
            crop = np.random.choice(len(pc), 1024)
            pc = pc[crop]
        return pc
    
    def __len__(self):
        return len(self.patch_file_list)
            
    def __repr__(self):
        return f"OxfordMatchDataset(path={self.path}, segments={self.segments}, train={self.train}, train_split={self.train_split})[{len(self.patch_file_list)}]"

    def __getitem__(self, idx):
        patch_path = self.patch_file_list[idx]
        volume_path = patch_path.replace(OxfordMatchDataset.SIFT_PATH, OxfordMatchDataset.ISS_PATH).replace(".png", ".pcd")
        
        im = imread(patch_path)
        pc = np.asarray(o3d.io.read_point_cloud(volume_path).points)
        
        im = self.__preproc_im__(im)
        pc = self.__preproc_pc__(pc)
        
        return [im, pc.T]
    
    
    def show_pair(self, idx):
        im, pc = self[idx]
        im, pc = im.numpy(), pc.numpy().T
        im = im - im.min()
        mp.plot(pc, c=pc[:, 1], shading={"point_size": 0.3, "width": 300, "height": 300})
        plt.imshow(im.transpose(1,2,0))
        plt.show()    
