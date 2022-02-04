import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from IPython.display import Video
import numpy as np
import cv2

def create_movie(frames, fname="movie.mp4"):
    fig = plt.figure()
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    ani.save(fname)
   
def kitti_to_array(kitti):
    frames = [] # for storing the generated images
    for i in range(0, len(kitti)):
        point_cloud, cam2, image3, oxts = kitti[i]
        cam2_ = np.asarray(cam2).copy()
        frames.append([plt.imshow(cam2_, animated=True)])
        
    return frames


def images_to_video(images, video_name = 'video.avi'):

    frame = images[0]
    height, width, layers = frame.shape

    video =  cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width,height))
    for image in images:
        r, g, b = image[:,:, 0], image[:,:, 1], image[:,:, 2]
        image = np.stack([b, g, r]).transpose(1,2,0)
        video.write(image)

    cv2.destroyAllWindows()
    video.release()