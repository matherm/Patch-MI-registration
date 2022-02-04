import torch
import numpy as np

def gaussian_mask(u, s, d, R, C):
    """
    :param u: centre of the first Gaussian.
    :param s: standard deviation of Gaussians.
    :param d: shift between Gaussian centres.
    :param R: int, number of rows in the mask, there is one Gaussian per row.
    :param C: int, number of columns in the mask.
    """
    # indices to create centres
    device = d.device
    R = torch.arange(R).view(1,1,1,-1).to(device) # batch, channels, cross, rows, 
    C = torch.arange(C).view(1,1,-1,1).to(device)  # batch, channels, cols, cross
    d = d.reshape(-1, 1, 1, 1) # expand for explicit broadcasting
    u = u.reshape(-1, 1, 1, 1)
    s = s.reshape(-1, 1, 1, 1)
    Rd = R * d
    centres = u + Rd
    column_centres = C - centres
    mask = torch.exp(-.5 * torch.pow(column_centres / s, 2))
    # we add eps for numerical stability
    normalised_mask = mask / (torch.sum(mask, dim=2, keepdims=True) + 1e-8)
    return normalised_mask # B, 1, R, C

def gaussian_glimpse(img_tensor, transform_params, crop_size):
    """
    :param img_tensor: (batch_size, channels, Height, Width)
    :param transform_params: Tensor of size (batch_size, 6), where params are  (mean_y, std_y, d_y, mean_x, std_x, d_x) specified in pixels.
    :param crop_size): tuple of 2 ints, size of the resulting crop
    """
    if not torch.is_tensor(img_tensor):
        return gaussian_glimpse(torch.FloatTensor(img_tensor), torch.FloatTensor(transform_params), crop_size).numpy()
    
    # parse arguments
    h, w = crop_size
    Bi, C, H, W = img_tensor.shape
    Bp = transform_params.shape[0]
    B = np.max([Bi, Bp])
    uy, sy, dy, ux, sx, dx = torch.split(transform_params, split_size_or_sections=6, dim=1)[0].T
    # create Gaussian masks, one for each axis
    Ay = gaussian_mask(uy, sy, dy, h, H).to(img_tensor.device)
    Ax = gaussian_mask(ux, sx, dx, w, W).to(img_tensor.device)
    # extract glimpse
    img_tensor = img_tensor.view(Bi*C,H,W)
    Ax = Ax.repeat(repeats=(1,C,1,1))
    Ay = Ay.repeat(repeats=(1,C,1,1))
    Ay = Ay.view(B*C,H,h)
    Ax = Ax.view(B*C,W,w)
    if Bi < Bp:
        raise NotImplementedError
        #print(img_tensor.shape, Ax.shape)
        # torch.Size([300, 375, 1242]) torch.Size([300, 1242, 10])
        # torch.Size([300, 375, 1242]) torch.Size([300, 1242, 10]) => 300
        #glimpse = torch.bmm(Ay.transpose(2,1), torch.mm(img_tensor, Ax))
    else:
        #print(img_tensor.shape, Ax.shape, torch.bmm(img_tensor, Ax).shape)
        #torch.Size([300, 375, 1242]) torch.Size([300, 1242, 10]) torch.Size([300, 375, 10])
        glimpse = torch.bmm(Ay.transpose(2,1), torch.bmm(img_tensor, Ax))
    glimpse = glimpse.view(B,C,h,w)
    return glimpse


def crop(img, xy, crop_size=(3,3), center=True, std_y = 1, std_x = 1 ):
    
    center_x, center_y = xy.T
    if center:
        center_x, center_y = center_x - crop_size[1]/2, center_y - crop_size[0]/2
    
    # STANDARD KERNEL
    std_y = (torch.ones_like(center_x) * std_y)
    std_x = (torch.ones_like(center_x)  * std_x)
    dx = (torch.ones_like(center_x)  * 1)
    dy = (torch.ones_like(center_x)  * 1)
        
    params = torch.stack([center_y, std_y, dy, center_x, std_x, dx ]).T      
    glimpse = gaussian_glimpse(img, params, crop_size)
    return glimpse