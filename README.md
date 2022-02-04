# Targetless Lidar-camera registration usingpatch-wise mutual information optimization

![](/teaser.PNG)

## Usage
    basedir = "/path/to/kitti/like/dataset"
    datas_test = [PyKitti2(basedir, f"{i:04d}", with_labels="none") for i in range(10,14)]
    
    kitti = datas_test_[0]
    
    ac = AutoCalibration(kitti.calib.T_cam2_velo, kitti.calib.P_rect_20).cuda()
    ac.MINE.load_state_dict(torch.load("checkpoints/mi_model_p96_small_registered_3999.pth.tar))
    ac.reset_calibration()
    
    interval_r, interval_t = randomRT(0.3, 2)
    ac.distort_calibration(interval_t, interval_r)
            
    for iters, lr, grad, sample in [(3000, 1e-3, False, False), (1000, 1e-4, True, False), (1000, 1e-5, True, False)]:

        ac.update_se_and_mine(lr_mi=lr, lr_R=lr, lr_T=10*lr)
        r = ac.fit(datas_test_, iters=iters, log_interval=30, eps=1e-8, grad_mine=grad)