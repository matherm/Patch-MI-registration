import os
import json

files_list = os.listdir()

files_list = filter(lambda x: ".py" not in x, files_list)
for file in files_list:
    with open(file, "r") as jsonFile:
        data = json.load(jsonFile)

    slice = file[len(file)-9:len(file)-5]
    i = slice[0:2]
    j = slice[-1]

    data["path_to_initial_guess"] = f"/home/ios/data3/kitti/tracking/training/gNMI/initial_guess_{i}_{j}.json"
    data["path_to_images"] = f"/home/ios/data3/kitti/tracking/training/gNMI/images_{i}.txt"
    data["path_to_point_clouds"] = f"/home/ios/data3/kitti/tracking/training/gNMI/point_clouds_{i}.txt"
    data["max_iter"] = 10
    data["delta_thresh"] = 0.00001
    data["gamma_trans_upper"] = 0.1
    data["gamma_trans_lower"] = 0.01
    data["gamma_rot_upper"] = 5.0
    data["gamma_rot_lower"] = 1.0

    data["x_min"] = 10.
    data["x_max"] = 10000.
    data["y_min"] = -10000.
    data["y_max"] = 10000.
    data["z_min"] = -50.
    data["z_max"] = 1000.


    with open(file, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)
