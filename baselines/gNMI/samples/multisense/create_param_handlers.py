import os
import json

files_list = os.listdir()
files_list = filter(lambda x: x.startswith("calibration_handler_param_multisense"), files_list)

for file in files_list:
    print(file)
    with open(file, "r") as jsonFile:
        data = json.load(jsonFile)
    
    # Change json params
    data["delta_trans"] = 0.01
    data["delta_rot_deg"] = 1.0
    data["gamma_trans_lower"] = 0.01
    data["gamma_rot_lower"] = 1.0
    data["gamma_trans_upper"] = 0.1
    data["gamma_rot_upper"] = 5.0

    data["delta_trans"] = 0.01
    data["delta_rot_deg"] = 1.0

    data["max_iter"]= 100

    data["x_min"] = -10000.
    data["x_max"] = 10000.
    data["y_min"] = -10000.
    data["y_max"] = 10000.
    data["z_min"] = -2.
    data["z_max"] = 50.


    slice = file[len(file)-8:len(file)-5]
    i = slice[0]
    j = slice[-1]

    if "stereo" not in file:
        desiredPathInitialGuess = f"/home/ios/data3/multisense_gNMI_data/initial_guess_{i}_{j}.json"
        desiredImagePath = "/home/ios/data3/multisense_gNMI_data/images.txt"
        desiredPointCloudPath = "/home/ios/data3/multisense_gNMI_data/point_clouds.txt"
        desiredCameraInfoPath = "/home/ios/data3/multisense_gNMI_data/camera_info.json"
    else:
        desiredPathInitialGuess = f"/home/ios/data3/multisense_gNMI_stereo_data/initial_guess_{i}_{j}.json"
        desiredImagePath = "/home/ios/data3/multisense_gNMI_stereo_data/images.txt"
        desiredPointCloudPath = "/home/ios/data3/multisense_gNMI_stereo_data/point_clouds.txt"
        desiredCameraInfoPath = "/home/ios/data3/multisense_gNMI_stereo_data/camera_info.json"

    data["path_to_initial_guess"] = desiredPathInitialGuess
    data["path_to_images"] = desiredImagePath
    data["path_to_point_clouds"] = desiredPointCloudPath
    data["path_to_camera_info"] = desiredCameraInfoPath

    with open(file, "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)
