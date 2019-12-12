from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
from train_utils import *
from drive_net import DriveNet
import json



with open('config_data_home.txt') as json_file:
    parameters = json.load(json_file)



def load_billboard(path):

    print("--------LOAD BILLBOARD--------")
    print(path)
    billboard = cv2.imread(path)
    billboard_rgb = billboard[:, :, ::-1]
    if parameters['show_imgs_for_debug']:
        plt.imshow(billboard_rgb)
        plt.show()
    return billboard, billboard_rgb


os.environ["CUDA_VISIBLE_DEVICES"] = str(parameters['gpu'])
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
billboard_choosen = False
billboard_idx = 0
# Setting for data
drive_net_dataset = 1
folder_offset = 0
last_folder_value = 15+1
data = {}
print("--------INITIALIZE MODEL-------")
model = DriveNet()

for folder_idx in range(len(parameters['raw_imgs_path'][folder_offset:last_folder_value])):
    billboard_choosen = False

    data["billboard"+str(folder_idx+1)] = []
    with open('evaluation.txt', 'w') as outfile:
        json.dump(data, outfile)

    for raw_imgs_path_idx, raw_imgs_path in enumerate(parameters['raw_imgs_path'][folder_offset:last_folder_value]):
        raw_imgs_path = parameters['vanishing_folder'] + raw_imgs_path
        print(raw_imgs_path+parameters['log_info_path'])
        if not billboard_choosen:
            billboard, billboard_rgb = load_billboard( parameters['vanishing_folder'] +
                                                       parameters['raw_imgs_path'][billboard_idx+folder_offset] +
                                                       parameters['predicted_angles_file_path'] +
                                                       'iters8channel2new_logo.png')
            billboard_choosen = True

        print("--------COORDINATES READ-------")
        with open(raw_imgs_path+parameters['coordinates_path']) as f:
            content_coordinates = f.readlines()
        coordinates = parse_logfile(content_coordinates)
        # print(coordinates)

        print("--------LOG FILE READ-------")
        with open(raw_imgs_path+parameters['log_info_path']) as f:
            content_log_file = f.readlines()
        log_file = parse_logfile(content_log_file)
        # print(log_file)

        print("--------IMG NAMES READ-------")
        img_array = get_sorted_list_of_raw_imgs(raw_imgs_path)
        # print(img_array)

        print("--------IMG SORT THREE CHANNELS-------")
        drivenet_inputs_mapping = prepare_imgs_for_drivenet(log_file, use_same_img=False)

        print("--------RESIZE COORDINATES-------")
        coordinates_resized = []
        coordinates_resized = resize_coordinates_after_crop(coordinates, False, drive_net_dataset)

        for idx_row, crd_row in enumerate(coordinates):
            for idx_clmn, crd_clmn in enumerate(crd_row):
                coordinates[idx_row][idx_clmn] = int(float(crd_clmn))

        print("--------GET BILLBOARD SIZE--------")
        billboard_start_coords, occl_sizes = get_occl_size_mod(coordinates, False)
        billboard_start_coords_resized, occl_sizes_resized = get_occl_size_mod(coordinates_resized, False)

        print("--------IMG PREPARE-------")
        imgs, orig_shape = crop_img_for_drivenet(img_array, drive_net_dataset)
        imgs_rgb = imgs_to_rgb_array(img_array)

        print("--------MAP BILLBOARD TO IMGS-------")
        imgs_update_rgb = update_image_tmp(np.array(imgs_rgb), billboard_rgb, billboard_start_coords, occl_sizes)
        imgs_update = update_image_tmp(imgs, billboard/255-0.5, billboard_start_coords_resized, occl_sizes_resized)
        #if parameters['show_imgs_for_debug']:
        for i in range(len(imgs)):
            plt.imshow(imgs_update_rgb[drivenet_inputs_mapping[i][0]][0])
            plt.show()

        print("--------PREDICT ANGLE AND GRAD-------")

        angle_without_attack = evaluate_angle(model, imgs, drivenet_inputs_mapping, log_file)
        angle_with_attack = evaluate_angle(model, imgs_update, drivenet_inputs_mapping, log_file)
        abs_angle_diff = 0
        count_valid_attack = 0
        for img_idx in range(0, len(img_array)):
            abs_angle_diff += abs(angle_without_attack[img_idx]-angle_with_attack[img_idx])
            if abs(angle_without_attack[img_idx]-angle_with_attack[img_idx])[0] > parameters['tau']:
                count_valid_attack += 1
        print("--------PREDICT ANGLE AND GRAD-------")
        print('MAE: ', abs_angle_diff/len(img_array))
        print('RAE: ', count_valid_attack/float(len(img_array)))
        mae = abs_angle_diff/len(img_array)
        rae = count_valid_attack/float(len(img_array))
        create_json(data, "billboard"+str(folder_idx+1), "szene"+str(raw_imgs_path_idx+1), mae[0], rae)
    billboard_idx += 1
print("billboard"+str(folder_idx+1))
with open('evaluation.txt', 'w') as outfile:
    json.dump(data, outfile)
