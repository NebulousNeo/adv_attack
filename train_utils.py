import os
from scipy.misc import imsave
import json
import numpy as np
import copy
import cv2
import math

with open('config_data_home_two_paths.txt') as json_file:
    parameters = json.load(json_file)


def transform_occl3(gradients, start_point, rect_shape, logo_data, order):
    start_point = [int(i) for i in start_point]
    rect_shape = [int(i) for i in rect_shape]
    # new_grads = np.zeros((np.shape(gradients)[0], rect_shape[0], rect_shape[1], np.shape(gradients)[3]))
    new_grads = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] +
                                                                                           rect_shape[1], :]
    logo_shape = np.shape(logo_data)
    # In this version (29/07/2018), we do not use own rescale code but opencv resize instead
    # print(np.shape(new_grads))
    logo_data[order] = cv2.resize(new_grads[0], dsize=(logo_shape[2], logo_shape[1]))

    return logo_data


def control_bound(logo):
    # nachdem logo erstellt wurde mit einer methode wird diese auf den druckbaren bereich beschraenkt fuer alle channels
    np.clip(logo[:, :, 0], -0.5, 0.5, out=logo[:, :, 0])
    np.clip(logo[:, :, 1], -0.5, 0.5, out=logo[:, :, 1])
    np.clip(logo[:, :, 2], -0.5, 0.5, out=logo[:, :, 2])
    return logo


def deprocess_image(x, shape=(52, 265, 3), ):
    tmp = copy.deepcopy(x)
    tmp = tmp.reshape(shape)
    # Remove zero-center by mean pixel
    tmp[:, :, 0] += 0.5
    tmp[:, :, 1] += 0.5
    tmp[:, :, 2] += 0.5
    # 'BGR'->'RGB'
    tmp = tmp[:, :, ::-1]
    tmp = np.clip(tmp * 255, 0, 255).astype('uint8')
    return tmp


def update_image_imgs(imgs, logo, start_point, occl_size):
    update_image = copy.deepcopy(imgs)
    # for i in start_point:
    #    start_point[i] = [int(j) for j in i]
    for i in range(len(update_image)):
        update_image[i][0, start_point[i][0]:start_point[i][0] + occl_size[i][0],
        start_point[i][1]:start_point[i][1] + occl_size[i][1], :] = cv2.resize(logo,
                                                                               (occl_size[i][1], occl_size[i][0]))[
                                                                    :min(occl_size[i][0],
                                                                         np.shape(update_image[i])[1] - start_point[i][0]),
                                                                    :min(occl_size[i][1],
                                                                         np.shape(update_image[i])[2] - start_point[i][1])]
    return update_image


def update_image_tmp(imgs_tmp, logo, start_point, occl_size):
    update_tmp = copy.deepcopy(imgs_tmp)
    # for i in start_point:
    #    start_point[i] = [int(j) for j in i]
    for i in range(len(update_tmp)):
        update_tmp[i][0, start_point[i][0]:start_point[i][0] + occl_size[i][0],
        start_point[i][1]:start_point[i][1] + occl_size[i][1], :] = cv2.resize(logo,
                                                                               (occl_size[i][1], occl_size[i][0]))[
                                                                    :min(occl_size[i][0],
                                                                         np.shape(update_tmp[i])[1] - start_point[i][0]),
                                                                    :min(occl_size[i][1],
                                                                         np.shape(update_tmp[i])[2] - start_point[i][1])]
    return update_tmp


def total_diff(imgs_list, model, angles_2, drivenet_inputs_mapping, log_file):
    angles_diff = []
    for count_diff, img in enumerate(imgs_list):
        angles_diff.append(abs(
            model.predict(count_diff, drivenet_inputs_mapping, imgs_list, log_file, True, False)[0].angle - angles_2[
                count_diff]))
    return sum(angles_diff)


def find_kth_max(array, k):
    tmp = array.flatten()
    tmp = abs(tmp)
    tmp.sort()
    return tmp[-k]


def constraint_occl(gradients, start_point, rect_shape):
    # Hier wird der Gradient fuer den Plakatbereich geupdatet
    # start_point = [int(i) for i in start_point]
    # rect_shape = [int(i) for i in rect_shape]
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def normalize(x):
    # wird normalisiert ueber RGB channel
    # utility function to normalize a tensor by its L2 norm
    # return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
    return x / (np.sqrt(np.mean(np.square(x))) + 1e-5)


def evaluate_on_generated_imgs(raw_imgs_path, model, imgs, drivenet_inputs_mapping, angle_without_attack, log_file):
    print("--------GENERATE THE OUTPUT-------")
    predicted_angles_txtfile = open(raw_imgs_path + parameters['predicted_angles_file_path'] +
                                    'truth_false_steering_angle.txt', "w")
    # index to import for train folder
    for img_idx in range(len(imgs)):
        predicted_value_new = model.predict(img_idx, drivenet_inputs_mapping, imgs, log_file, True, False)
        angle_for_generated_img = predicted_value_new[0].angle
        current_angle = angle_without_attack[img_idx]
        predicted_angles_txtfile.write(str(img_idx) + " " + str(float(current_angle)) +
                                       " " + str(float(angle_for_generated_img)) + "\n")
        gen_img_deprocessed = draw_arrow3(deprocess_image(imgs[img_idx]),
                                          min(max(current_angle, -math.pi / 2), math.pi / 2),
                                          angle_for_generated_img)
        imsave(raw_imgs_path + parameters['predicted_angles_file_path'] +
               str(img_idx) + 'th_img.png', gen_img_deprocessed)
    predicted_angles_txtfile.close()


def draw_arrow3(img, angle1, angle2):
    pt1 = (int(img.shape[1] / 2), img.shape[0])
    pt2_angle1 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle1 * math.pi * -2 / 360)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle1 * math.pi * -2 / 360)))
    pt2_angle2 = (int(img.shape[1] / 2 - img.shape[0] / 3 * math.sin(angle2 * math.pi * -2 / 360)),
                  int(img.shape[0] - img.shape[0] / 3 * math.cos(angle2 * math.pi * -2 / 360)))
    img = cv2.arrowedLine(img, pt1, pt2_angle1, (0, 0, 255), 5)
    img = cv2.arrowedLine(img, pt1, pt2_angle2, (0, 255, 0), 5)
    return img


def save_information(raw_imgs_path, img_size, x_size_logo, y_size_logo, x_min, y_min, x_max, y_max, mean_angle_error):
    with open(raw_imgs_path + parameters['predicted_angles_file_path'] +
              'billboard_and_mean_angle_data.txt', "w") as billboard_and_mean_angle_data:
        billboard_and_mean_angle_data.write('img_size, x_size_logo, y_size_logo, '
                                            'x_min, y_min, x_max, y_max, mean_angle_error' + '\n')
        billboard_and_mean_angle_data.write(str(img_size) + ',' + str(x_size_logo) + ',' +
                                            str(y_size_logo) + ',' + str(x_min) + ',' + str(y_min) + ',' +
                                            str(x_max) + ',' + str(y_max) + ',' + str(mean_angle_error))
        print(img_size, x_size_logo, y_size_logo, x_min, y_min, x_max, y_max, mean_angle_error)


def evaluate_angle(model, imgs, drivenet_inputs_mapping, log_file):
    angle_without_attack = list()
    for img_idx in range(len(log_file)):
        predicted_value = model.predict(img_idx, drivenet_inputs_mapping, imgs, log_file, True, False)
        predicted_truth_angle = predicted_value[0].angle
        angle_without_attack.append(predicted_truth_angle)
    return angle_without_attack


def crop_img_for_drivenet(img_array, drive_net_dataset=0):
    orig_shape = img_array[1][0].shape
    image_array_croped = copy.deepcopy(img_array)
    for img_idx in range(0, len(image_array_croped)):
        image_array_croped[img_idx][0] = image_array_croped[img_idx][0][
                                         parameters['crop_offset_height'][drive_net_dataset]:
                                         parameters['crop_offset_height'][drive_net_dataset] +
                                         parameters['crop_height'][drive_net_dataset],
                                         parameters['crop_offset_width'][drive_net_dataset]:
                                         parameters['crop_offset_width'][drive_net_dataset] +
                                         parameters['crop_width'][drive_net_dataset]]
        image_array_croped[img_idx][0] = cv2.resize(image_array_croped[img_idx][0],
                                                    (parameters['resize_width'],
                                                     parameters['resize_height'])) / 255.0 - 0.5
        # (rows, cols, channels) = image_array_croped[img_idx][0].shape
        image_array_croped[img_idx] = np.asarray(image_array_croped[img_idx])

    return image_array_croped, orig_shape

def imgs_to_rgb_array(img_array):
    imgs_np_array = copy.deepcopy(img_array)
    for img_idx in range(0, len(imgs_np_array)):
        imgs_np_array[img_idx][0] = imgs_np_array[img_idx][0][:,:,::-1]
    return  imgs_np_array

def parse_logfile(data):
    file_parsed = [
        line.replace(' ', ',').replace('\n', '').replace('\'', '')
            .replace('\t', ',').replace('[', '').replace(']', '').replace(
            '\r', ',').split(',') for
        line
        in data]
    for row_file_parsed in range(len(file_parsed)):
        file_parsed[row_file_parsed] = [x for x in file_parsed[row_file_parsed] if x != '']
    return file_parsed


def prepare_imgs_for_drivenet(log_file_values, use_same_img=False):
    data_order = list()
    distance_four = 0
    if use_same_img:
        # use same image is used three times for a single input to DriveNet
        for log_file_idx in range(len(log_file_values)):
            current_img = int(log_file_values[log_file_idx][0])
            data_order.append([current_img])
            data_order[log_file_idx].append(current_img)
            data_order[log_file_idx].append(current_img)
    else:
        # three different images with distance=4m are used as a single input for DriveNet
        for log_file_idx in range(len(log_file_values)):
            data_order.append([int(log_file_values[log_file_idx][0])])
            idx_second_input = 0
            if log_file_idx == 0:
                data_order[log_file_idx].append(int(log_file_values[log_file_idx][0]))
                data_order[log_file_idx].append(int(log_file_values[log_file_idx][0]))
                continue
            for idx_second_input in range(log_file_idx, -1, -1):  # type: int
                if idx_second_input == log_file_idx:
                    distance_between_next = 0
                    distance_four = 0
                else:
                    distance_between_next = ((int(log_file_values[idx_second_input + 1][1]) + 10 ** (-9) *
                                              int(log_file_values[idx_second_input + 1][2]))
                                             - (int(log_file_values[idx_second_input][1]) + 10 ** (-9) *
                                                int(log_file_values[idx_second_input][2]))) * 0.2 * float(
                        log_file_values[idx_second_input][3])
                distance_four = distance_four + distance_between_next
                if distance_four >= 4:
                    data_order[log_file_idx].append(int(log_file_values[idx_second_input][0]))
                    break
                if idx_second_input == 0:
                    data_order[log_file_idx].append(int(log_file_values[idx_second_input][0]))

            for idx_third_input in range(idx_second_input, -1, -1):
                current_img_name = log_file_values[idx_third_input]
                if idx_third_input == idx_second_input:
                    distance_between_next = 0
                    distance_eight = distance_four  # type: int
                else:
                    next_img_name = log_file_values[idx_third_input + 1]
                    distance_between_next = ((int(next_img_name[1]) + 10 ** (-9) * int(next_img_name[2])) -
                                             (int(current_img_name[1]) + 10 ** (-9) *
                                              int(current_img_name[2]))) * 0.2 * float(
                        log_file_values[idx_second_input][3])
                distance_eight = distance_eight + distance_between_next
                if distance_eight >= 8:
                    data_order[log_file_idx].append(int(current_img_name[0]))
                    distance_eight = distance_four
                    break
                if idx_third_input == 0:
                    data_order[log_file_idx].append(int(current_img_name[0]))

    return data_order


def get_sorted_list_of_raw_imgs(raw_imgs_path):
    img_names = sorted(
        [int(filename.replace('.png', '')) for filename in os.listdir(raw_imgs_path) if
         (filename != 'coordinates.txt' and filename != 'log_info.txt' and filename != 'train_output')])
    img_names = [raw_imgs_path + str(img_name) + ".png" for img_name in img_names]

    image_array = []
    for img in img_names:
        orig_image = cv2.imread(img)
        image_array.append([orig_image])
    return image_array


def resize_coordinates_after_crop(contents_parsed, show_out_of_range, drive_net_dataset=0):
    contents_parsed = copy.deepcopy(contents_parsed)
    num_coords = 9
    x_y_coords = [2, 4, 6, 8]
    for record_idx in range(len(contents_parsed)):
        for coord_idx in range(num_coords):
            coord = int(float(contents_parsed[record_idx][coord_idx]))
            if coord_idx in x_y_coords:
                if parameters['crop_offset_height'][drive_net_dataset] > coord:
                    # top out of range
                    new_value = 0
                    if show_out_of_range:
                        print(record_idx, 'top out of range')
                elif parameters['crop_offset_height'][drive_net_dataset] + \
                        parameters['crop_height'][drive_net_dataset] < coord:
                    # bottom out of range
                    new_value = int(parameters['crop_height'][drive_net_dataset] *
                                    parameters['resize_height'] / parameters['crop_height'][drive_net_dataset])
                    if show_out_of_range:
                        print(record_idx, 'bottom out of range')
                else:
                    new_value = int((coord - parameters['crop_offset_height'][drive_net_dataset]) *
                                    parameters['resize_height'] / parameters['crop_height'][drive_net_dataset])
            elif coord_idx == 0:
                new_value = coord
            else:
                if parameters['crop_offset_width'][drive_net_dataset] > coord:
                    # left side out of range
                    new_value = 0
                    if show_out_of_range:
                        print(record_idx, 'left side out of range')
                elif parameters['crop_offset_width'][drive_net_dataset] + \
                        parameters['crop_width'][drive_net_dataset] < coord:
                    # right side out of range
                    new_value = int(parameters['crop_width'][drive_net_dataset] *
                                    parameters['resize_width'] / parameters['crop_width'][drive_net_dataset])
                    if show_out_of_range:
                        print(record_idx, 'right side out of range')
                else:
                    new_value = int((coord - parameters['crop_offset_width'][drive_net_dataset]) *
                                    parameters['resize_width'] / parameters['crop_width'][drive_net_dataset])
            contents_parsed[record_idx][coord_idx] = new_value
    return contents_parsed


def get_occl_size_mod(coordinates, dimension_billboard_raw):
    occl_sizes = list()
    start_points = list()
    if not dimension_billboard_raw:
        for record_idx in range(len(coordinates)):
            start_points.append(
                [coordinates[record_idx][2], coordinates[record_idx][1]])
            occl_sizes.append([max(1, coordinates[record_idx][8] - coordinates[record_idx][2]),
                               max(1, coordinates[record_idx][7] - coordinates[record_idx][1])])
        return start_points, occl_sizes
    else:
        for record_idx in range(len(coordinates)):
            occl_sizes.append([max(1, int(float(coordinates[record_idx][8])) - int(float(coordinates[record_idx][2]))),
                               max(1, int(float(coordinates[record_idx][7])) - int(float(coordinates[record_idx][1])))])
        return occl_sizes

def create_json(data, key, szene, value_mae, value_rae):
    # with open('evaluation.txt', 'w') as outfile:
    #    data = json.load(data, outfile)
    data[key].append({szene: [str(value_mae), str(value_rae)]})

    # with open('evaluation.txt', 'w') as outfile:
    #    json.dump(data, outfile)
