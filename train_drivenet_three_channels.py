from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
from train_utils import *
from drive_net import DriveNet
import json

# from train_utils import crop_img_for_drivenet, evaluate_angle, evaluate_on_generated_imgs, parse_logfile, \
#    prepare_imgs_for_drivenet, get_sorted_list_of_raw_imgs, resize_coordinates_after_crop, get_occl_size_mod, \
#    save_information, normalize


with open('config_data_home_two_paths.txt') as json_file:
    parameters = json.load(json_file)

os.environ["CUDA_VISIBLE_DEVICES"] = str(parameters['gpu'])
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
FOLDER_COUNTER = 0
drive_net_dataset = 0
cwd = os.getcwd()

print("--------INITIALIZE MODEL-------")
model = DriveNet()

# for raw_imgs_path in parameters['raw_imgs_path']:
for raw_imgs_path in parameters['raw_imgs_path'][2:3]:
    raw_imgs_path = parameters['vanishing_folder'] + raw_imgs_path
    print(raw_imgs_path)
    # os.chdir(cwd)
    # print(raw_imgs_path)
    # if FOLDER_COUNTER > 15:
    # print(raw_imgs_path, 'TEST FOLDER COUNT', FOLDER_COUNTER)
    drive_net_dataset = 1
    print("--------COORDINATES READ-------")
    with open(raw_imgs_path + parameters['coordinates_path']) as f:
        content_coordinates = f.readlines()
    coordinates = parse_logfile(content_coordinates)
    # print(coordinates)

    print("--------LOG FILE READ-------")
    with open(raw_imgs_path + parameters['log_info_path']) as f:
        content_log_file = f.readlines()
    log_file = parse_logfile(content_log_file)
    # print(log_file)

    print("--------IMG NAMES READ-------")
    img_array = get_sorted_list_of_raw_imgs(raw_imgs_path)
    # print(img_array)

    print("--------IMG SORT THREE CHANNELS-------")
    drivenet_inputs_mapping = prepare_imgs_for_drivenet(log_file, use_same_img=False)

    print("--------RESIZE COORDINATES-------")
    coordinates_resized = resize_coordinates_after_crop(coordinates, False, drive_net_dataset)
    # print(coordinates_resized)

    patch_start_coords, occl_sizes = get_occl_size_mod(coordinates_resized, False)
    occl_sizes_raw = get_occl_size_mod(coordinates, True)

    print("--------IMG PREPARE-------")
    imgs, orig_shape = crop_img_for_drivenet(img_array, drive_net_dataset)
    if parameters['show_imgs_for_debug']:
        for i in range(len(imgs)):
            plt.imshow(imgs[drivenet_inputs_mapping[i][0]][0])
            plt.show()
    img_size = [orig_shape[0], orig_shape[1]]

    # cwd = os.getcwd()
    # os.chdir(raw_imgs_path+parameters['predicted_angles_file_path'])
    # FOLDER_COUNTER += 1

    print("--------PREDICT ANGLE AND GRAD-------")
    angle_without_attack = evaluate_angle(model, imgs, drivenet_inputs_mapping, log_file)

    print("--------TRAINNING START-------")
    patch = np.zeros((parameters['logo_height'], parameters['logo_width'], 3))

    img_indices = np.arange(len(imgs), dtype=np.int32)
    imgs = np.array(imgs)
    tmp_imgs = copy.deepcopy(imgs)
    diff_in_last_batch = 0  # the total difference in last minibatch
    change_times = 0  # the total change times of logo in ONE ITERATION
    bad_change_times = 0

    for num_iter in range(parameters['grad_iterations']):
        change_times = 0
        bad_change_times = 0
        if parameters['greedy_stratage'] != 'sequence_fix':
            np.random.shuffle(img_indices)

        # iterate over three images which form a single input to drivenet
        for drivenet_input_idx in range(3):
            print(drivenet_input_idx, '. channel')
            for img_idx in range(0, len(imgs), parameters['batch_size']):
                if img_idx <= len(imgs) - parameters['batch_size']:
                    minibatch = [imgs[img_indices[j]] for j in range(img_idx, img_idx + parameters['batch_size'])]
                else:
                    minibatch = [imgs[img_indices[j]] for j in range(img_idx, len(imgs))]
                    # Diese Funktion ist dafuer da um die letzten aufnahmen zu einem minibatch zu machen
                    # denn gegen Ende ist, wenn die Aufnahmenanzahl nich n*batch entspricht das letzte batch
                    # mit weniger Elementen
                patch_for_current_batch = np.zeros((parameters['batch_size'],
                                                    parameters['logo_height'],
                                                    parameters['logo_width'], 3))
                count = 0
                count_index = 0

                # hier wird gen_img nicht verwendet darum muss das noch rausgenommen werden evtl. so:
                # for value in range(0, minibatch):
                for value, gen_img in enumerate(minibatch):
                    # Hier wird der gradient ausgerechnet
                    gradient_for_current_img = model.predict(img_indices[value +
                                                                         (parameters['batch_size'] * count_index)],
                                                             drivenet_inputs_mapping, imgs, log_file, False, True)
                    grads_value = normalize(
                        [gradient_for_current_img[1][drivenet_input_idx]])  # Hier wird eines der channels ausgewaehlt
                    if parameters['transformation'] == 'occl':
                        grads_value = constraint_occl(grads_value, patch_start_coords[
                            drivenet_inputs_mapping[img_indices[img_idx + count]][drivenet_input_idx]],
                                                      # grads_value reduziert sich auf den
                                                      # Bereich des Plakats im channel
                                                      occl_sizes[drivenet_inputs_mapping[img_indices[img_idx + count]][
                                                          drivenet_input_idx]])  # constraint the gradients value
                    if parameters['jsma']:
                        k_th_value = find_kth_max(grads_value, parameters['jsma_n'])
                        super_threshold_indices = abs(grads_value) < k_th_value
                        # Alle gradienten dessen absolut geringer als ein bestimmter Wert sind werden zu 0 gesetz
                        grads_value[super_threshold_indices] = 0
                    patch_for_current_batch = transform_occl3(grads_value, patch_start_coords[
                        drivenet_inputs_mapping[img_indices[img_idx + count]][drivenet_input_idx]],
                                                              occl_sizes[
                                                                  drivenet_inputs_mapping[img_indices[img_idx + count]][
                                                                      drivenet_input_idx]],
                                                              patch_for_current_batch,
                                                              count)
                    # count entspricht dem countten wert des batch grossen log_data
                    count += 1
                count_index += 1
                if parameters['overlap_strategy'] == 'sum':
                    patch_for_current_batch = np.sum(patch_for_current_batch, axis=0)
                tmp_logo = patch_for_current_batch * parameters['step'] + patch
                tmp_logo = control_bound(tmp_logo)
                tmp_imgs = update_image_tmp(tmp_imgs, tmp_logo, patch_start_coords, occl_sizes)
                # If this minibatch generates a higher total difference we will consider this one.
                this_diff = total_diff(tmp_imgs, model, angle_without_attack, drivenet_inputs_mapping, log_file)
                # print("iteration ",iters,". batch count ",i,".
                # this time diff ",this_diff,". last time diff ", last_diff)
                if this_diff > diff_in_last_batch:
                    patch += patch_for_current_batch * parameters['step']
                    patch = control_bound(patch)  # Hier evtl. noch bearbeitungen vornehmen
                    imgs = update_image_imgs(imgs, patch, patch_start_coords, occl_sizes)
                    diff_in_last_batch = this_diff
                    change_times += 1
                else:
                    bad_change_times += 1

            # Hier kommt er nur rein wenn alle batches fuer ein channel evaluiert wurden
            gray_angle_diff = 0  # type: int
            for img_idx in range(0, len(log_file)):
                predicted_value_new = model.predict(img_idx, drivenet_inputs_mapping, imgs, log_file, True, False)
                angle_final = predicted_value_new[0].angle
                gray_angle_diff += abs(angle_final - angle_without_attack[img_idx])

            if num_iter % 1 == 0:
                print("iteration ", num_iter, ". diff between raw and adversarial",
                      gray_angle_diff[0],
                      ". change time is,", change_times, ". no_change_times,", bad_change_times)
            if num_iter % 1 == 0:
                # np.save('./train_output/' + str(iters) + 'new_logo.npy', logo)
                imsave(raw_imgs_path + parameters['predicted_angles_file_path'] + 'iters' + str(num_iter) + 'channel' +
                       str(drivenet_input_idx) + 'new_logo.png',
                       deprocess_image(patch, shape=(parameters['logo_height'], parameters['logo_width'], 3)))
    FOLDER_COUNTER += 1
    print("--------TRAINNING COMPLETE-------")

    evaluate_on_generated_imgs(raw_imgs_path, model, imgs, drivenet_inputs_mapping, angle_without_attack, log_file)
    save_information(raw_imgs_path, len(coordinates) - 1, parameters['logo_width'], parameters['logo_height'],
                     occl_sizes_raw[0][0], occl_sizes_raw[0][1], occl_sizes_raw[len(coordinates) - 1][0],
                     occl_sizes_raw[len(coordinates) - 1][1], gray_angle_diff[0])