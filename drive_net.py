from keras import backend as K
import numpy as np
import tensorflow as tf
import json


with open('config_data_home.txt') as json_file:
    parameters = json.load(json_file)

# print("Built with CUDA?", tf.test.is_built_with_cuda())
# print("GPU available?", tf.test.is_gpu_available())


class DrivingParameters(object):
    def __init__(self, angle, speed, confidence=np.zeros([10, 1040, 3])):
        self.angle = angle
        self.speed = speed
        self.confidence = confidence


class DriveNet(object):

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Hier wird glaube ich das default graph geaendert und mit dem gespeicherten graph ueberschrieben
            tf.train.import_meta_graph(parameters['graph_file'], clear_devices=True)
            saver = tf.train.Saver(write_version=2)
        config = tf.ConfigProto()
        self.session = tf.Session(graph=self.graph, config=config)
        saver.restore(self.session, parameters['checkpoint_file'])
        self.image_input = self.graph.get_collection('image_input')[0]
        self.steering_angle = self.graph.get_collection('steering_angle')[0]
        self.indicator_input = None
        self.speed_input = None

    def predict(self, img_idx, drivenet_inputs_mapping, imgs_list, log_file, calculate_angle, calculate_grad):
        # plt.imshow(img[sorted_images[i][0]][0])
        # plt.show()
        if len(self.graph.get_collection('indicator_input')) > 0:
            self.indicator_input = self.graph.get_collection('indicator_input')[0]
        if len(self.graph.get_collection('speed_input')) > 0:
            self.speed_input = self.graph.get_collection('speed_input')[0]
        net_in = {self.image_input: [np.stack(
            [imgs_list[drivenet_inputs_mapping[img_idx][0]][0], imgs_list[drivenet_inputs_mapping[img_idx][1]][0],
             imgs_list[drivenet_inputs_mapping[img_idx][2]][0]])]}
        if self.indicator_input is not None:
            net_in[self.indicator_input] = [[int(log_file[img_idx][4]) * 100, int(log_file[img_idx][5]) * 100]]
        if self.speed_input is not None:
            net_in[self.speed_input] = [[log_file[img_idx][3]]]

        if calculate_angle and not calculate_grad:
            net_output = self.session.run(self.steering_angle, feed_dict=net_in)
            # print(DrivingParameters(net_output[0], -1.0).angle)
            return [DrivingParameters(net_output[0], -1.0), 0]
        elif not calculate_angle and calculate_grad:
            net_output_2 = self.session.run(K.gradients(self.steering_angle, self.image_input), feed_dict=net_in)
            return [0, net_output_2[0][0]]
        elif calculate_angle and calculate_grad:
            net_output = self.session.run(self.steering_angle, feed_dict=net_in)
            net_output_2 = self.session.run(K.gradients(self.steering_angle, self.image_input), feed_dict=net_in)
            # print(DrivingParameters(net_output[0], -1.0).angle)
            return [DrivingParameters(net_output[0], -1.0), net_output_2[0][0]]
        else:
            return [0, 0]
