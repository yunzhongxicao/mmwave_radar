import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
from PIL import Image
import time
from multiprocessing import Process, Manager, Lock
import get_image_3c_const as radar

# 隐藏matplotlib的版本迭代警告
import warnings
import matplotlib.cbook

# 模拟键盘
from pynput.keyboard import Key, Controller

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)



# set path
data_path = "./fake_npy_data_path/"  # define UCF-101 RGB data path
action_name_path = "./UCF101actions.pkl"

save_model_path = "./CRNN_ckpt/detect_and_classifier"
save_result_path = "./classifier_check/wanghl1105"

# use same encoder CNN saved!
# CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
# CNN_embed_dim = 512  # latent dim extracted by 2D CNN
# img_x, img_y = 256, 342  # resize video 2d frame size
# dropout_p = 0.0  # dropout probability
#
# # use same decoder RNN saved!
# RNN_hidden_layers = 3
# RNN_hidden_nodes = 512
# RNN_FC_dim = 256

CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
# CNN_embed_dim = 512  # latent dim extracted by 2D CNN
CNN_embed_dim = 256  # dim extracted by lenet ?? how to set this value
# img_x, img_y = 256, 342  # resize video 2d frame size
img_x, img_y = 32, 32  # Data_RD_FFT size
dropout_p = 0.0  # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k_detector = 2  # number of target category
k_classifier = 11  # number of target category
batch_size = 1
# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 50, 1

frames_to_sample = 1600
slip_length = 4
window_scale = 4

learning_rate = 1e-4
log_interval = 1  # interval for displaying training info

# with open(action_name_path, 'rb') as f:
#     action_names = pickle.load(f)   # load UCF101 actions names

action_names = ['action', 'none']

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)

actions = []
fnames = os.listdir(data_path)

all_names = []
for f in fnames:
    loc1 = f.find('v_')
    loc2 = f.find('_g')
    actions.append(f[(loc1 + 2): loc2])
    all_names.append(f)

# list all data files
all_X_list = all_names  # all video file names
all_y_list = labels2cat(le, actions)  # all video labels


# prepare data to matrix with windows
# 这里，每一个frame是一个平面，但是最后是按照frame堆叠成了体。
# 一个体，是每一次预测数据的最小元素。
def mk_time_data(image_data, slip_length, window_scale):
    data_length = len(image_data[0])
    if window_scale > data_length:
        print('The data length is too small.')

    image_seq = []
    for num in range(0, data_length - window_scale + 1, slip_length):
        image_seq.append(image_data[0][num:num + window_scale, :, :])

    image_seq = torch.stack(image_seq, axis=0)
    return image_seq


# prepare data to matrix with windows
def mk_action_data(image_data, frame_start, frame_end):
    how_many_seq = len(frame_start)

    image_matrix = []
    for num in range(0, how_many_seq):
        image_seq = []
        for i in range(3):
            image_seq.append(image_data[i][frame_start[num]:frame_end[num]])
        image_matrix.append(image_seq)

    return image_matrix


# prepare data to matrix with windows
# 对于三个通道来说，每个通道都是list中的一个元素，每一个又是一个3d的张量
# 因为这里每一次分类仅仅有一个单位，因此需要直接用一个list将一个识别单位封装起来
def mk_action_data_for_lenet(image_data, frame_start, frame_end):
    # image_matrix = []
    image_seq = []
    image_data_lenth = len(image_data[2])
    for i in range(3):
        image_seq.append(image_data[i][frame_start:frame_end+1, :, :])
    # image_matrix.append(image_seq)
    image_matrix = [image_seq]

    return image_matrix


def to_real_frames(frame_start, frame_end, slip_length, filter_level):
    end_flag = False
    for i in range(len(frame_start)):
        try:
            # frame_start[i] = slip_length * (frame_start[i] - filter_level)
            # frame_end[i] = slip_length * (frame_end[i] - filter_level) + 2
            frame_start[i] = slip_length * (frame_start[i] - filter_level)
            frame_end[i] = slip_length * (frame_end[i] - filter_level) + 2
        except:
            print('action end at the end!')
            end_flag = True
        else:
            pass
            # todo 这里处理一下，目前是直接退出

    if end_flag:
        frame_start.pop()

    return frame_start, frame_end


def mkpath(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return path
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return path


def save_images(image_list, root_path):
    mkpath(root_path)
    frame_number = 1
    for i in range(0, len(image_list)):
        img = image_list[i]
        index = str(frame_number)
        for i in range(6 - len(index)):
            index = '0' + index
        index = r'\frame' + index
        img.save(root_path + index + '.jpg')
        frame_number = frame_number + 1


def detector(shared_switch_flag, image_list_raw):
    transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                    transforms.ToTensor()])
    selected_frames = np.arange(1, 20, 1).tolist()

    use_cuda = False  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    # params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    CNN_embed_dim_ = 256
    cnn_encoder = LeNetEncoder(CNN_embed_dim=CNN_embed_dim_).to(device)
    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim_, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                             h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k_detector).to(device)
    if use_cuda:
        cnn_encoder.load_state_dict(
            torch.load(os.path.join(save_model_path, 'lenet_1a_encoder.pth')))
        rnn_decoder.load_state_dict(
            torch.load(os.path.join(save_model_path, 'lenet_lstm_1a_encoder.pth')))
    else:
        cnn_encoder.load_state_dict(
            torch.load(os.path.join(save_model_path, 'lenet_1a_encoder.pth'), map_location=torch.device('cpu')))
        rnn_decoder.load_state_dict(
            torch.load(os.path.join(save_model_path, 'lenet_lstm_1a_encoder.pth'), map_location=torch.device('cpu')))
    print('CRNN detector model reloaded!')

    scaned_index = 0
    last_prediction = 1  # 1是none
    last2_prediction = 1
    last3_prediction = 1
    last4_prediction = 1
    last5_prediction = 1
    pre_time = -1
    while True:
        try:
            full_index = image_list_raw[2].shape[0]-1
            # print("image_list length is " + str(image_list_raw[0].shape[0]))
            # print('In image_list apeared images.')
        except:
            full_index = 0
            # print('has no img now.')


        if full_index - scaned_index > slip_length-2:
            # print('there are more than 4 image still no scan')
            image_list = [[], [], []]
            image_list[0] = image_list_raw[0][scaned_index:full_index + 1, :, :]
            image_list[1] = image_list_raw[1][scaned_index:full_index + 1, :, :]
            image_list[2] = image_list_raw[2][scaned_index:full_index + 1, :, :]

            image_matrix = mk_time_data(image_list, slip_length, window_scale)

            # 这里开始动作存在性检测

            detect_times = 0
            while detect_times < image_matrix.shape[0]:
                # print('detect time: ' + str(detect_times+1))

                # reset data loader
                all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4,
                                   'pin_memory': True} if use_cuda else {}
                all_data_loader = data.DataLoader(
                    Dataset_realtime2([image_matrix[detect_times]], all_y_list, selected_frames, transform=transform),
                    **all_data_params)

                all_y_pred, all_y_pred_value = CRNN_final_prediction_with_value([cnn_encoder, rnn_decoder], device,
                                                                                all_data_loader)

                # make all video predictions by reloaded model
                # print('Predicting all {} videos:'.format(len(all_data_loader.dataset)))
                # try:
                #     all_y_pred, all_y_pred_value = CRNN_final_prediction_with_value([cnn_encoder, rnn_decoder], device,
                #                                                                 all_data_loader)
                # except:
                #     print("predict failed...")
                #     pass
                # else:
                #     pass

                if (pre_time != -1):
                    f_frame = 1 / (time.time() - pre_time)
                    pre_time = time.time()
                else:
                    f_frame = 0
                    pre_time = time.time()

                # print('frequency: ' + str(f_frame))

                # print(str(cat2labels(le, all_y_pred)) + " frames: " + str(scaned_index) + " to " + str(full_index))
                # print(str(cat2labels(le, all_y_pred)))
                last_number = 2
                if last_number == 1:
                    if last_prediction == 1 and all_y_pred[0] == 0:
                        if shared_switch_flag[1] == 0:  # 必须要在关闭时才能开始
                            shared_switch_flag[0] = scaned_index - 0 * slip_length + slip_length * detect_times
                    if last_prediction == 0 and all_y_pred[0] == 1:
                        if shared_switch_flag[0] != 0:  # 必须要在开始之后才能结束
                            shared_switch_flag[1] = scaned_index - 0 * slip_length + slip_length * detect_times

                if last_number == 2:
                    if last2_prediction == 1 and last_prediction == 0 and all_y_pred[0] == 0:
                        if shared_switch_flag[1] == 0:  # 必须要在关闭时才能开始
                            shared_switch_flag[0] = scaned_index - 1 * slip_length + slip_length * detect_times
                    if last2_prediction == 0 and last_prediction == 1 and all_y_pred[0] == 1:
                        if shared_switch_flag[0] != 0:  # 必须要在开始之后才能结束
                            shared_switch_flag[1] = scaned_index - 1 * slip_length + slip_length * detect_times

                if last_number == 3:
                    if last3_prediction == 1 and last2_prediction == 0 and last_prediction == 0 and all_y_pred[0] == 0:
                        if shared_switch_flag[1] == 0:  # 必须要在关闭时才能开始
                            shared_switch_flag[0] = scaned_index - 2 * slip_length + slip_length * detect_times
                    if last3_prediction == 0 and last2_prediction == 1 and last_prediction == 1 and all_y_pred[0] == 1:
                        if shared_switch_flag[0] != 0:  # 必须要在开始之后才能结束
                            shared_switch_flag[1] = scaned_index - 2 * slip_length + slip_length * detect_times

                if last_number == 4:
                    if last4_prediction == 1 and last3_prediction == 0 and last2_prediction == 0 and last_prediction == 0 and all_y_pred[0] == 0:
                        if shared_switch_flag[1] == 0:  # 必须要在关闭时才能开始
                            shared_switch_flag[0] = scaned_index - 3 * slip_length + slip_length * detect_times
                    if last4_prediction == 0 and last3_prediction == 1 and last2_prediction == 1 and last_prediction == 1 and all_y_pred[0] == 1:
                        if shared_switch_flag[0] != 0:  # 必须要在开始之后才能结束
                            shared_switch_flag[1] = scaned_index - 3 * slip_length + slip_length * detect_times

                if last_number == 42:
                    if last4_prediction == 1 and last3_prediction == 0 and last2_prediction == 0 and last_prediction == 0 and all_y_pred[0] == 0:
                        if shared_switch_flag[1] == 0:  # 必须要在关闭时才能开始
                            shared_switch_flag[0] = scaned_index - 3 * slip_length + slip_length * detect_times
                    if last2_prediction == 0 and last_prediction == 1 and all_y_pred[0] == 1:
                        if shared_switch_flag[0] != 0:  # 必须要在开始之后才能结束
                            shared_switch_flag[1] = scaned_index - 1 * slip_length + slip_length * detect_times

                if last_number == 99:
                    tmode = '00'
                    prediction_result = str(last4_prediction) + str(last3_prediction) + str(last2_prediction) + str(last_prediction) + str(all_y_pred[0])
                    if prediction_result == '00000':
                        tmode = '40'
                    elif prediction_result == '00001':
                        tmode = '40'
                    elif prediction_result == '00011':
                        tmode = '40'
                    elif prediction_result == '00111':
                        tmode = '02'
                    elif prediction_result == '01111':
                        tmode = '03'
                    elif prediction_result == '11111':
                        tmode = '04'
                    elif prediction_result == '00010':
                        tmode = '40'
                    elif prediction_result == '00100':
                        tmode = '40'
                    elif prediction_result == '01000':
                        tmode = '40'
                    elif prediction_result == '10000':
                        tmode = '40'
                    elif prediction_result == '00110':
                        tmode = '00'
                    elif prediction_result == '01100':
                        tmode = '00'
                    elif prediction_result == '11000':
                        tmode = '30'
                    elif prediction_result == '01110':
                        tmode = '30'
                    elif prediction_result == '11000':
                        tmode = '30'
                    elif prediction_result == '11000':
                        tmode = '30'

                    if tmode[0] != '0':
                        if shared_switch_flag[0] == 0:  # 已经打开了就不要打开了
                            shared_switch_flag[0] = scaned_index - int(tmode[0]) * slip_length + slip_length * detect_times
                    if tmode[1] != '0':
                        if shared_switch_flag[1] == 0:  # 已经关闭就不要再关闭了
                            shared_switch_flag[1] = scaned_index - int(tmode[1]) * slip_length + slip_length * detect_times

                last5_prediction = last4_prediction
                last4_prediction = last3_prediction
                last3_prediction = last2_prediction
                last2_prediction = last_prediction
                last_prediction = all_y_pred[0]
                detect_times += 1

            scaned_index = full_index + 1



def classifier(shared_switch_flag, image_list):
    # 键盘类
    keyboard = Controller()

    # training parameters
    batch_size = 1
    # Select which frame to begin & end in videos
    begin_frame, end_frame, skip_frame = 1, 50, 1
    # with open(action_name_path, 'rb') as f:
    #     action_names = pickle.load(f)   # load UCF101 actions names
    # action_names = ['into', 'out', 'check', 'cross', 'right', 'left']
    action_names = ['right', 'left', 'push', 'pull', 'clockwise',
                    'anti_clock', 'pinch', 'double_push', 'check', 'cross', 'move_finger']
    # action_names = ['right', 'left']
    # convert labels -> category
    le = LabelEncoder()
    le.fit(action_names)
    # show how many classes there are
    list(le.classes_)
    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)

    # actions = ['into', 'out', 'check', 'cross']
    actions = action_names  # Copy
    all_y_list = labels2cat(le, actions)
    # reload CRNN model
    use_cuda = False  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    # params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    cnn_encoder_action = LeNetEncoder_3x(CNN_embed_dim=CNN_embed_dim).to(device)
    RNN_hidden_layers_ = 3
    rnn_decoder_action = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers_, h_RNN=RNN_hidden_nodes,
                                    h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k_classifier).to(device)
    if use_cuda:
        cnn_encoder_action.load_state_dict(
            torch.load(os.path.join(save_model_path, 'lenet_3a_encoder.pth')))
        rnn_decoder_action.load_state_dict(
            torch.load(os.path.join(save_model_path, 'lenet_lstm_3a_encoder.pth')))
    else:
        cnn_encoder_action.load_state_dict(
            torch.load(os.path.join(save_model_path, 'lenet_3a_encoder.pth'),
                       map_location=torch.device('cpu')))
        rnn_decoder_action.load_state_dict(
            torch.load(os.path.join(save_model_path, 'lenet_lstm_3a_encoder.pth'),
                       map_location=torch.device('cpu')))
    print('CRNN action model reloaded!')

    classifier_number = 0
    while True:
        if shared_switch_flag[0] > 0 and shared_switch_flag[1] > 0:
            # print("#### classifier start and end frames:" + str(shared_switch_flag[0]) + " " + str(shared_switch_flag[1]))
            pre_time = time.time()

            current_list = mk_action_data_for_lenet(image_list, shared_switch_flag[0], shared_switch_flag[1])

            transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                            transforms.ToTensor()])
            selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
            # reset data loader
            all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4,
                               'pin_memory': True} if use_cuda else {}
            all_data_loader = data.DataLoader(
                Dataset_realtime_3c(current_list, all_y_list, selected_frames, transform=transform), **all_data_params)
            # make all video predictions by reloaded model
            # print('Predicting all {} videos:'.format(len(all_data_loader.dataset)))
            all_y_pred = CRNN_3c_final_prediction_realtime([cnn_encoder_action, rnn_decoder_action],
                                                                             device,
                                                                             all_data_loader)
            now_time = time.time()
            actions_pre_result = cat2labels(le, all_y_pred)
            print(' ')
            print("#################### this is the action name: " + str(actions_pre_result))
            print(' ')
            # print('classifier fre:' + str(1/ (now_time - pre_time)))
            #
            # 控制键盘
            if actions_pre_result[0] == 'left':
                keyboard.press(Key.left)
                keyboard.release(Key.left)
            elif actions_pre_result[0] == 'right':
                keyboard.press(Key.right)
                keyboard.release(Key.right)
            elif actions_pre_result[0] == 'clockwise':
                keyboard.press(Key.f5)
                keyboard.release(Key.f5)
            elif actions_pre_result[0] == 'anti_clock':
                keyboard.press(Key.esc)
                keyboard.release(Key.esc)

            # 是否保存分类的数据用以检查
            need_save_check = False
            need_pic_save = False
            if need_save_check == True:
                if classifier_number %2 == 0:
                    # result_path = mkpath(os.path.join(save_result_path, 'classifier_result{:02d}'.format(classifier_number)))
                    # action_names = ['right', 'left', 'push', 'pull', 'clockwise',
                    #                 'anti_clock', 'pinch', 'double_push', 'check', 'cross', 'move_finger']
                    action_name = 'v_cross_g_wanghaili_auto01_' + str(classifier_number)
                    root_path_classifier_1 = os.path.join(save_result_path, action_name, 'dem1')
                    root_path_classifier_2 = os.path.join(save_result_path, action_name, 'dem2')
                    root_path_classifier_3 = os.path.join(save_result_path, action_name, 'dem3')

                    mkpath(root_path_classifier_1)
                    mkpath(root_path_classifier_2)
                    mkpath(root_path_classifier_3)
                    np.save(root_path_classifier_1 + '/data_rd_fft.npy', current_list[0][0])
                    np.save(root_path_classifier_2 + '/data_rd_fft.npy', current_list[0][1])
                    np.save(root_path_classifier_3 + '/data_rd_fft.npy', current_list[0][2])

                    if need_pic_save == True:
                        for i in range(current_list[0][0].shape[0]):
                            figure = plt.figure()
                            # plt.subplot(1, 1, 1)
                            pcm = plt.pcolormesh(current_list[0][0][i, :, :],
                                           cmap='plasma', vmin=0)  # 'PuBu_r'
                            plt.colorbar(pcm)
                            plt.title("Figure %d " % (i))
                            plt.draw()

                            figure.savefig(root_path_classifier_2 + '/frame{:06d}.jpg'.format(i + 1))
                            plt.close()

            classifier_number += 1
            shared_switch_flag[0] = 0
            shared_switch_flag[1] = 0


if __name__ == '__main__':
    # 获得一段数据
    # image_data = radar.get_image(300)
    # image_matrix = mk_time_data(image_data, 10, 50)
    # image_list = radar.get_image(frames_to_sample) #frames_to_sample没有作用了
    # radar.get_image(frames_to_sample)
    # image_list = radar.img_matrix

    # data loading parameters
    # use_cuda = torch.cuda.is_available()  # check if GPU exists

    manager = Manager()
    # shared_img_list = manager.list()
    shared_switch_flag = manager.list([0, 0])
    shared_raw_list = manager.list()
    shared_radar_running_flag = manager.list([0])
    shared_img_list = manager.list([[], [], []])

    process_list = []
    # radar进程开启
    frames_number = frames_to_sample
    thread_radar = Process(target=radar.get_image,
                           args=(frames_number, shared_raw_list, shared_radar_running_flag, shared_img_list))
    thread_radar.start()
    process_list.append(thread_radar)
    # detector进程开启
    thread_detector = Process(target=detector,
                              args=(shared_switch_flag, shared_img_list))
    thread_detector.start()
    process_list.append(thread_detector)

    # classifier进程开启
    thread_classifier = Process(target=classifier,
                                args=(shared_switch_flag, shared_img_list))
    thread_classifier.start()
    process_list.append(thread_classifier)

    # 关闭已开启的所有进程
    # print('there are ' + str(len(process_list)) + ' processes have been started.')
    # for res in process_list:
    #     res.join()
    # classifier(shared_switch_flag[0], shared_switch_flag[1], shared_img_list)

    # actions_last_time = []
    # for index in range(len(real_frame_start)):
    #     action_last_time = real_frame_end[index] - real_frame_start[index]
    #     actions_last_time.append(action_last_time)
    #
    # # 增加动作分类结果标识
    # for index, result in enumerate(actions_pre_result):
    #     if index != 0 and index != len(actions_pre_result) - 1:
    #         if actions_last_time[index] < 25:
    #             result = '---'
    #     plt.text(old_frame_start[index], 0.9, result, fontsize=10)
    #
    # # 显示结果
    # plt.show()
    #
    #
    # # # write in pandas dataframe
    # # df = pd.DataFrame(data={'filename': fnames, 'y': cat2labels(le, all_y_list), 'y_pred': cat2labels(le, all_y_pred)})
    # # df.to_pickle("./check_predictions/UCF101_videos_prediction.pkl")  # save pandas dataframe
    # # # pd.read_pickle("./all_videos_prediction.pkl")
    # print('video prediction finished!')
    #
    # choice = input("是否保存本次测试的图片数据（y/n）:")
    # if choice == 'y':
    #     root_path = './test_image'
    #     root_path1 = os.path.join(root_path, 'dem1')
    #     root_path2 = os.path.join(root_path, 'dem2')
    #     root_path3 = os.path.join(root_path, 'dem3')
    #
    #     save_images(image_list[0], root_path1)
    #     save_images(image_list[1], root_path2)
    #     save_images(image_list[2], root_path3)
    # else:
    #     print('程序退出...')
    #     sys.exit(0)
    #
    # choice2 = input("是否进行分开保存（y/n）:")
    # if choice2 == 'y':
    #     choice3 = input("请输入分段号（1，2，3 ....）:")
    #     choice3 = int(choice3) - 1
    #
    #     root_path = './test_image_divided'
    #     root_path1 = os.path.join(root_path, 'dem1')
    #     root_path2 = os.path.join(root_path, 'dem2')
    #     root_path3 = os.path.join(root_path, 'dem3')
    #
    #     save_images(image_list[0][real_frame_start[choice3]:real_frame_end[choice3]], root_path1)
    #     save_images(image_list[1][real_frame_start[choice3]:real_frame_end[choice3]], root_path2)
    #     save_images(image_list[2][real_frame_start[choice3]:real_frame_end[choice3]], root_path3)
    # else:
    #     print('程序退出...')
    #     sys.exit(0)
