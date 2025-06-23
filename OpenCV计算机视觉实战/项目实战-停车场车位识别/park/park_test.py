from __future__ import division

import glob
import os
import pickle

import matplotlib.pyplot as plt
from keras.models import load_model

from Parking import Parking

cwd = os.getcwd()


def img_process(test_images, park):
    white_yellow_images = list(map(park.select_rgb_white_yellow, test_images))
    park.show_images(white_yellow_images)

    # BGR图像转换为灰度图
    gray_images = list(map(park.convert_gray_scale, white_yellow_images))
    park.show_images(gray_images)

    # 边缘检测（得到二值图像）
    edge_images = list(map(lambda image: park.detect_edges(image), gray_images))
    park.show_images(edge_images)

    # 手动计算停车场在原图中的范围
    roi_images = list(map(park.select_region, edge_images))
    park.show_images(roi_images)

    # 直线检测：检测出图像中直线所在的位置（停车场的停车线）
    # 注意：此时获取到的直线是图中所有的直线，不一定能是水平的
    list_of_lines = list(map(park.hough_lines, roi_images))

    # 保存已标注停车线的图像列表
    line_images = []
    # 将检测到的停车线绘制到原始图像上。
    # test_images: 原始停车场图像（未经掩膜处理的完整图像）。
    # list_of_lines：对每个ROI（感兴趣区域）图像直线检测结果。
    for image, lines in zip(test_images, list_of_lines):
        # image：对应每张原始图像
        # lines：对应图像中检测到的停车线段列表。
        # draw_lines：过滤霍夫变换检测到直线，只保留水平方向的，水平方向才是代表的停车线位置
        line_images.append(park.draw_lines(image, lines))
    park.show_images(line_images)

    # 将停车场中的位置按照一列一列进行划分（确定具体位置）
    # 存储划分后的图像
    rect_images = []
    # 存储坐标值每列的坐标值
    rect_coords = []
    for image, lines in zip(test_images, list_of_lines):
        # 划分停车场位置
        new_image, rects = park.identify_blocks(image, lines)
        rect_images.append(new_image)
        rect_coords.append(rects)
    park.show_images(rect_images)

    # 切分每个停车位
    # 这块代码需要根据真是业务进行微调切分
    delineated = []
    spot_pos = []
    for image, rects in zip(test_images, rect_coords):
        new_image, spot_dict = park.draw_parking(image, rects)
        delineated.append(new_image)
        spot_pos.append(spot_dict)
    park.show_images(delineated)

    # 获取传递的第一张图像，要存储停车位信息
    final_spot_dict = spot_pos[1]
    print(len(final_spot_dict))

    # 保存停车位字典
    # 将 final_spot_dict（停车位字典）通过 pickle 序列化保存到文件 spot_dict.pickle。
    # pickle.HIGHEST_PROTOCOL 使用最高效的二进制协议
    with open('spot_dict.pickle', 'wb') as handle:
        pickle.dump(final_spot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # 保存每个停车位的图像
    # test_images[0]：第一张停车场图像。
    # final_spot_dict：第一张图片中各个停车位点位信息
    park.save_images_for_cnn(test_images[0], final_spot_dict)
    return final_spot_dict


def keras_model(weights_path):
    # 将路径转换为绝对路径并处理编码
    weights_path = os.path.abspath(weights_path)
    try:
        model = load_model(weights_path, compile=False)
    except UnicodeDecodeError:
        # 尝试GBK编码（Windows系统）
        weights_path = weights_path.encode('utf-8').decode('gbk')
        model = load_model(weights_path, compile=False)
    return model


def img_test(test_images, final_spot_dict, model, class_dictionary):
    for i in range(len(test_images)):
        predicted_images = park.predict_on_image(test_images[i], final_spot_dict, model, class_dictionary)


def video_test(video_name, final_spot_dict, model, class_dictionary):
    name = video_name
    park.predict_on_video(name, final_spot_dict, model, class_dictionary, ret=True)


if __name__ == '__main__':
    # 读取原始停车场图片数据集
    test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
    # 存储最终训练完成的模型（权重、偏置等）
    weights_path = 'car1.h5'
    # 定义最最终要测试的停车场数据
    video_name = 'parking_video.mp4'
    # 存储车位是否被占领的标识
    class_dictionary = {0: 'empty', 1: 'occupied'}
    # 创建Parking对象，调用其中的方法
    park = Parking()
    # 调用展示图片方法，这个地方会将图像集合合并展示
    park.show_images(test_images)
    # 原始图像预处理操作
    final_spot_dict = img_process(test_images, park)
    model = keras_model(weights_path)
    # 使用测试图片来验证模型识别率
    img_test(test_images, final_spot_dict, model, class_dictionary)
    # 使用视频来验证模型识别率
    video_test(video_name, final_spot_dict, model, class_dictionary)
