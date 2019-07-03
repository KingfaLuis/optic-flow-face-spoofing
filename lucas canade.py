# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import sys
import argparse
import time
import logging
from tqdm import tqdm
from skimage import transform as trans
import minicaffe as mcaffe

fd_proto = 'faceboxes_sfzhang15/deploy.prototxt'
fd_model = 'faceboxes_sfzhang15/faceboxes.caffemodel'
fa_proto = 'faceboxes_sfzhang15/48net.prototxt'
fa_model = 'faceboxes_sfzhang15/48net.caffemodel'

land_proto = 'faceboxes_sfzhang15/lnet106_112.prototxt'
land_model = 'faceboxes_sfzhang15/lnet106_112.caffemodel'

def test_yanmo():
    # image = cv2.imread("test_fake.jpg", 0)
    image = cv2.imread("real_me.jpg", 0)
    mask = np.zeros(image.shape, np.uint8)
    mask[20:92, 20:92] = 255
    mask_ellipse=cv2.ellipse(image, (56, 56),  (50, 80), 0, 0, 360, (0, 0, 0), 1)

    mi = cv2.bitwise_and(image, mask)
    # mi = cv2.bitwise_and(image, mask_ellipse)
    histMI = cv2.calcHist([image], [0], mask, [256], [0, 255])
    histImage = cv2.calcHist([image], [0], None, [256], [0, 255])

    plt.hist(histMI, normed=1)
    plt.hist(histImage, normed=1)
    plt.show()

    cv2.imshow('original', image)
    cv2.imshow('mask', mask)
    cv2.imshow('mi', mi)



    image = mi
    mask = np.zeros(image.shape, np.uint8)
    mask[20:92, 20:92] = 255
    histMI = cv2.calcHist([image], [0], mask, [256], [0, 255])
    histImage = cv2.calcHist([image], [0], None, [256], [0, 255])
    plt.plot(histImage)
    plt.plot(histMI)
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()

def test_mask_hist(image):
    # image = cv2.imread("image\\girl.bmp", cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 把图像转为灰度图
    mask = np.zeros(image.shape, np.uint8)
    mask[200:400, 200:400] = 255
    histMI = cv2.calcHist([image], [0], mask, [256], [0, 255])
    histImage = cv2.calcHist([image], [0], None, [256], [0, 255])
    plt.plot(histImage)
    plt.plot(histMI)
    plt.show()

    plt.hist(histMI, normed=1)
    plt.hist(histImage, normed=1)
    plt.show()

def forward_lnet106(img, lm_net, bbox, is_minicaffe=True):
    xmin, ymin, xmax, ymax = bbox
    box_width = xmax - xmin
    box_height = ymax - ymin

    w = box_height if box_height > box_width else box_width
    h = w
    try:
        if box_width != box_height:
            pad_img = np.zeros(shape=(h, w, 3), dtype=float)
            pad_img[0:box_height, 0:box_width] = img[ymin: ymax, xmin: xmax]
        else:
            pad_img = img[ymin: ymax, xmin: xmax]
    except:
        print(xmin, ymin, xmax, ymax)

    lm_image = (pad_img.copy() - 127.5) / 128
    scale_img = cv2.resize(lm_image, (112, 112))
    scale_img = scale_img.swapaxes(1, 2).swapaxes(0, 1)

    lm_net.blobs['data'].reshape(1, 3, 112, 112)
    lm_net.blobs['data'].data[...] = scale_img

    if is_minicaffe:
        lm_net.forward()
    else:
        landmark = lm_net.get_blob('bn6_3').data
        landmark = lm_net.forward()['bn6_3']


    landmark = np.reshape(landmark, (-1, 212)).flatten()

    landmark[::2] = xmin + w * landmark[::2]
    landmark[1::2] = ymin + w * landmark[1::2]

    return landmark

def get_forward_lnet106(img, lm_net, bbox, is_minicaffe=True):
    xmin, ymin, xmax, ymax = bbox
    box_width = xmax - xmin
    box_height = ymax - ymin

    w = box_height if box_height > box_width else box_width
    h = w
    try:
        if box_width != box_height:
            pad_img = np.zeros(shape=(h, w, 3), dtype=float)
            pad_img[0:box_height, 0:box_width] = img[ymin: ymax, xmin: xmax]
        else:
            pad_img = img[ymin: ymax, xmin: xmax]
    except:
        print(xmin, ymin, xmax, ymax)
    lm_image = (pad_img.copy() - 127.5) / 128
    scale_img = cv2.resize(lm_image, (112, 112))
    scale_img = scale_img.swapaxes(1, 2).swapaxes(0, 1)

    lm_net.blobs['data'].reshape(1, 3, 112, 112)
    lm_net.blobs['data'].data[...] = scale_img

    if is_minicaffe:
        lm_net.forward()
        landmark = lm_net.get_blob('bn6_3').data
    else:
        landmark = lm_net.forward()['bn6_3']


    landmark = np.reshape(landmark, (-1, 212)).flatten()

    landmark[::2] = xmin + w * landmark[::2]
    landmark[1::2] = ymin + w * landmark[1::2]

    return landmark

def image_hist(image):     #画三通道图像的直方图

    # img = image
    # hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    # print(type(hist))
    # print(hist.size)
    # print(hist.shape)
    # print(hist)
    #
    # mask = np.zeros(image.shape, np.uint8)
    # mask[200:400, 200:400] = 255
    # histMI = cv2.calcHist([image], [0], mask, [256], [0, 255])
    # histImage = cv2.calcHist([image], [0], None, [256], [0, 255])
    # cv2.imshow('histMI', histMI)
    # cv2.imshow('histImage',histImage)


    # color = ('b', 'g', 'r')   #这里画笔颜色的值可以为大写或小写或只写首字母或大小写混合
    #
    # for i , color in enumerate(color):
    #     hist = cv2.calcHist([image], [i], None, [256], [0, 256])  #计算直方图
    #     plt.plot(hist, color)
    #     plt.xlim([0, 256])
    # plt.show()
    # cv2.imshow("rect" ,image)

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])  # 计算直方图
    plt.plot(hist, 'r')
    plt.xlim([0, 256])
    plt.show()
    cv2.imshow("rect" ,image)

    plt.hist(hist, normed=1)
    plt.show()




def faceboxes_process(fd_net, o_net,landmark_net, img):
    img_width = img.shape[1]
    img_height = img.shape[0]

    im = img.copy()
    im = im.astype(np.float32)
    im[:, :, 0] = im[:, :, 0] - 104.0
    im[:, :, 1] = im[:, :, 1] - 117.0
    im[:, :, 2] = im[:, :, 2] - 123.0
    transformed_image = im.swapaxes(1, 2).swapaxes(0, 1)
    fd_net.blobs['data'].reshape(1, 3, img_height, img_width)
    fd_net.blobs['data'].data[...] = transformed_image
    fd_net.forward()
    detections = fd_net.get_blob('detection_out').data
    print(detections)
    # print(detections)

    bounding_boxes = detections[0, 0, :, 3:7] * np.array([img_width, img_height, img_width, img_height])
    conf = detections[0, 0, :, 2] # 计算出的置信度，在计算熵的时候会用到
    nrof_faces = bounding_boxes.shape[0]
    #print(nrof_faces)
    print(nrof_faces)
    crop_img_list = []
    for i in range(nrof_faces):
        if (conf[i] > 0.5):
            xmin = bounding_boxes[i][0]
            ymin = bounding_boxes[i][1]
            xmax = bounding_boxes[i][2]
            ymax = bounding_boxes[i][3]
            box_w = xmax - xmin
            box_h = ymax - ymin

            box_side = max(box_w, box_h)
            xmin += box_w * 0.5 - box_side * 0.5 ;
            ymin += box_h * 0.5 - box_side * 0.5;
            xmax = xmin + box_side;
            ymax = ymin + box_side;

            box = np.array([max(xmin, 0), max(ymin, 0), min(xmax, img_width), min(ymax, img_height)])
            box = box.astype(np.int32)


            img_ = img.copy()
            crop_img = img_[box[1]:box[3], box[0]:box[2]]
            landmark_img = crop_img.copy()
            test_img = img_[box[1]-20:box[3]+20, box[0]-20:box[2]+20]
            image_hist(test_img)  # 绘制人脸的直方图

            cv2.imshow(test_img)
            cv2.waitKey(1)

            # 在这里画出添加一个区分背景和前景的,论文中使用掩模的方法
            # 这里裁剪的图像的大小是多少呢？
            test_mask_hist(test_img)

            # 画出人脸的点
            src_lmarks = forward_lnet106(img, landmark_net, box, is_minicaffe=True)
            for j in range(3, 30):
                cv2.circle(img, (src_lmarks[2 * j], src_lmarks[2 * j + 1]), 2, (0, 0, 255), -1)

            cv2.imshow('landmark',img)
            # cv2.waitKey(0)

            image_hist(test_img)
            # cv2.imwrite("onet_result_"+ str(i) +".jpg", crop_img)
            cv2.imwrite("onet_result_"+ str(i) +".jpg", crop_img)
            # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB) # 这里没有转灰度图像
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) # 这里转为灰度图像
            crop_img = (crop_img - 127.5) / 128 # 归一化
            scale_img = cv2.resize(crop_img, (48, 48))

            scale_img = np.swapaxes(scale_img, 0, 2)  # hwc chw
            o_net.blobs['data'].data[...] = scale_img
            out = o_net.forward()
            # src_lmarks = out['conv6-3']
            src_lmarks = o_net.get_blob('conv6-3').data # 获取某一层的数据
            src_lmarks = src_lmarks.reshape(2, 5).T

            for j in range(5):
                cv2.circle(landmark_img, (int(src_lmarks[j][0]*landmark_img.shape[1]), int(src_lmarks[j][1]*landmark_img.shape[0])),2,(0,0,255),-1)
                src_lmarks[j][0] = src_lmarks[j][0] * landmark_img.shape[1] + box[0]
                src_lmarks[j][1] = src_lmarks[j][1] * landmark_img.shape[0] + box[1]

            tform = trans.SimilarityTransform()  # 这是相似转换？
            tform.estimate(src_lmarks, dst_lmarks)
            M = tform.params[0:2, :]
            crop_img = cv2.warpAffine(img, M, (112, 112))

            delt_w = (src_lmarks[1][0] - src_lmarks[0][0]) / 4
            delt_h = (src_lmarks[3][1] - src_lmarks[0][1]) / 4
            if src_lmarks[2][0] > src_lmarks[1][0] - delt_w or \
               src_lmarks[2][0] < src_lmarks[0][0] + delt_w or \
               src_lmarks[2][1] > src_lmarks[3][1] - delt_h or \
               src_lmarks[2][1] < src_lmarks[0][1] + delt_h:
                continue
            else:

                for j in range(5):
                    cv2.circle(img, (int(src_lmarks[j][0]), int(src_lmarks[j][1])),2,(0,0,255),-1)

            cv2.ellipse(img, (int((src_lmarks[0][0] + src_lmarks[1][0]) / 2), src_lmarks[2][1]), \
                             (int((box[2] - box[0])/2.2), int((box[3] - box[1])/2)), 0, 0, 360, (255, 255, 255), 1)
            cv2.imshow("landmark", crop_img)
            cv2.imwrite("onet_result_" + str(i) + ".jpg", crop_img)
            print(crop_img.shape)
            crop_img_list.append(crop_img)

            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 5)
            bboxes = box
    return crop_img_list,box

def main():
    fd_net = mcaffe.Net(fd_proto, fd_model)
    o_net = mcaffe.Net(fa_proto, fa_model)
    landmark_net = mcaffe.Net(land_proto, land_model)
    img = cv2.imread('face.jpg')
    # img = cv2.imread('fake_181_2749_0.jpg')
    cv2.imshow('original',img)
    frame = img

    # cap = cv2.VideoCapture(1)
    # while (1):
    #     ret, frame = cap.read()
    #     k = cv2.waitKey(1)
    #     if k == 27:
    #         break
    #     crop_img_list,bbox = faceboxes_process(fd_net, o_net, frame)
    #     for i in range(len(crop_img_list)):
    #         cv2.imwrite("onet_result_" + str(i) + ".jpg", crop_img_list[i])
    #     cv2.imshow("capture", frame)
    # cap.release()

    crop_img_list, bbox = faceboxes_process(fd_net, o_net, landmark_net,frame)



    for i in range(len(crop_img_list)):
        cv2.imwrite("onet_result_" + str(i) + ".jpg", crop_img_list[i])
    cv2.imshow("capture", frame)

    # print(bbox)
    # crop_img_list, bbox = faceboxes_process(fd_net, landmark_net, frame)
    # image = cv2.imread("onet_result_0.jpg", 0)
    # image = cv2.imread("face.jpg", 0)

    # src_lmarks = forward_lnet106(img, landmark_net, bbox, is_minicaffe=True)
    # for j in range(3, 30):
    #    cv2.circle(img, (src_lmarks[2 * j], src_lmarks[2 * j + 1]), 2, (0, 0, 255), -1)
    # cv2.imshow("landmark_106", img)
    # cv2.waitKey(0)

    # land_mark = get_forward_lnet106(img,landmark_net,bbox,True)
    # print(land_mark)

if __name__ == '__main__':
    dst_lmarks = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    parser = argparse.ArgumentParser(description="use faceboxes to detect face")
    parser.add_argument('--filelist', type=str, default='', help='the filelist of input images')
    parser.add_argument('--src_rootpath', type=str, default='', help='the input rootpath of input images')
    parser.add_argument('--dst_rootpath', type=str, default='', help='the output rootpath of output images')

    args = parser.parse_args()

    logging.info(args)
    # main()
    # cal_img_hist()
    # test_yanmo()
    # image = cv2.imread("\\chapter14\\image\\girl.bmp", cv2.IMREAD_GRAYSCALE)
    image = cv2.imread("image\\girl.bmp", cv2.IMREAD_GRAYSCALE)
    # image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # test_mask_hist(image_gray)
    test_yanmo()
    # get_forward_lnet106()
