#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
from VO import MonoVo as VO
import os


classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor']
colors = np.random.uniform(0, 255, size=(len(classes), 3))


def recognize(image, confidence_thresh: float):
    '''
    recognize takes an image and draws boxes on it in place then passes
    it back to the calling function

    Parameters
    ----------
    image - The opencv read matrix image
    confidence_thresh (float) - The confidence under which we filter detections
    '''
    network = cv2.dnn.readNetFromCaffe(
            './caffe_model.prototxt.txt', './model.caffemodel')

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (800, 800)), 0.007843, (800, 800), 127.5)

    print('[INFO] Detecting...')
    network.setInput(blob)
    detections = network.forward()

    # Get all the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > confidence_thresh:
            # Get the class label, then place the box
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype('int')

            # show prediction
            label = '{}: {:.2f}%'.format(classes[idx], confidence * 100)
            cv2.rectangle(
                    image, (start_x, start_y), (end_x, end_y), colors[idx], 2)
            y = start_y - 15 if start_y - 15 > 15 else start_y + 15
            cv2.putText(image,
                        label,
                        (start_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
    return image


def run(image_path, pose_path, focal, pp, R_total, t_total):
    lk_params = dict(winSize=(21, 21),
                     criteria=(
                         cv2.TERM_CRITERIA_EPS |
                         cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    vo = VO(image_path, pose_path, focal, pp, lk_params)
    traj = np.zeros(shape=(600, 800, 3))

    while(vo.has_next_frame):
        frame = recognize(vo.current_frame, 0.7)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

        if k == 121:
            mask = np.zeros_like(vo.old_frame)
            mask = np.zeros_like(vo.current_frame)

        vo.process_frame()

        print(vo.get_mono_coordinates())

        mono_coord = vo.get_mono_coordinates()
        true_coord = vo.get_true_coordinates()

        print('MSE Error: ', np.linalg.norm(mono_coord - true_coord))
        print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in mono_coord]))
        print("true_x: {}, true_y: {}, true_z: {}".format(
            *[str(pt) for pt in true_coord]))

        draw_x, draw_y, draw_z = [int(round(x)) for x in mono_coord]
        true_x, true_y, true_z = [int(round(x)) for x in true_coord]

        traj = cv2.circle(
                traj, (true_x + 400, true_z + 100), 1, list((0, 0, 255)), 4)
        traj = cv2.circle(
                traj, (draw_x + 400, draw_z + 100), 1, list((0, 255, 0)), 4)

        cv2.putText(
                traj,
                'Actual Position:',
                (140, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255), 1)

        cv2.putText(
                traj,
                'Red',
                (270, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255), 1)

        cv2.putText(
                traj,
                'Estimated Odometry Position:',
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255), 1)

        cv2.putText(
                traj,
                'Green',
                (270, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0), 1)

        cv2.imshow('trajectory', traj)
    cv2.imwrite("./images/trajectory.png", traj)

    cv2.destoryAllWindows()


if __name__ == '__main__':
    pose_path = '../poses/dataset/poses/00.txt'
    image_path = '../gray/dataset/sequences/00/image_0/'
    focal = 718.8560
    pp = (607.1928, 185.2157)
    R_total = np.zeros((3, 3))
    t_total = np.empty(shape=(3, 1))

    run(image_path, pose_path, focal, pp, R_total, t_total)
