# -*- coding: utf-8 -*-
"""
Time    : 9/20/20 12:28 PM
Author  : Zhao Mingxin
Email   : zhaomingxin17@semi.ac.cn
File    : demo.py
Description:
"""


import cv2
import numpy as np
from zoo.siamfc.siamfc import SiamFCTracker


pause = False


def capture(d_q):
    global pause
    url = "rtsp://admin:zgq123456@172.16.36.23//Streaming/Channels/2"

    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    d_q.put(cv2.resize(frame, (640, 480)))
    while ret:
        if not pause:
            ret, frame = cap.read()
            d_q.put(cv2.resize(frame, (640, 480)))


def track(d_q):
    global pause

    import torch

    from utils.layer import search_replace_convolution2d, QConv2d

    tracker = SiamFCTracker('./siamfc_lite.pth', 0)
    search_replace_convolution2d(tracker.model.features, bit_width=8)
    tracker.model.features.load_state_dict(torch.load('./result/pth/true_quantized.pth'))

    for m in tracker.model.features.modules():
        if isinstance(m, QConv2d):
            m.use_quantization_simulation()

    counter = 0
    name = 'SiamFC Tracker'
    box = (0, 0, 0, 0)
    key = ord('u')
    while True:
        if not d_q.empty():
            frame = d_q.get()
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if counter == 0:
                if key == ord('s'):
                    pause = True
                    box = cv2.selectROI(name, frame)
                    if np.asarray(box).sum() != 0:
                        x1, y1, w, h = map(int, box)
                        x1, y1 = x1 + 1, y1 + 1
                        tracker.init(rgb_img,
                                     (x1, y1, w, h))
                        box = (x1 - 1, y1 - 1, x1 - 1 + w, y1 - 1 + h)
                        counter += 1
                    pause = False
                else:
                    box = (0, 0, 0, 0)
            elif counter > 0:
                box = tracker.update(rgb_img)
                box = (box[0] - 1, box[1] - 1, box[2] - 1, box[3] - 1)
                counter += 1

            if np.asarray(box).sum() != 0:
                frame = frame.copy()
                frame = cv2.rectangle(frame,
                                      (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])),
                                      color=(255, 255, 255), thickness=2)
            cv2.imshow(name, frame)
            key = cv2.waitKey(1)
        else:
            continue


if __name__ == '__main__':
    import queue
    import threading
    vid_queue = queue.Queue()

    cap_thread = threading.Thread(target=capture, args=(vid_queue, ))
    pro_thread = threading.Thread(target=track, args=(vid_queue,))

    cap_thread.start()
    pro_thread.start()

