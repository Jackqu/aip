import os
import cv2
import numpy as np
import json
import yaml
from easydict import EasyDict as edict
from collections import deque
from aip import AipBodyAnalysis

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()
def get_client():
    APP_ID = '17067353'
    API_KEY = 'ovdLIpCXNFAHHYXCASwVrO4u'
    SECRET_KEY = 'eyCudNEFKkQtH9KPfvyLr16KMendt1ir'
    client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)
    return client



class PoseParser(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = get_client()
        self.smooth_queue = deque(maxlen = self.cfg.smooth_frame_num)
    def parse(self):
        ######## read video
        video = self.cfg.video
        cap = cv2.VideoCapture(video)
        _, img = cap.read()
        print(img.shape)
        if self.cfg.flip:
            img = cv2.flip(img.transpose((1, 0, 2)), 1)
        img = cv2.resize(img, (0,0),fx = self.cfg.resize_ratio, fy = self.cfg.resize_ratio)
        video_new = cv2.VideoWriter(self.cfg.video_save,
                                    cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25, (img.shape[1], img.shape[0]))
        while True:
            ret, img = cap.read()
            if ret == False:
                break
            if self.cfg.flip:
                img = cv2.flip(img.transpose((1, 0, 2)), 1)
            img = cv2.resize(img, (0, 0), fx=self.cfg.resize_ratio, fy=self.cfg.resize_ratio)

            kpt_result = self.parse_img(img)
            img = self.vis(img, kpt_result)
            video_new.write(img)
            cv2.imshow('img', img)
            cv2.waitKey(10)
        video_new.release()


    def parse_img(self, img):
        cv2.imwrite('temp.png', img)
        img = get_file_content('temp.png')
        kpt_result = self.client.bodyAnalysis(img)
        return kpt_result
    def vis(self, img, kpt_result):
        kpts, area = self.get_kpts(kpt_result)
        kpts = self.smooth(kpts, area)

        for kpt_index in range(len(kpts)):
            y = kpts[kpt_index][0]
            x = kpts[kpt_index][1]
            is_visible = kpts[kpt_index][2]
            if is_visible > self.cfg.vis_th:
                cv2.circle(img, (int(x), int(y)), 4, self.cfg.kpts_color, thickness=1)

        for kpt_pair in self.cfg.keypoints_pair:
            if kpts[kpt_pair[0]][2] > self.cfg.vis_th and kpts[kpt_pair[1]][2] > self.cfg.vis_th:
                p1 = (int(kpts[kpt_pair[0]][1]), int(kpts[kpt_pair[0]][0]))
                p2 = (int(kpts[kpt_pair[1]][1]), int(kpts[kpt_pair[1]][0]))
                cv2.line(img, tuple(p1), tuple(p2), self.cfg.line_color, 2)

        return img

    def get_kpt_index(self):
        ########the kpt mapping between baidu and our defination
        kpt_index = dict()
        kpt_index['top_head'] = 0
        kpt_index['neck'] = 1
        kpt_index['right_shoulder'] = 2
        kpt_index['left_shoulder'] = 3
        kpt_index['right_elbow'] = 4
        kpt_index['left_elbow'] = 5
        kpt_index['right_wrist'] = 6
        kpt_index['left_wrist'] = 7
        kpt_index['right_hip'] = 8
        kpt_index['left_hip'] = 9
        kpt_index['right_knee'] = 10
        kpt_index['left_knee'] = 11
        kpt_index['right_ankle'] = 12
        kpt_index['left_ankle'] = 13
        return kpt_index
    def get_kpts(self, data):
        ######## get kpt mapping
        kpt_index = self.get_kpt_index()

        ######## for now, only process the first person
        person_num = data['person_num']
        kpts = []
        person_index = 0
        for kpt_name, _ in kpt_index.items():
            #print(kpt_name)
            kpt = []
            kpt.append(data['person_info'][person_index]['body_parts'][kpt_name]['y'])  # height
            kpt.append(data['person_info'][person_index]['body_parts'][kpt_name]['x'])  # width
            kpt.append(data['person_info'][person_index]['body_parts'][kpt_name]['score'])  # score
            kpts.append(kpt)
        rect = [data['person_info'][person_index]['location']['height'], data['person_info'][person_index]['location']['width']]
        area = rect[0] * rect[1]
        return kpts, area

    def smooth(self, kpts, area):
        ########en-queue
        self.smooth_queue.append(kpts)
         ######## handle each joint
        for kpt_index in range(len(kpts)):
             ######## cal smooth weight
            weight = list()
            target_kpt = np.array(kpts[kpt_index][:2])
            for i in range(len(self.smooth_queue)):
                current_kpt = np.array(self.smooth_queue[i][kpt_index][:2])
                norm = np.linalg.norm(target_kpt - current_kpt)
                norm = norm/area ######## further normalized by rect area
                weight.append(np.exp(-norm*norm/2/self.cfg.smooth_sigma))
            weight = np.array(weight)
            weight = weight / np.sum(weight)
            target_kpt = np.zeros(3)
            print(weight)
            for i in range(len(self.smooth_queue)):
                current_kpt = np.array(self.smooth_queue[i][kpt_index])
                target_kpt = target_kpt + weight[i] * current_kpt
            kpts[kpt_index][0] = target_kpt[0]
            kpts[kpt_index][1] = target_kpt[1]
            kpts[kpt_index][2] = target_kpt[2]

        return kpts


def main():
    cfg_file = './config/body_kpt_video.yaml'
    cfg = yaml.load(open(cfg_file))
    cfg = edict(cfg)
    print(cfg)
    pose_parser = PoseParser(cfg)
    pose_parser.parse()


if __name__ == '__main__':
    main()