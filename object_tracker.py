from absl import flags
import sys

FLAGS = flags.FLAGS
FLAGS(sys.argv)

from bidict import bidict
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import random
import string
import math

class_names = [c.strip() for c in open('./data/label/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)
vid_width, vid_height = 368, 640
from _collections import deque

pts = [deque(maxlen=30) for _ in range(10000)]

# imgfolder = '/Users/guoao/Desktop/comp9517/group_proj/step_images/train/STEP-ICCV21-02'
imgfolder = '/Users/guoao/Desktop/comp9517/group_proj/step_images/test/STEP-ICCV21-01'
outfolder = '/Users/guoao/Desktop/comp9517/group_proj/step_images/panoptic_maps/train/0002_vis'
imglist = os.listdir(imgfolder)
imglist.sort()


# defining function for random
# string id with parameter
def ran_gen(size, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))


def get_track_id(value, dict):
    result = 0
    for k, v in dict.items():
        if value in v:
            result = k
    return result


def get_group_id(value, dict):
    result = 0
    for k, v in dict.items():
        if value in list(v.keys()):
            result = k
    return result


start_point, end_point = None, None
state = 0


def mouse_click(event, x, y, flags, data):
    global state, start_point, end_point

    # first click the mouse, set the start point
    if event == cv2.EVENT_LBUTTONDOWN:
        if state == 0:
            start_point = (x, y)
            state += 1
    # Second click the mouse, set the end point
    elif event == cv2.EVENT_LBUTTONUP and state == 1:
        end_point = (x, y)
        state += 1
    # If right click, delete the area
    if event == cv2.EVENT_RBUTTONUP:
        start_point, end_point = None, None
        state = 0


# {track_id: [(x_mid, y_mid)，(x_min, y_min), (x_max, y_max)], ...}
# group_dict = {id1: {track_id1:err_num, track_id2:err_num, ...}, id2: ...}
group_dict = {}

# fixed distance
group_distance = 18
# distance ratio
group_distance_ratio = 0.5
# error number
max_occurrence = 3
history_points = {}
loop_index = 0
counter = []

while True:
    cv2.namedWindow('output')
    cv2.setMouseCallback('output', mouse_click)

    imgname = imglist[loop_index]
    loop_index += 1
    if not imgname.endswith('.jpg'):
        continue

    imgpath = os.path.join(imgfolder, imgname)
    img = cv2.imread(imgpath)

    img = cv2.resize(img, (vid_height, vid_width))
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)

    classes = classes[0]

    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)

    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    current_count = int(0)
    current_frame_count = int(0)

    track_dict = {}
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        if class_name != 'person':
            continue

        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 15)), (int(bbox[0]) + (len(class_name)
                                                                               + len(str(track.track_id))) * 7,
                                                               int(bbox[1])), color, -1)
        aa3 = track.track_id
        cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 5)), 0, 0.3,
                    (255, 255, 255), 1)

        center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
        pts[track.track_id].append(center)

        if track.track_id not in track_dict:
            track_dict[track.track_id] = []

        track_dict[track.track_id].append(center)  # x_mid, y_mid
        track_dict[track.track_id].append((int(bbox[0]), int(bbox[1])))  # x_min, y_min
        track_dict[track.track_id].append((int(bbox[2]), int(bbox[3])))  # x_max, y_max

        cc = track.track_id
        bb = pts
        aa = pts[track.track_id]
        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                continue
            # thickness = int(np.sqrt(64/float(j+1))*2)
            thickness = 2
            cv2.line(img, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color, thickness)

        height, width, _ = img.shape
        center_y = int(((bbox[1]) + (bbox[3])) / 2)
        center_x = int(((bbox[0]) + (bbox[2])) / 2)

        if height >= center_y >= 0 and width >= center_x >= 0:
            current_frame_count += 1
            counter.append(int(track.track_id))

        if state > 1:
            print('Check')
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
            state = -1

        elif state == -1:
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
            if int(end_point[1]) >= center_y >= int(start_point[1]) and \
                    int(end_point[0]) >= center_x >= int(start_point[0]):
                current_count += 1

    # group_dict = {id1: {track_id1:err_num, track_id2:err_num, ...}, id2: ...}
    # check group dict, according to track dict

    try:
        if track_dict[12]:
            aa3 = True
    except:
        aa3 = False
    aa4 = aa3
    if len(group_dict) > 0:
        for i in list(group_dict.keys()):
            all_dict = group_dict[i]
            if len(all_dict) < 2:
                del group_dict[i]
                continue
            # track_list = [m[0] for m in all_dict]
            track_list = list(all_dict.keys())
            length = len(track_list)
            error_points = []
            index = 0
            while True:
                point = track_list[index]
                check = length - index - 1
                try:
                    point_x = track_dict[point][0][0]
                    point_y = track_dict[point][0][1]
                    point_distance_x = int(abs(track_dict[point][2][0] - track_dict[point][1][0]))
                    point_distance_y = int(abs(track_dict[point][2][1] - track_dict[point][1][1]))
                    point_height = int(abs(track_dict[point][2][1] - track_dict[point][1][1]))
                    point_width = int(abs(track_dict[point][2][0] - track_dict[point][1][0]))
                    for p in range(index + 1, length):

                        test_point = track_list[p]
                        try:
                            test_point_x = track_dict[test_point][0][0]
                            test_point_y = track_dict[test_point][0][1]
                            test_point_distance_x = int(
                                abs(track_dict[test_point][2][0] - track_dict[test_point][1][0]))
                            test_point_distance_y = int(
                                abs(track_dict[test_point][2][1] - track_dict[test_point][1][1]))
                            test_point_height = int(abs(track_dict[test_point][2][1] - track_dict[test_point][1][1]))
                            test_point_width = int(abs(track_dict[test_point][2][0] - track_dict[test_point][1][0]))

                            euclidean_distance = int(
                                math.sqrt((test_point_x - point_x) ** 2 + (test_point_y - point_y) ** 2))
                            height_list = [point_height, test_point_height]
                            dis = int((max(height_list) / min(height_list)) * euclidean_distance)
                            dis_threshold = group_distance + group_distance_ratio * (
                                        point_distance_x + test_point_distance_x)
                            if dis < dis_threshold:
                                continue
                            else:
                                check -= 1
                        except:
                            check = 0
                except Exception as es:
                    check = 0

                if check == 0:  # no point cut error num
                    all_dict[point] += 1
                else:  # exist point recover error num
                    if all_dict[point] > 0:
                        all_dict[point] -= 1

                if all_dict[point] >= max_occurrence:
                    error_points.append(point)

                index += 1
                if index == length - 1:
                    break
            # delete error points
            for j in error_points:
                del all_dict[j]
            # update group_dict
            if len(all_dict) < 2:
                del group_dict[i]
            else:
                group_dict[i] = all_dict

    # check track dict
    if len(track_dict) > 0:
        all_track_point_list = list(track_dict.values())
        center_track_point_list = [m[0] for m in all_track_point_list]
        min_track_point_list = [m[1] for m in all_track_point_list]
        max_track_point_list = [m[2] for m in all_track_point_list]
        length = len(center_track_point_list)
        add_track_points = []
        track_index = 0
        while True:
            track_point = center_track_point_list[track_index]
            track_point_x = track_point[0]
            track_point_y = track_point[1]
            point_distance_x = int(abs(min_track_point_list[track_index][0] - max_track_point_list[track_index][0]))
            point_distance_y = int(abs(min_track_point_list[track_index][1] - max_track_point_list[track_index][1]))
            point_height = int(abs(min_track_point_list[track_index][1] - max_track_point_list[track_index][1]))
            point_width = int(abs(min_track_point_list[track_index][0] - max_track_point_list[track_index][0]))
            track_check = len(center_track_point_list) - track_index - 1
            for p in range(track_index + 1, len(center_track_point_list)):
                test_track_point = center_track_point_list[p]
                test_track_point_x = test_track_point[0]
                test_track_point_y = test_track_point[1]
                test_point_distance_x = int(abs(min_track_point_list[p][0] - max_track_point_list[p][0]))
                test_point_distance_y = int(abs(min_track_point_list[p][1] - max_track_point_list[p][1]))
                test_point_height = int(abs(min_track_point_list[p][1] - max_track_point_list[p][1]))
                test_point_width = int(abs(min_track_point_list[p][0] - max_track_point_list[p][0]))
                euclidean_distance = int(
                    math.sqrt((test_track_point_x - track_point_x) ** 2 + (test_track_point_y - track_point_y) ** 2))

                height_list = [point_height, test_point_height]
                dis = int((max(height_list) / min(height_list)) * euclidean_distance)
                dis_threshold = group_distance + group_distance_ratio * (point_distance_x + test_point_distance_x)
                if dis < dis_threshold:
                    print(dis, dis_threshold)
                    track_point_track_id = get_track_id(track_point, track_dict)
                    test_track_point_track_id = get_track_id(test_track_point, track_dict)

                    track_point_group_id = get_group_id(track_point_track_id, group_dict)
                    test_track_point_group_id = get_group_id(test_track_point_track_id, group_dict)

                    if track_point_group_id != 0 and test_track_point_group_id == 0:
                        group_dict[track_point_group_id][test_track_point_track_id] = 0
                    elif track_point_group_id == 0 and test_track_point_group_id != 0:
                        group_dict[test_track_point_group_id][track_point_track_id] = 0
                    elif track_point_group_id == 0 and test_track_point_group_id == 0:
                        add_track_check = False
                        for i in add_track_points:
                            if track_point_track_id in i and test_track_point_track_id not in i:
                                i.append(test_track_point_track_id)
                                add_track_check = True
                                break
                            elif test_track_point_track_id in i and track_point_track_id not in i:
                                i.append(track_point_track_id)
                                add_track_check = True
                                break
                        if add_track_check == False:
                            add_track_points.append([track_point_track_id, test_track_point_track_id])
            track_index += 1
            if track_index == length - 1:
                break

        group_point_list = [list(m.keys()) for m in group_dict.values()]

        # add new points to group
        for item in range(len(add_track_points)):
            duplicate_check = False
            if len(group_point_list) > 0:
                for i in group_point_list:
                    if set(add_track_points[item]).issubset(set(i)):
                        duplicate_check = True
                        break
            if duplicate_check == True:
                continue
            group_key = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(4)])
            group_dict[group_key] = {}
            for i in add_track_points[item]:
                group_dict[group_key][i] = 0

    num_of_people_in_groups = 0
    for i in group_dict.values():
        for j in i.keys():
            if j in list(track_dict.keys()):
                num_of_people_in_groups += 1
    total_count = len(set(counter))
    cv2.putText(img, "Total Pedestrian Count: " + str(total_count), (0, 15), 0, 0.5, (0, 0, 255), 2)
    cv2.putText(img, "Current Rectangle Pedestrian Count: " + str(current_count), (0, 30), 0, 0.5, (0, 0, 255), 2)
    cv2.putText(img, "Current Frame Pedestrian Count: " + str(current_frame_count), (0, 45), 0, 0.5, (0, 0, 255), 2)
    cv2.putText(img, "pedestrians walk in group: " + str(num_of_people_in_groups), (0, 60), 0, 0.5, (0, 0, 255), 2)
    cv2.putText(img, "pedestrians walk alone: " + str(current_frame_count - num_of_people_in_groups), (0, 75), 0, 0.5,
                (0, 0, 255), 2)

    # draw rectangle according to group dict
    for i in list(group_dict.keys()):
        point_list = list(group_dict[i].keys())
        try:
            min_coordinate_list = [track_dict[m][1] for m in point_list]
            max_coordinate_list = [track_dict[n][2] for n in point_list]
            min_x_list = [m[0] for m in min_coordinate_list]
            min_y_list = [m[1] for m in min_coordinate_list]
            max_x_list = [m[0] for m in max_coordinate_list]
            max_y_list = [m[1] for m in max_coordinate_list]
            x_min = min(min_x_list)
            y_min = min(min_y_list)
            x_max = max(max_x_list)
            y_max = max(max_y_list)
            history_points[i] = [x_min, y_min, x_max, y_max]
            cv2.rectangle(img, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 0, 0), 2)
        except Exception as es:
            x_min = history_points[i][0]
            y_min = history_points[i][1]
            x_max = history_points[i][2]
            y_max = history_points[i][3]
            cv2.rectangle(img, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 0, 0), 2)
            print(es)

    cv2.resizeWindow('output', 1024, 768)
    cv2.imshow('output', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        start_point, end_point = None, None
        state = 0
    if key == ord('q'):
        break

cv2.destroyAllWindows()
