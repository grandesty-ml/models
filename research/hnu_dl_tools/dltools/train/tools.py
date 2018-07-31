#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/19 13:13
@desc: 
"""
import os

import numpy as np
import tensorflow as tf

from dltools.utils.basic import is_rectangle_overlap
from dltools.utils.math import get_overlap_area
from dltools.utils.functions import sort_all_list


def load_model(checkpoint):
    """
    load the frozen graph of tensorflow as a detection model

    Parameters
    ----------
    checkpoint

    Returns
    -------

    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def run_detection(sess, detection_graph, image_np):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={
            image_tensor: image_np
        })
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)
    return boxes, classes, scores


def merge_bounding_boxes(boxes, scores, classes, threshold=0.5):
    """
    将重合度较大的 bounding box 合并

    Parameters
    ----------
    boxes
    scores
    classes
    threshold

    Returns
    -------

    """
    [boxes, classes], scores = sort_all_list(boxes, classes,
                                             key_list=scores, reverse=True)
    for idx1 in range(len(scores) - 1):
        if 0 == scores[idx1]:
            continue
        box1 = boxes[idx1]
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        for idx2 in range(idx1 + 1, len(scores)):
            if 0 == scores[idx2] or classes[idx1] != classes[idx2]:
                continue
            box2 = boxes[idx2]
            if is_rectangle_overlap(box1, box2):
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                if 0.001 > area2:
                    scores[idx2] = 0
                    continue
                area = get_overlap_area(box1, box2)
                if (area / (area1 + area2 - area)) > threshold:
                    scores[idx2] = 0
    boxes = boxes[scores != 0]
    classes = classes[scores != 0]
    scores = scores[scores != 0]
    return boxes, scores, classes

