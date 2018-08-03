#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/19 13:04
@desc: 图像格式转换的方法
"""
import base64

import cv2
import numpy as np


def image_encode_base64(image):
    """
    图像转换为base64编码,然后转换为字符串

    Parameters
    ----------
    image

    Returns
    -------

    """
    code = cv2.imencode('.jpg', image)[1]
    code = base64.b64encode(code)
    code = base64.encodebytes(code)
    code = bytes.decode(code)
    return code


def image_decode_base64(string):
    """
    将字符串转化为图像

    Parameters
    ----------
    string

    Returns
    -------

    """
    byte = str.encode(string)
    byte = base64.decodebytes(byte)
    byte = base64.b64decode(byte)
    byte = np.frombuffer(byte, np.uint8)
    img = cv2.imdecode(byte, cv2.CAP_MODE_RGB)
    return img
