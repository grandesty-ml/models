#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/4/4 16:37
@desc: 
"""
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
from dltools.utils.log import get_console_logger


class BaseCropper(object):
    """
    用于切割图像的基类
    """
    __metaclass__ = ABCMeta

    def __init__(self, image, logger=None):
        """
        Parameters
        ----------
        image: 图像或图像路径
        logger: 日志对象
        """
        # 配置切割参数
        self._buf_data = {}
        # 定义日志
        if logger is not None:
            self._logger = logger
        else:
            self._logger = get_console_logger('BaseCropper', 1)
        # 读入图像
        self._image = None
        # 判断是否满足条件
        self._break = True
        self._check_params(image)
        if not self._break:
            self._set_up()

    def _check_params(self, image):
        """
        判断是否满足条件

        Parameters
        ----------
        image
        Returns
        -------

        """
        # 检测图像
        if isinstance(image, str):
            self._image = cv2.imread(image)
            if self.image is None:
                self._logger.warn('There is not a image: {}'.format(image))
                return
        elif isinstance(image, np.ndarray):
            image = np.squeeze(image)
            if len(image.shape) == 3 and image.shape[0] <= 4:
                image = np.transpose(image, [1, 2, 0])
            self._image = image.copy()
        else:
            self._logger.warn('图像对象错误！')
            return

        if self._preprocess():
            self._logger.warn('Not Meet The Condition.')
            return

        self._break = False

    @abstractmethod
    def _set_up(self):
        """
        基本配置

        Returns
        -------

        """
        raise NotImplementedError('this method is not implemented !')

    @abstractmethod
    def _crop_image(self):
        """
        切割子图

        Returns
        -------

        """
        raise NotImplementedError('this method is not implemented !')

    @abstractmethod
    def _write_image(self):
        """
        写入子图到文件

        Returns
        -------

        """
        raise NotImplementedError('this method is not implemented !')

    @abstractmethod
    def _preprocess(self):
        """
        预判断

        Returns
        -------

        """
        return False

    @abstractmethod
    def _update(self):
        """
        执行切割

        Returns
        -------

        """
        raise NotImplementedError('this method is not implemented !')

    @property
    def image(self):
        return self._image

    def update(self):
        if self._break:
            return False
        else:
            self._logger.info('cropping ...')
            self._update()

        return True
