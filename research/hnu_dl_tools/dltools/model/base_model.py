#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/2/28 16:28
@desc: 这是一个基本模型模板类
"""

import tensorflow as tf

__all__ = ['BaseModel']


class BaseModel(object):
    """
    基本模型模板
    """

    def __init__(self, data_format='channels_last'):
        assert data_format in ('channels_last', 'channels_first')
        self._data_format = data_format
        self._feature = None
        self._label = None
        self._mode = None
        self._params = None
        self._is_training = None
        self._buffer_data = {}

    def _preprocess(self):
        """
        对输入数据进行预处理

        Returns
        -------

        """
        raise NotImplementedError

    def _network_model(self):
        """
        重载这个方法，定义网络模型

        Returns
        -------

        """
        raise NotImplementedError

    def __call__(self, features, labels, mode, params):
        """
        调用模型

        Parameters
        ----------
        features: 字典对象，包含训练需要的特征数据
        labels: 字典对象，包含训练需要的标签数据
        mode: 运行模式
        params: 字典对象，包含训练、预测和测试需要的参数

        Returns
        -------

        """
        self._feature = features
        self._label = labels
        self._mode = mode
        self._params = params
        self._is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        self._preprocess()
        self._network_model()
        self._postprocess()
        if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
            loss = self._loss()
            metrics = self._metrics()
        else:
            loss = None
            metrics = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = self._train_op(loss)
        else:
            train_op = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = self._predict()
        else:
            predictions = None
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    def _train_op(self, loss):
        """
        定义训练评估
        Returns
        -------

        """
        raise NotImplementedError

    def _metrics(self):
        """
        定义测试评估

        Returns
        -------

        """
        raise NotImplementedError

    def _predict(self):
        """
        定义预测评估
        Returns
        -------

        """
        raise NotImplementedError

    def _loss(self):
        """
        定义损失函数
        Returns
        -------

        """
        raise NotImplementedError

    def _postprocess(self):
        """
        对网络输出进行处理
        Returns
        -------

        """
        pass
