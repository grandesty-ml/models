#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/1 10:07
@desc: 
"""

import tensorflow as tf

__all__ = ['BaseDataGenerator']


class BaseDataGenerator(object):
    """
    用于生成 tensorflow 支持的训练数据的基本类
    """

    def __init__(self, data_files,
                 keys_to_features=None,
                 shuffle=False,
                 batch_size=32,
                 num_epochs=1):
        # 初始化 tf dataset
        if len(data_files) > 1:
            self._dataset = tf.data.Dataset.from_tensor_slices(data_files)
            if shuffle:
                self._dataset = self._dataset.shuffle(
                    buffer_size=len(data_files))
            self._dataset = self._dataset.flat_map(tf.data.TFRecordDataset)
        else:
            self._dataset = tf.data.TFRecordDataset(data_files)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_epochs = num_epochs

        self._keys_to_features = keys_to_features

        self._fetch_data()

    def _fetch_data(self):
        self._dataset = self._dataset.map(self._record_parser,
                                          num_parallel_calls=5)
        self._dataset = self._dataset.prefetch(self._batch_size)

        if self._shuffle:
            # When choosing shuffle buffer sizes, larger sizes result in better
            # randomness, while smaller sizes have better performance.
            self._dataset = self._dataset.shuffle(buffer_size=4096)

            # We call repeat after shuffling, rather than before, to prevent
            # separate epochs from blending together.
            self._dataset = self._dataset.repeat(self._num_epochs)
        self._dataset = self._dataset.batch(self._batch_size)

    def _record_parser(self, value):
        """
        在这里解析 tf record

        Parameters
        ----------
        value

        Returns
        -------

        """
        raise NotImplementedError

    def __call__(self):
        iterator = self._dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
