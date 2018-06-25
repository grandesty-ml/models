#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/2/28 19:58
@desc: 用于生成 tensorflow 支持的二进制文件
"""
from pathlib import Path
from typing import Iterable

import tensorflow as tf

from dltools.utils import log

__all__ = ['BaseRecordGenerator']


class BaseRecordGenerator(object):
    """
    用于生成 tensorflow 支持的二进制文件的基本类
    """

    def __init__(self, data, output,
                 display=100,
                 logger=None):
        """

        Parameters
        ----------
        data: 可迭代的数据对象
        output: 保存 record 的文件名
        display: 显示间隔
        logger: 日志对象
        """
        assert isinstance(data, Iterable), '请输入正确的数据！data必须是可迭代的对象。'
        self._data = data

        path = Path(Path(output).parent)
        if not path.exists():
            path.mkdir(parents=True)
        self._output = output

        self._buf_data = {}
        self._display = display

        if logger is not None:
            self._logger = logger
        else:
            self._logger = log.get_console_logger('RecordGenerator')

    def _encode_data(self):
        """
        将数据转化为 tf example

        Returns
        -------

        """
        raise NotImplementedError

    def _write_data(self, writer):
        """
        将数据写入文件

        Parameters
        ----------
        writer

        Returns
        -------

        """
        raise NotImplementedError

    def update(self):
        """
        处理全部数据

        Returns
        -------

        """
        with tf.python_io.TFRecordWriter(self._output) as writer:
            for idx, meta in enumerate(self._data):
                if idx % self._display == 0:
                    self._logger.info(
                        'Processing the number of {} data.'.format(idx))
                self._buf_data['raw'] = meta
                self._encode_data()
                self._write_data(writer)
                self._buf_data.clear()
