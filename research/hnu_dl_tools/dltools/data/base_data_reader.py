#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/2/28 17:17
@desc: 基本数据
"""
from pathlib import Path
from dltools.utils import log

__all__ = ['BaseFileReader']


class BaseFileReader(object):
    """
    基本原始图像数据读取
    """

    def __init__(self, root,
                 recurrence=True,
                 max_number=1e12,
                 display=100,
                 logger=None):
        """

        Parameters
        ----------
        root: 根目录
        recurrence: 是否递归搜索子目录
        max_number: 读入的最大样本数
        display: 显示间隔
        logger: 日志对象
        """
        if not Path(root).exists():
            raise NotADirectoryError

        # 使用栈保存目录
        self._directory_stack = [Path(root)]

        # 保存中间数据
        self._buf_data = None
        self._recurrence = recurrence

        self._max_number = max_number
        self._count = 0
        self._display = display

        if logger is not None:
            self._logger = logger
        else:
            self._logger = log.get_console_logger('FileReader')

    def __iter__(self):
        while 0 < len(self._directory_stack):
            path = self._directory_stack.pop()
            for file in path.iterdir():
                self._has_next()
                if file.is_file():
                    if self._filter(file):
                        yield self._buf_data
                elif self._recurrence and file.is_dir():
                    self._directory_stack.append(file)
                else:
                    raise TypeError('不能确定的文件类型！')

    def _has_next(self):
        """
        判断是否已经超过最大读取数量

        Returns
        -------

        """
        if self._count % self._display == 0:
            self._logger.info(
                '正在读取第 {} 个文件对象 ...'.format(self._count))
        if self._count <= self._max_number:
            self._count += 1
        else:
            raise StopIteration

    def _filter(self, file):
        """
        自定义用于筛选文件的方法, 可以重载

        Returns
        -------

        """
        # 统一使用这样的方法表示数据
        self._buf_data = {'feature': str(file),  # feature 表示数据的特征
                          'label': None,  # label 表示数据的标记信息
                          'data': None,  # data 表示其他辅助信息
                          'name': str(file)}  # name 表示数据的表示
        return True
