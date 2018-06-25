#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/28 19:14
@desc: 
"""
import itertools

from dltools.utils.basic import *


def test_num_samples():
    a = (1, 2, 3, 4, 5)
    b = [1, 2, 3, 4, 5, 6]
    c = np.array([1, 2, 3, 4, 5])
    d = np.arange(256).reshape((16, 16))

    assert 5 == num_samples(a)
    assert 6 == num_samples(b)
    assert 5 == num_samples(c)
    assert 16 == num_samples(d)


def test_check_consistent_length():
    a = (1, 2, 3, 4, 5)
    b = [1, 2, 3, 4, 5]
    c = np.array([1, 2, 3, 4, 5])
    lists = [a, b, c]
    for i in range(len(lists)):
        for items in itertools.combinations(lists, i + 1):
            assert check_consistent_length(*items) is None


def test_get_name_and_ext():
    path1 = '/sd/sds/sds.sds'
    path2 = 'ad/sada/sad.sad'
    path3 = '/sds/sd.sd.sd'

    name, ext = get_name_and_ext(path1)
    assert name == 'sds' and ext == '.sds'

    name, ext = get_name_and_ext(path2)
    assert name == 'sad' and ext == '.sad'

    name, ext = get_name_and_ext(path3)
    assert name == 'sd.sd' and ext == '.sd'


def test_is_rectangle_overlap():
    rect1 = [0.4, 0.4, 0.6, 0.6]
    rect2 = [0.5, 0.5, 0.7, 0.7]
    rect3 = [0.2, 0.2, 0.3, 0.3]

    assert is_rectangle_overlap(rect1, rect2)
    assert not is_rectangle_overlap(rect2, rect3)


def test_is_in_rectangle():
    rect = [0.4, 0.4, 0.6, 0.6]
    point1 = [0.42, 0.45]
    point2 = [0.32, 0.5]

    assert is_in_rectangle(point1, rect)
    assert not is_in_rectangle(point2, rect)


def test_is_in_polygon():
    poly = [[0.2, 0.2],
            [0.1, 0.5],
            [0.3, 0.8],
            [0.8, 0.6],
            [0.7, 0.3]]
    point1 = [0.15, 0.25]
    point2 = [0.45, 0.4]

    assert is_in_polygon(point2, poly)
    assert not is_in_polygon(point1, poly)
