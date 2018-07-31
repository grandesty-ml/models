#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/27 22:25
@desc: 
"""
from dltools.utils.functions import *


def test_sort_all_list():
    a = [1, 4, 2, 7, 3]
    b = [2, 35, 61, 24, 63]
    [b], a = sort_all_list(b, key_list=a)
    assert np.allclose(np.asarray(a), np.array([1, 2, 3, 4, 7]))
    assert np.allclose(np.asarray(b), np.array([2, 61, 63, 35, 24]))


def test_merge_dict():
    a = {'1': 1, '2': 1}
    b = {'3': 1, '2': 1}
    c = merge_dict(a, b)
    assert 1 == c['1']
    assert 2 == c['2']
    assert 1 == c['3']
