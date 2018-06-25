#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/19 14:06
@desc: 
"""
import numpy as np

from dltools.utils.basic import check_consistent_length


def merge_dict(*dicts):
    """
    合并一系列字典，合并后的字典的key是所有字典的key的并集，
    其value取和

    Parameters
    ----------
    dicts: 字典列表

    Returns
    -------
    合并后的字典
    """
    res = {}
    for _dict in dicts:
        for key, value in _dict.items():
            if key not in res:
                res[key] = value
            else:
                res[key] += value
    return res


def sort_all_list(*lists,
                  key_list=None,
                  reverse=False):
    """
    同时排序多个列表，并转换为数组

    Parameters
    ----------
    lists
    key_list
    reverse

    Returns
    -------

    """
    if key_list is None:
        raise ValueError('Key is None! Expect a comparable element.')
    check_consistent_length(*lists, key_list)
    all_lists = map(lambda x: np.asarray(x), lists)
    key_list = np.asarray(key_list)
    if reverse:
        idx = np.argsort(-key_list)
    else:
        idx = np.argsort(key_list)
    all_lists = [*map(lambda x: x[idx], all_lists)]
    return all_lists, key_list[idx]

