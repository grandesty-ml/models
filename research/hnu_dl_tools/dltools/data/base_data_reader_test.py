#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/4/3 16:20
@desc: 
"""
from .base_data_reader import BaseFileReader


def test_base_data_reader():
    reader = BaseFileReader('./data', display=2, max_number=10)
    files = [p for p in reader]
    print(files)
    assert len(files) == 10
