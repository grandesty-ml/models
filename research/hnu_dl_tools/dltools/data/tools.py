#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/19 14:11
@desc: 
"""

from object_detection.core import standard_fields as fields


class NewTfExampleFields(fields.TfExampleFields):
    all_x_axis = 'image/objects/all_x'
    all_y_axis = 'image/objects/all_y'
    per_object_numpy = 'image/objects/per_number'
