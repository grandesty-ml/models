#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/2/28 20:28
@desc: 
"""
import logging

FORMATTER = ['%(name)s %(levelname)s %(asctime)s: %(message)s',
             '%(levelname)s:%(name)s:%(asctime)s:%(message)s',
             '%(levelname)s:%(name)s:%(message)s',
             '%(levelname)s:%(module)s:%(asctime)s:%(message)s']


def get_console_logger(name, formatter=2):
    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(FORMATTER[formatter]))
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    return logger


def get_file_logger(name, file_name, formatter=2):
    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(logging.Formatter(FORMATTER[formatter]))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(FORMATTER[formatter]))
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
