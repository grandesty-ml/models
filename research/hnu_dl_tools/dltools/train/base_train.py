#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/1 11:15
@desc: 
"""
import os

import tensorflow as tf


class BaseTrain(object):
    """
    基本训练类
    """

    def __init__(self, model_fn,
                 model_save_dir,
                 train_input,
                 params=None,
                 keep_checkpoint_max=10,
                 evaluate_input=None):
        # Set up a RunConfig to only save checkpoints once per training cycle.
        run_config = tf.estimator.RunConfig(
            keep_checkpoint_max=keep_checkpoint_max)
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=model_save_dir,
            config=run_config, params=params)
        self.model_save_dir = model_save_dir
        self.train_input = train_input
        self.evaluate_input = evaluate_input if evaluate_input else train_input

    def train(self, log_fmt, log_iter=10, save_steps=100, hooks=None):
        """
        训练

        Returns
        -------

        """
        logging_hook = tf.train.LoggingTensorHook(
            tensors=log_fmt, every_n_iter=log_iter)
        saving_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=self.model_save_dir, save_steps=save_steps)

        all_hooks = [logging_hook, saving_hook]
        if hooks is not None:
            all_hooks += hooks

        self.estimator.train(input_fn=self.train_input, hooks=all_hooks)

    def evaluate(self, hooks=None, checkpoint_path=None):
        """
        测试

        Returns
        -------

        """
        if checkpoint_path is not None:
            assert os.path.exists(checkpoint_path), 'Not such Directory found !'
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

        return self.estimator.evaluate(input_fn=self.evaluate_input,
                                       checkpoint_path=checkpoint_path,
                                       hooks=hooks)
