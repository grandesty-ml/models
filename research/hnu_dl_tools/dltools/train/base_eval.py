#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/2 10:08
@desc: 
"""
from collections import Iterable

import tensorflow as tf

from dltools.utils import log


class _Base(object):

    def __init__(self, model_dir,
                 model_name,
                 config=None,
                 logger=None,
                 cpu_only=False):
        self.model_dir = model_dir
        self.model_name = model_name
        self.config = config
        self.device = None
        self._cpu_only = cpu_only
        if logger is not None:
            self.logger = logger
        else:
            self.logger = log.get_console_logger('Eval Task')
        self._device()

    def _device(self):
        if self.config is None:
            if self._cpu_only:
                self.config = tf.ConfigProto(device_count={'CPU': 1})
                self.device = '/cpu:0'
            else:
                self.config = tf.ConfigProto(device_count={'GPU': 1})
                self.device = '/gpu:0'
        else:
            device_count = self.config.device_count
            if device_count is None:
                if self._cpu_only:
                    self.config.device_count = {'CPU': 1}
                    self.device = '/cpu:0'
                else:
                    self.config.device_count = {'GPU': 1}
                    self.device = '/gpu:0'


class BaseExporter(_Base):
    """
    适用于利用 tensorflow SaveModel 导出的模型
    """

    def __init__(self, model_dir,
                 model_name,
                 input_tensor_map,
                 output_tensor_map,
                 config=None,
                 logger=None,
                 cpu_only=False):
        """

        Parameters
        ----------
        model_dir: checkpoint 路径
        model_name: 导出的模型标记名
        input_tensor_map: Dict 对象，模型的输入数据张量名称；
                          key 是计算图中的张量名称，
                          value 是导出后的模型对应输入节点的名称
        output_tensor_map: Dict 对象，模型的输出数据张量名称；
                           key 是计算图中的张量名称，
                           value 是导出后的模型对应输出节点的名称
        config: tf Session 的配置
        logger: 日志对象
        cpu_only
        """
        super(BaseExporter, self).__init__(model_dir, model_name, config,
                                           logger, cpu_only)
        self.input_tensor_map = {'map': input_tensor_map, 'name': 'Input'}
        self.output_tensor_map = {'map': output_tensor_map, 'name': 'Output'}

    def _get_tensor_map(self, graph, src_tensor_map):
        """
        基于导入和导出张量的名字构造张量字典

        Parameters
        ----------
        graph
        src_tensor_map

        Returns
        -------

        """
        dst_tensor_map = {}
        self.logger.info(
            'Generating tensor map from {}'.format(src_tensor_map['name']))
        for import_name, export_name in src_tensor_map['map'].items():
            tensor = graph.get_tensor_by_name(import_name)
            tensor = tf.saved_model.utils.build_tensor_info(tensor)
            dst_tensor_map[export_name] = tensor
            self.logger.info(
                'Importing {} from graph and Exporting as {}'.format(
                    import_name, export_name))
        return dst_tensor_map

    def export(self):
        """
        执行导出操作

        Returns
        -------

        """
        self.logger.info('Exporting trained model from {}', self.model_dir)

        with tf.Session() as sess:
            with tf.device(self.device):
                builder = tf.saved_model.builder.SavedModelBuilder(
                    self.model_dir)
                saver = tf.train.import_meta_graph(
                    '{}.meta'.format(self.model_dir))
                saver.restore(sess, self.model_dir)
                self.logger.info('Loading trained model !')

                graph = tf.get_default_graph()
                inputs = self._get_tensor_map(graph, self.input_tensor_map)
                outputs = self._get_tensor_map(graph, self.output_tensor_map)

                self.logger.info('Exporting...')
                signature = (
                    tf.saved_model.signature_def_utils.build_signature_def(
                        inputs, outputs,
                        tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
                builder.add_meta_graph_and_variables(
                    sess, ['frozen_model'],
                    signature_def_map={self.model_name: signature})
                builder.save()

        self.logger.info('Done exporting!')


class BasePredictor(_Base):
    """
    使用 Saved Model 进行预测
    """

    def __init__(self, model_dir,
                 model_name,
                 data,
                 output_tensors,
                 logger=None,
                 config=None,
                 cpu_only=False):
        """

        Parameters
        ----------
        model_dir: 导出的 pb 文件的路径
        model_name: 导出的模型标记名
        data: 可迭代的数据对象
        output_tensors: 所需要的输出数据的张量名称
        logger: 日志对象
        config: tensorflow session 的配置对象
        cpu_only
        """
        assert isinstance(data, Iterable), '请输入正确的数据！data必须是可迭代的对象。'
        super(BasePredictor, self).__init__(model_dir, model_name, config,
                                            logger, cpu_only)
        self.data = data
        self.config.gpu_options.allow_growth = True

        self.input_tensors = None
        self.output_tensors = output_tensors
        self.buf_data = {}

    def _prepare_output_tensor_map(self, sess, signature):
        """
        获取输出变量的关键字和张量字典

        Returns
        -------

        """
        output_tensor_map = {}
        for tensor_name in self.output_tensors:
            tensor = signature.inputs[tensor_name].name
            tensor = sess.graph.get_tensor_by_name(tensor)
            output_tensor_map[tensor_name] = tensor
        self.output_tensors = output_tensor_map

    def _prepare_input_data_map(self, sess, signature):
        """
        实现出来计算所需的输入

        Returns
        -------

        """
        raise NotImplementedError

    def _evaluate_result(self):
        """
        对预测结果的评估

        Returns
        -------

        """
        raise NotImplementedError

    def predict(self):
        """
        执行预测操作

        Returns
        -------

        """
        with tf.Session(config=self.config) as sess:
            with tf.device(self.device):
                meta_graph_def = tf.saved_model.loader.load(sess,
                                                            ['frozen_model'],
                                                            self.model_dir)
                signature = meta_graph_def.signature_def[self.model_name]
                self.logger.info('Loading model completed !')
                self._prepare_output_tensor_map(sess, signature)

                for data in self.data:
                    self.buf_data['data'] = data
                    self._prepare_input_data_map(sess, signature)
                    self.buf_data['result'] = (
                        sess.run(self.output_tensors,
                                 feed_dict=self.input_tensors))
                    self.logger.info('Predicting {} ...'.format(data['name']))
                    self._evaluate_result()
