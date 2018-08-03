#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/4/4 20:30
@desc: 
"""
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from abc import abstractmethod
from pathlib import Path

import cv2
from dltools.data.dataset.cropper.rect_cropper import RectCropper
from dltools.utils.basic import is_rectangle_overlap
from dltools.utils.io import read_voc_xml


def add_element(root, name, value):
    """
    向xml元素树的 `root` 节点添加一个名为 `name`，
    值为 `value` 的子节点

    Parameters
    ----------
    root: 根节点
    name: 要添加的子节点的名字
    value: 要添加的子节点的值

    Returns
    -------

    """
    sub_element = ET.SubElement(root, name)
    sub_element.text = value


class VOCImageCropper(RectCropper):
    """
    VOC 数据格式图像的切割类
    """

    def __init__(self, image_path, xml_path,
                 output, crop_size, stride,
                 threshold=0.8, label_list=None,
                 logger=None, scope=None):
        """

        Parameters
        ----------
        image_path: 图像路径
        output: 输出文件夹路径
        logger: 日志对象
        crop_size: 子图尺寸
        stride: 切割步长
        xml_path: 标注文件
        threshold: 对象在子图中可以接受的阈值
        label_list: 对象名称列表，为 None 测考察全部对象
        scope: scope name
        """
        self._xml_path = xml_path
        self._output = output
        self._image_info = None
        self._label_list = label_list
        self._image_path = image_path
        self._scope = scope
        super(VOCImageCropper, self).__init__(image_path, crop_size,
                                              stride, logger)
        self._threshold = threshold
        self._do_write = True

    @abstractmethod
    def _preprocess(self):
        """
        预判断, 图像或标注文件是否符合要求

        Returns
        -------

        """
        # 确认输出路径
        if self._output is not None:
            self._scope = self._scope or Path(self._image_path).stem
            self._output = Path(self._output) / self._scope
            if not self._output.exists():
                self._output.mkdir(parents=True)
        else:
            self._do_write = False

        self._image_info, break_instance = read_voc_xml(
            self._xml_path, self.image.shape, self._label_list)
        if break_instance or (len(self._image_info['objects']) == 0):
            with open(self._output.parent.parent / 'break.txt',
                      'a', encoding='utf-8') as out:
                out.write(self._image_path + '\n')
            return True
        return False

    @abstractmethod
    def _crop_image(self):
        """
        切割子图

        Returns
        -------

        """
        x, y = self._buf_data['x'], self._buf_data['y']
        w_min, w_max = x, x + self._size[1] - 1
        h_min, h_max = y, y + self._size[0] - 1
        output_objects = []
        for ob in self._image_info['objects']:
            box = ob['box']
            new_ob = ob.copy()
            if not is_rectangle_overlap(box, [h_min, w_min, h_max, w_max]):
                continue
            else:
                xmin, ymin = max(w_min, box[1]), max(h_min, box[0])
                xmax, ymax = min(w_max, box[3]), min(h_max, box[2])
                area = (xmax - xmin + 1) * (ymax - ymin + 1)
                ob_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
                if (area / ob_area) >= self._threshold:
                    new_ob['box'] = [ymin - y, xmin - x, ymax - y, xmax - x]
                    output_objects.append(new_ob)
        if len(output_objects) > 0:
            self._is_write = self._do_write
            self._buf_data['objects'] = output_objects
            self._buf_data['sub_image'] = (
                self.image[h_min:h_max + 1, w_min:w_max + 1, :])
        else:
            self._is_write = False

    @abstractmethod
    def _write_image(self):
        """
        将子图和标注信息保存

        Returns
        -------

        """
        x, y = self._buf_data['x'], self._buf_data['y']
        image_name = str(self._output / '{}_{}'.format(y, x))
        cv2.imwrite(image_name + '.jpg', self._buf_data['sub_image'])
        root = ET.Element('annotation')
        add_element(root, 'src_img', self._image_path)
        add_element(root, 'xml_path', self._xml_path)
        size = ET.SubElement(root, 'size')
        add_element(size, 'src_height', str(self.image.shape[0]))
        add_element(size, 'src_width', str(self.image.shape[1]))
        add_element(size, 'y', str(y))
        add_element(size, 'x', str(x))
        add_element(size, 'height', str(self._size[0]))
        add_element(size, 'width', str(self._size[1]))
        add_element(size, 'depth', str(self.image.shape[2]))
        for ob in self._buf_data['objects']:
            box = ob['box']
            ob_xml = ET.SubElement(root, 'object')
            add_element(ob_xml, 'name', ob['name'])
            add_element(ob_xml, 'difficult', 'Unspecified')
            add_element(ob_xml, 'pose', '0')
            add_element(ob_xml, 'truncated', '0')
            bndbox = ET.SubElement(ob_xml, 'bndbox')
            add_element(bndbox, 'xmin', str(box[1]))
            add_element(bndbox, 'ymin', str(box[0]))
            add_element(bndbox, 'xmax', str(box[3]))
            add_element(bndbox, 'ymax', str(box[2]))
        rough_string = ET.tostring(root, 'utf-8')
        reared_content = minidom.parseString(rough_string)
        with open(image_name + '.xml', 'w', encoding='utf-8') as fs:
            reared_content.writexml(
                fs, addindent=' ' * 4, newl='\n', encoding='utf-8')
