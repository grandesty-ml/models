#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/9 21:15
@desc: 
"""
import xml.etree.ElementTree as ET

from object_detection.utils import label_map_util

from dltools.utils.basic import check_consistent_length


def read_label_from_pd_file(pd_file, class_num):
    """
    读取 .pdtxt 文件并返回类别索引

    Parameters
    ----------
    pd_file
    class_num

    Returns
    -------
    category index, like:
                            {1: {'id': 1, 'name': 'fisheye'},
                             2: {'id': 2, 'name': 'person'},
                             ...}
    """
    label_map = label_map_util.load_labelmap(pd_file)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=class_num, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def get_names_from_category_index(category_index=None,
                                  pd_file='', class_num=0):
    """
    从类别索引或 .pdtxt 文件获取类名

    Parameters
    ----------
    category_index
    pd_file
    class_num

    Returns
    -------

    """
    label_list = []
    if pd_file != '':
        category_index = read_label_from_pd_file(pd_file, class_num)
    for idx in range(len(category_index)):
        item = category_index[idx + 1]
        label_list.append(item['name'])
    return label_list


def read_voc_xml(file_path, image_size,
                 label_list=None,
                 default_name='no_type'):
    """
    读取 VOC 格式的 xml 文件并提取对象信息

    Parameters
    ----------
    file_path
    image_size
    label_list
    default_name

    Returns
    -------

    """
    image_info = {}
    tree = ET.parse(file_path.strip())
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    image_info['objects'] = []

    def _read_xml(changed=False):
        for obj in root.iter('object'):
            try:
                cls_name = obj.find('name').text
            except KeyError:
                cls_name = default_name
            if label_list is not None:
                if cls_name not in label_list:
                    continue
            xml_box = obj.find('bndbox')
            xmin = int(xml_box.find('xmin').text)
            ymin = int(xml_box.find('ymin').text)
            xmax = int(xml_box.find('xmax').text)
            ymax = int(xml_box.find('ymax').text)
            if changed:
                xmin, ymin, xmax, ymax = ymin, xmin, ymax, xmax
            label = -1 if label_list is None else label_list.index(cls_name)
            image_info['objects'].append({'name': cls_name,
                                          'label': label,
                                          'box': [ymin, xmin, ymax, xmax]})

    image_info['shape'] = {'width': image_size[1], 'height': image_size[0]}
    if (image_size[0] == height) and (image_size[1] == width):
        _read_xml()
        return image_info, False
    elif (image_size[0] == width) and (image_size[1] == height):
        _read_xml(True)
        return image_info, False
    else:
        return None, True


def write_text_file(file_name, *file_lists, split=' ', encoding='w'):
    """
    将多个字符串列表写入文本文件中

    Parameters
    ----------
    file_name
    file_lists
    split
    encoding

    Returns
    -------

    """
    check_consistent_length(*file_lists)
    with open(file_name, encoding) as file:
        for strings in zip(*file_lists):
            string = split.join(strings)
            file.write(string + '\n')


def create_map_pdtxt(file_path, label_list):
    """
    根据已有的类别生成 .pdtxt 文件

    Parameters
    ----------
    file_path
    label_list

    Returns
    -------

    """
    base_str = """item {}\n  id: {}\n  name: '{}'\n{}\n"""
    string = []
    for idx, label in enumerate(label_list):
        string.append(base_str.format('{', idx + 1, label, '}'))
    write_text_file(file_path, string)
