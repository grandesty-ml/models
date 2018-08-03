#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd ${DIR}/

# 删除之前的生成与安装文件
rm -r build/
rm object_detection/protos/*_pb2.py

# 编译 Proto 文件
protoc object_detection/protos/*.proto \
--python_out=.

# 创建包
python setup.py build
