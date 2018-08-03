# TensorFlow Models

This repository contains a number of different models implemented in [TensorFlow](https://www.tensorflow.org):

The [official models](official) are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest stable TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read. We especially recommend newer TensorFlow users to start here.

The [research models](https://github.com/tensorflow/models/tree/master/research) are a large collection of models implemented in TensorFlow by researchers. They are not officially supported or available in release branches; it is up to the individual researchers to maintain the models and/or provide support on issues and pull requests.

The [samples folder](samples) contains code snippets and smaller models that demonstrate features of TensorFlow, including code presented in various blog posts.

The [tutorials folder](tutorials) is a collection of models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

## Contribution guidelines

If you want to contribute to models, be sure to review the [contribution guidelines](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE)

## 开发提示

- 这个项目中，我们开发的额外功能全部放在 `./research/hnu_dl_tools/dltools` 下。
- 对于该项目原来自带的代码请不要随意改动，除非能够保证原本的功能不受影响。

## 安装教程

### 安装 `Python` 和 `TensorFlow`

1. 安装 `python`，建议采用 `Miniconda` 或者 `Anaconda`，可以很方便管理包和环境；
2. 安装 `TensorFlow` 最新发行版，可以参考[相关文档](https://www.tensorflow.org/install/)；
3. 安装 `pillow`, `jupyter`, `matplotlib`, `lxml`等依赖项，通过`pip`或者`conda`完成安装；
4. 安装 `Protoc`，[项目地址](https://github.com/google/protobuf)；
5. (可选)安装 `git`。

### 获取代码

假设你已经安装好 `Python` 和 `TensorFlow`。
可以直接下载 `zip` 文件，然后解压。
或者已经安装 `git`，打开命令行，使用如下命令获取代码：

```bash
git clone https://gitee.com/study-cooperation/models.git
```

假设 `models` 位于 `HERE` 目录下。

### 编译`Proto`文件

打开命令终端，进入 `models` 代码根目录，确保已经激活相关 `Python` 环境，然后执行

```bash
cd ${HERE}/models
./research/setup.sh
```

**注意**在 `windows` 环境下，建议使用类似 `unix` 终端的工具，例如 `Git Bash`、`Cygwin` 等，使用 `CMD` 或者 `PowerShell` 可能会导致如下错误：

```bash
object_detection/protos/*.proto: No such file or directory
```

在 `build/lib` 下就能看到 `object_detection` 、`dltools` 和 `slim` 相关的包。

### 添加环境变量

修改 `~/.bashrc` 文件，将上述的两个包添加进 `Python` 的搜索路径。

```bash
vim ~/.bashrc
```

在打开的 vim 编辑内容的最后添加：

```bash
export PYTHONPATH=${HERE}/models/research/build/lib:${PYTHONPATH}
```

保存退出。然后执行下面的命令使环境变量生效：

```bash
source ~/.bashrc
```
