# 深度学习环境管理及安装

**由于我们实验的过程中可能会涉及到不同版本的Pytorch以及cuda，因此一个规范的环境管理是十分重要的。对于深度学习，我们一般会选择Anaconda来进行我们的包管理**

## 一.Anaconda

### 1.base环境

​	对于我们的电脑，最基本的配置环境是`base`环境，例如当我们打开`cmd`时，就会出现以下情况：

![image](https://i.imgur.com/HgNOCZN.png)

最前面括号里的名字就表示我们当前使用的环境是`base`环境，此时我们执行任何操作都是在`base`环境中进行的。不过通常来说，我们并不会在`base`中直接修改环境。

**单环境出现问题：**

- 高版本安装会把低版本的包给覆盖
- 多项目共用导致`base`环境过大
- 包管理复杂混乱

### 2.虚拟环境

**为了解决`base`环境所出现的问题，我们需要安装虚拟环境**

#### 2.1 定义

​	虚拟环境（Virtual Environment）是一个独立的、隔离的Python工作环境，它允许用户在同一台计算机上同时运行多个Python项目，而不会发生包和依赖关系的冲突。虚拟环境为每个项目提供了一个独立的Python解释器和独立的包库，从而确保不同项目之间的依赖关系互不干扰。

#### 2.3 环境创建

```python
conda create -n 环境名 python=3.xx
```

所有创建好的环境都在`Anaconda/env`文件夹下面，因此如果要拷贝环境的话，可以直接复制`env`下对应的虚拟环境文件夹

#### 2.4 环境激活

```python
conda activate 环境名
```

#### 2.5 环境退出

```python
conda deactivate
```

#### 2.6 环境删除

```python
conda remove -n 环境名 --all
```

#### 2.7 环境包管理

```python
conda list
pip list
```

#### 2.8 环境包安装

```python
conda install 包名字==版本号
pip intall  包名字==版本号
```

#### 2.9 更换下载源

​	由于在虚拟环境中直接使用pip或者conda安装都是从国外的源进行下载，下载速度非常的慢，因此我们可以通过更换清华源的方法来提高下载速度。

首先要在`Anaconda/env/虚拟环境名称`文件夹下新建一个txt文件，并输入

```python
[global]
index-url = http://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host = mirrors.aliyun.com
```

将其命名为`pip.ini`即可

## 二.CUDA

### 1.GPU基础

​	对计算机而言，中央处理器 CPU 是主板上的芯片，图形处理器 GPU 是显卡上的芯片。每台计算机必有主板，但少数计算机可能没有显卡。显卡可以用来加速深度学习的运算速度（GPU 比 CPU 快 10-100 倍）。

### 2.CUDA以及cuda

#### 2.1 CUDA

​	NVIDIA 显卡中的运算平台是 CUDA，是显卡中内置的CUDA，该组件是加速GPU的并行计算能力。CUDA的安装一般通过英伟达的[官网]([CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive))进行直接的安装。

##### 2.1.1 CUDA安装

​	安装之前，我们首先要查看自己GPU的型号以及支持的最高的CUDA版本

```
nvidia-smi
```

![image](https://i.imgur.com/xmv16A5.png)

上面的`CUDA Version`则是GPU支持的最高版本，因此从[官网]([CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive))上下载时必须小于等于该版本才能够正常运行

下面就是安装的界面。

<img src="https://i.imgur.com/KZT0PfX.png" alt="image" style="zoom:80%;" />

安装的时候选择自定义安装，并且把所有的驱动组件都勾选上

<img src="https://imgur.com/ssSjRJ8.png" alt="image-20240614123937878" style="zoom:80%;" />

安装完成后，利用下面的命令检查是否安装成功

```
nvcc -V
```

#### 2.2 cuda

​	cuda是Pytorch中内置的CUDA，为了区分，用小写表示。引入cuda的原因就是为了能够方便动态的去调整CUDA的版本而不需要重新安装CUDA

**一般来讲，要满足：CUDA 版本≥cuda 版本。**

##### 2.2.1 cuda安装

​	cuda的安装会随着Pytorch的安装一起进行，所以只需要安装好Pytorch即可

## 三.Pytorch

### 1.Pytorch组成

​	PyTorch 一分为三：torch、torchvision 与 torchaudio。这三个库中，torch 有2G 左右，而 torchvision 和 torchaudio 只有 2M 左右，因此一般在代码里只会

`import torch`。当 torch 的版本给定后，另外两个附件的版本也唯一确定了。

### 2.Pytorch安装

​	安装 torch 前，先给出一张安装表，安装Pytorch时，需要根据自己虚拟环境中Python的版本进行选择。

![image-20240614124845081](https://imgur.com/W3ZIt1n.png)

进入 PyTorch [官网](https://pytorch.org/get-started/previous-versions/)，在官网中根据自己的安装系统和想要安装的cuda版本选择对应的安装命令。这里以windows系统11.8的cuda版本为例。

```python
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

该语句共安装了四个组件，`pytorch,torchvision,torchaudio以及cuda11.8`，因此在安装Pytorch时就会把cuda给安装进去。

#### 2.1 安装问题

​	在安装的时候，如果因为网速问题安装失败，则可以通过下载`pip3`指令中的`index-url`网址下载对应的`pytorch,torchvision,torchaudio`即可。

![image-20240614175900941](https://imgur.com/BffQgSK.png)

选入对应的下载模块，进入链接后选中想要下载的版本，将`.whl`文件下载好后，放在合适的文件夹下，利用下面的命令进行本地安装

```python
pip install Location
```

## 四.安装验证

### 1.查看当前环境的所有库

​	根据安装的指令，我们可以直接查看对应的库中是否存在`pytorch,torchvision,torchaudio以及cuxx.x`安装包

```
conda list
pip list
```

例如运行pip list时

![image-20240614180829806](https://imgur.com/sAwdNF1.png)

可以发现三个组件都存在，并且是+cu121的，也就是说cu12.1也安装上了

### 2.**进入** **Python** 解释器检验

​	进入python指令，直接运行python语句检测torch及cuda版本

```
python
import torch
torch.cuda.is_avaliable()
torch.version.cuda
```

![image-20240614181013766](https://imgur.com/jHkZ205.png)