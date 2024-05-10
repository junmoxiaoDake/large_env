# large_env
包含了大规模星际争霸多智能体环境（SMAC_Large）、大规模谷歌足球多智能体环境（GF_Large）、大规模神经网络在线游戏环境（NMMO_Large）以及大规模无人机对战模拟平台（Drone_Large）等四个大规模环境代码实现。

安装步骤，首先是将对应的代码克隆进服务器。
```python
git clone https://github.com/junmoxiaoDake/large_env.git
```

## 安装SMAC_Large步骤：
```python
cd smac_large
```
切换进smac_large文件夹后，采用下述命令安装smac_large

```python
pip install -e smac_large/
```

需要下载星际争霸Ⅱ的游戏，并且在home目录下的.bashrc里进行配置游戏位置
```python
export SC2PATH="/data/zpq/StarCraftII" #其中的/data/zpq/StarCraftII为当前的星际争霸Ⅱ所在的游戏位置。
```

采用下述命令运行smac_large来检测是否安装成功。
```python
python -m smac.bin.map_list 
```

## 安装GF_Large步骤：

首先需要安装如下的环境依赖包
```python

#linux系统
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip
python3 -m pip install --upgrade pip setuptools psutil wheel

#macOS系统
brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3

python3 -m pip install --upgrade pip setuptools psutil wheel

#windows系统
python -m pip install --upgrade pip setuptools psutil wheel
```

切换进GF_large目录
```python
cd gf_large
```

进行编译安装

```python
python3 -m pip install .
```

检测运行：

```python
python3 -m gfootball.play_game --action_set=full
```

## 安装NMMO_Large步骤：

```python
cd nmmo_large
bash scripts/setup/setup.sh
python setup.py
```

测试运行：

```python
python Forge.py --render #Run the environment with rendering on
```

采用不同的模式进行运行：

```python
# Run Options:
python Forge.py --nRealm 2 --api native #Run 2 environments with native API
python Forge.py --nRealm 2 --api vecenv #Run 2 environments with vecenv API
```
## 安装Drone_Large步骤：
依赖包：
* python==3.6.1
* gym==0.9.2 (might work with later versions)
* matplotlib if you would like to produce Ising model figures


