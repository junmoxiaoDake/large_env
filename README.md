# large_env
包含了大规模星际争霸多智能体环境（SMAC_Large）、大规模谷歌足球多智能体环境（GF_Large）、大规模神经网络在线游戏环境（NMMO_Large）以及大规模无人机对战模拟平台（Drone_Large）等四个大规模环境代码实现。

安装步骤，首先是将对应的代码克隆进服务器。
```python
git clone https://github.com/junmoxiaoDake/large_env.git
```

安装SMAC_Large步骤：
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

