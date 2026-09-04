# Federated Rare-Earth and Image Classification Framework

本仓库同时支持三类任务：

- `rare`：三客户端稀土价格时间序列预测
- `mnist`：三客户端图像分类
- `cifar10`：三客户端图像分类

如果你按本文档建议的 `workspace/data + workspace/src` 目录组织工程，仓库默认配置里的相对路径可以直接工作，无需额外修改 `data.split_dir`。

如果你准备在 `tmux` 里长时间跑训练，或者使用 gRPC 端口外部流量监控，建议在机器上提前确认以下 Linux 命令可用：

- `tmux`：推荐用于多节点部署、长时间训练和断线后恢复终端会话
- `pkill`：通常由 `procps` 或 `procps-ng` 提供，用于清理残留监控进程
- `tcpdump`：用于抓取指定 gRPC 端口的实际 TCP 流量

最简单的自检方式：

```bash
which tmux
which pkill
which tcpdump
```

## 1. 工作区组织

推荐先创建一个独立工作区，例如 `workspace/`，然后把原始数据和代码分别放到 `data/` 与 `src/`：

```bash
mkdir -p workspace/data
cd workspace
git clone https://github.com/Zhang-Yukun/fl.git src
```

推荐目录结构如下：

```text
workspace/
├── data/
│   ├── raw_data/
│   │   ├── rare/                  # rare 的 rawdata2 Excel 原始文件
│   │   ├── mnist/                 # MNIST 原始下载目录
│   │   └── cifar10/               # CIFAR-10 原始下载目录
│   ├── rare/                      # rare 预处理后的联邦数据目录
│   ├── mnist/                     # MNIST 预处理后的联邦数据目录
│   └── cifar10/                   # CIFAR-10 预处理后的联邦数据目录
└── src/
    ├── configs/
    ├── fedlab/
    ├── scripts/
    ├── requirements.txt
    └── README.md
```

进入代码目录后，后文所有命令默认都在 `src/` 下执行：

```bash
cd workspace/src
```

## 2. 环境与依赖安装

### 2.1 使用 `requirements.txt` 安装运行依赖

框架当前运行时实际使用到的第三方包已经包含在 `src/requirements.txt` 中，包括：

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `PyYAML`
- `loguru`
- `wandb`
- `grpcio`
- `einops`
- `reformer-pytorch`
- `matplotlib`
- `scipy`
- `xlrd`

推荐使用 conda 创建环境，Python 版本建议使用 `3.13` 或 `3.14`。

```bash
cd workspace
conda create -n torch_env python=3.14 -y
conda activate torch_env
pip install --upgrade pip
pip install -r src/requirements.txt
```

如果你本地更方便使用 `3.13`，也可以改成：

```bash
conda create -n torch_env python=3.13 -y
conda activate torch_env
pip install --upgrade pip
pip install -r src/requirements.txt
```

如果你还需要运行测试，可以额外安装：

```bash
pip install pytest
```

如果你要在 `tmux` 里管理多进程训练，或在服务器节点上对 gRPC 端口做外部流量监控，除了 Python 依赖外，还建议系统层面提供：

> 当前 `scripts/monitor_tcp_port_traffic.sh` 依赖 `tcpdump` 抓包，并使用 `tshark` 解析统计；不同发行版的安装包名可能不同，但最终需要保证这两个命令都可用。


- `tmux`
- `tcpdump`
- `tshark`
- `pkill`

在常见 Debian / Ubuntu 系统上通常对应：

```bash
sudo apt-get update
sudo apt-get install -y tmux tcpdump tshark procps
```

在常见 CentOS / Rocky / AlmaLinux 系统上通常对应：

```bash
sudo yum install -y tmux tcpdump wireshark-cli procps-ng
```

### 2.2 快速检查安装是否成功

```bash
cd workspace/src
python -m fedlab.entrypoints.train --config configs/rare/fedavg.yaml --help
python -m fedlab.tools.prepare_image_classification_data --help
python -m fedlab.tools.analyze_experiment_suite --help
```

## 3. 数据准备

如果你是把服务端和三个客户端分开部署，推荐先在任意一台机器上完成一次完整数据预处理，再把下面第 3.5 节列出的最小必要文件分别拷贝到服务端和各客户端机器。

### 3.1 框架最终期望的联邦数据布局

框架不会在训练时动态切分数据，而是直接读取 `data.split_dir` 指向的预切分目录。

如果你还没创建原始数据目录，推荐先执行一次：

```bash
cd workspace/src
mkdir -p ../data/raw_data/rare ../data/raw_data/mnist ../data/raw_data/cifar10
```

`rare` 默认读取：

```yaml
data:
  split_dir: ../data/rare
  clients: [Nd2O3, CeO2, La2O3]
```

`mnist` 默认读取：

```yaml
data:
  split_dir: ../data/mnist
  clients: [m1, m2, m3]
```

`cifar10` 默认读取：

```yaml
data:
  split_dir: ../data/cifar10
  clients: [c1, c2, c3]
```

### 3.2 `rare` 原始数据如何放置

`rare` 以后统一只使用 `rawdata2` Excel 原始数据，不再使用 `XT_data`。

推荐把 `rare` 原始文件放到：

```text
workspace/data/raw_data/rare/
├── *.xls
└── *.xlsx
```

最常见的情况就是这里直接放三个 Excel，分别对应三个稀土客户端（下载的三个 excel 直接放到该目录下即可）；只要文件内容里的 `品名` 列能区分出：

- `氧化钕` -> `Nd2O3`
- `氧化铈` -> `CeO2`
- `氧化镧` -> `La2O3`

那就可以直接预处理，不要求再额外分客户端子目录。

每个 Excel 至少需要包含列：

- `日期`
- `品名`
- `平均价`
- `单位`
- `价格类型`
- `付款方式`

同时要求满足：

- 单位包含 `元/吨`
- 价格类型包含 `出厂价`
- 付款方式包含 `含税现款`

如果 `raw_data/rare/` 下有多个属于同一客户端的 Excel，预处理脚本会先按日期合并；若日期重复，会对同一天的数值取平均。之后再构造三客户端的日级插值宽表，并按时间顺序切分成 `train/val/test`。

生成联邦训练数据：

```bash
cd workspace/src
python -m fedlab.tools.prepare_rawdata2 \
  --raw-dir ../data/raw_data/rare \
  --output-dir ../data/rare
```

### 3.3 `mnist` / `cifar10` 原始数据如何放置

图像分类数据推荐统一放到 `workspace/data/raw_data/mnist` 和 `workspace/data/raw_data/cifar10`；也可以让框架自动下载到这两个目录。

#### 推荐方式：让框架自动下载并切分

```bash
cd workspace/src
python -m fedlab.tools.prepare_image_classification_data   --dataset all   --raw-root ../data/raw_data   --output-root ../data   --num-clients 3   --val-ratio 0.1   --seed 2026
```

只准备 MNIST：

```bash
python -m fedlab.tools.prepare_image_classification_data   --dataset mnist   --raw-root ../data/raw_data   --output-root ../data
```

只准备 CIFAR-10：

```bash
python -m fedlab.tools.prepare_image_classification_data   --dataset cifar10   --raw-root ../data/raw_data   --output-root ../data
```

#### 可选方式：手动 `wget` 原始数据

MNIST 原始文件可放到 `workspace/data/raw_data/mnist/MNIST/raw/`：

```bash
mkdir -p ../data/raw_data/mnist/MNIST/raw
wget -O ../data/raw_data/mnist/MNIST/raw/train-images-idx3-ubyte.gz   https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget -O ../data/raw_data/mnist/MNIST/raw/train-labels-idx1-ubyte.gz   https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget -O ../data/raw_data/mnist/MNIST/raw/t10k-images-idx3-ubyte.gz   https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget -O ../data/raw_data/mnist/MNIST/raw/t10k-labels-idx1-ubyte.gz   https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
```

CIFAR-10 原始文件可放到 `workspace/data/raw_data/cifar10/`：

```bash
mkdir -p ../data/raw_data/cifar10
wget -O ../data/raw_data/cifar10/cifar-10-python.tar.gz   https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf ../data/raw_data/cifar10/cifar-10-python.tar.gz   -C ../data/raw_data/cifar10
```

手动下载后，仍然使用 `prepare_image_classification_data` 生成联邦训练数据。

### 3.4 预处理完成后的目录结构

`rare` 预处理完成后通常会得到：

```text
../data/rare/
├── clients/
│   ├── Nd2O3/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── CeO2/
│   └── La2O3/
├── server/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── merged_wide.csv
└── summary.json
```

`mnist` / `cifar10` 预处理完成后通常会得到：

```text
../data/mnist/
├── clients/
│   ├── m1/
│   │   ├── train.pt
│   │   ├── val.pt
│   │   └── test.pt
│   ├── m2/
│   └── m3/
├── server/
│   ├── train.pt
│   ├── val.pt
│   └── test.pt
└── summary.json
```

CIFAR-10 同理，只是客户端目录为 `c1/c2/c3`。

### 3.5 多节点最小数据准备步骤

如果四台机器分别准备数据，最简单也最稳妥的流程是：

1. 先在任意一台机器上按上面的方式准备完整数据目录。
2. 保证四台机器上的 `workspace/src` 配置一致。
3. 再把最小必要数据按角色分发到服务端和客户端。
4. 最好让所有机器都保持相同的相对目录，例如都使用 `workspace/data/...`，这样通常不需要额外改 `data.split_dir`。

当前代码下，按训练阶段的最小需求准备即可：

#### 3.5.1 `rare` 的最小文件集合

如果只考虑联邦训练、不做攻击，服务端不再需要任何客户端 `train.csv`。
服务端最少需要三份客户端的 `val.csv`；如果你也希望训练结束时直接执行最终测试，再额外准备三份 `test.csv`。验证和测试会按客户端各自上传的 scaler 统计进行恢复：

```text
../data/rare/
└── clients/
    ├── Nd2O3/
    │   └── val.csv
    ├── CeO2/
    │   └── val.csv
    └── La2O3/
        └── val.csv
```

客户端只需要自己的 `train.csv`：

- `Nd2O3` 客户端：`../data/rare/clients/Nd2O3/train.csv`
- `CeO2` 客户端：`../data/rare/clients/CeO2/train.csv`
- `La2O3` 客户端：`../data/rare/clients/La2O3/train.csv`

#### 3.5.2 `mnist` / `cifar10` 的最小文件集合

图像任务下，服务端当前推荐最少需要：

- `server/val.pt`，用于共享验证
- 如果你希望训练结束时直接执行最终测试，再额外准备 `server/test.pt`

以 `mnist` 为例，服务端最少目录为：

```text
../data/mnist/
└── server/
    └── val.pt
```

如果没有单独准备 `server/val.pt`，服务端也可以退回读取各客户端的 `val.pt` 来拼接验证集；`test.pt` 同理可选。

客户端只需要自己的 `train.pt`：

- `m1` 客户端：`../data/mnist/clients/m1/train.pt`
- `m2` 客户端：`../data/mnist/clients/m2/train.pt`
- `m3` 客户端：`../data/mnist/clients/m3/train.pt`

`cifar10` 同理，只是客户端目录换成 `c1/c2/c3`。

各客户端只需要自己的 `train.pt`：

- `m1` 客户端：`../data/mnist/clients/m1/train.pt`
- `m2` 客户端：`../data/mnist/clients/m2/train.pt`
- `m3` 客户端：`../data/mnist/clients/m3/train.pt`
- `c1` 客户端：`../data/cifar10/clients/c1/train.pt`
- `c2` 客户端：`../data/cifar10/clients/c2/train.pt`
- `c3` 客户端：`../data/cifar10/clients/c3/train.pt`

#### 3.5.3 一句话总结

- 服务端：准备验证集；如果需要训练结束后立刻做最终测试，再额外准备测试集
- 客户端 `i`：只准备自己的本地训练集
- 如果后续还要在某台机器上做离线攻击回放，那台机器还需要能访问被攻击客户端对应的训练集

#### 3.5.4 使用 `prepare_role_datasets` 直接生成按角色部署的目录

如果你已经先按第 3.2 和第 3.3 节完成了完整预处理，可以再执行一次：

```bash
cd workspace/src
python -m fedlab.tools.prepare_role_datasets \
  --task all \
  --source-root ../data \
  --output-root ../data/role_datasets
```

如果要给 `rare` 的离线测试目录同时带上训练输出里的 `evaluation_context.json`，可以额外传：

```bash
python -m fedlab.tools.prepare_role_datasets \
  --task rare \
  --source-root ../data \
  --output-root ../data/role_datasets
```

生成后的目录按角色组织，大致如下：

```text
../data/role_datasets/rare/
├── server/
│   ├── clients/
│   │   ├── Nd2O3/val.csv
│   │   ├── CeO2/val.csv
│   │   └── La2O3/val.csv
│   └── summary.json
├── client/
│   ├── Nd2O3/clients/Nd2O3/train.csv
│   ├── CeO2/clients/CeO2/train.csv
│   └── La2O3/clients/La2O3/train.csv
├── attack/
│   ├── clients/
│   │   ├── Nd2O3/train.csv
│   │   ├── CeO2/train.csv
│   │   └── La2O3/train.csv
│   └── summary.json
└── test/
    ├── clients/
    │   ├── Nd2O3/test.csv
    │   ├── CeO2/test.csv
    │   └── La2O3/test.csv
    ├── evaluation_context.json
    └── summary.json
```

如果你准备把这些角色目录真正分发到不同机器，建议不要直接在目标机器上保留 `role_datasets/...` 这一层，而是把对应角色目录里的内容复制到目标机器最终要读取的 `split_dir` 根目录下。

以 `rare` 为例，假设四类机器最终都使用各自本地的 `workspace/data/rare/` 作为读取根目录，那么部署后的目录建议长这样。

服务端机器：

```text
workspace/
├── data/
│   └── rare/
│       ├── clients/
│       │   ├── Nd2O3/val.csv
│       │   ├── CeO2/val.csv
│       │   └── La2O3/val.csv
│       └── summary.json
└── src/
```

这时服务端直接读取：

```yaml
data:
  split_dir: ../data/rare
```

也就是说，把 `../data/role_datasets/rare/server/` 目录里的内容整体复制到服务端机器的 `workspace/data/rare/` 下。

客户端 `Nd2O3` 机器：

```text
workspace/
├── data/
│   └── rare/
│       ├── clients/
│       │   └── Nd2O3/train.csv
│       └── summary.json
└── src/
```

客户端 `CeO2` 机器：

```text
workspace/
├── data/
│   └── rare/
│       ├── clients/
│       │   └── CeO2/train.csv
│       └── summary.json
└── src/
```

客户端 `La2O3` 机器：

```text
workspace/
├── data/
│   └── rare/
│       ├── clients/
│       │   └── La2O3/train.csv
│       └── summary.json
└── src/
```

也就是说，分别把：

- `../data/role_datasets/rare/client/Nd2O3/` 的内容复制到 `Nd2O3` 客户端机器的 `workspace/data/rare/clients/`
- `../data/role_datasets/rare/client/CeO2/` 的内容复制到 `CeO2` 客户端机器的 `workspace/data/rare/clients/`
- `../data/role_datasets/rare/client/La2O3/` 的内容复制到 `La2O3` 客户端机器的 `workspace/data/rare/clients/`

这样客户端仍然可以直接沿用默认的：

```yaml
data:
  split_dir: ../data/rare
```

攻击机（正常也是服务器）：

攻击机不必沿用默认的 `workspace/data/rare/`。如果你希望把攻击数据单独放到别的位置，比如 `workspace/data/attack/rare/attack`，那么部署后的目录可以直接是：

```text
workspace/data/attack/rare/attack
├── clients/
│   ├── Nd2O3/train.csv
│   ├── CeO2/train.csv
│   └── La2O3/train.csv
└── summary.json
```

也就是说，把 `../data/role_datasets/rare/attack/` 的内容整体复制到你真正打算给攻击脚本读取的那个目录即可。部署后，这个目录本身就是攻击回放原生可读的标准 `split_dir` 格式，可以直接传：

```bash
--override data.split_dir=../data/attack/rare/attack
```

测试机(正常也是服务器)：

测试机同样不必沿用默认的 `workspace/data/rare/`。如果你希望把离线测试数据单独放到别的位置，比如 `workspace/data/test/rare/test`，那么部署后的目录可以直接是：

```text
workspace/data/test/rare/test
├── clients/
│   ├── Nd2O3/test.csv
│   ├── CeO2/test.csv
│   └── La2O3/test.csv
├── evaluation_context.json
└── summary.json
```

也就是说，把 `../data/role_datasets/rare/test/` 的内容整体复制到你真正打算给离线测试脚本读取的那个目录即可。部署后离线测试时可以直接传：

```bash
--data-dir ../data/test/rare/test
```

因此：

- 服务端和训练客户端如果仍然沿用配置里的默认 `split_dir`，可以尽量保持原路径不变
- 攻击机和测试机则完全可以把数据放在任意别的位置，只要在攻击/测试命令里显式传入对应路径即可

`mnist` 和 `cifar10` 同理，只是文件从 `.csv` 变成 `.pt`：

- 服务端机器保留 `server/val.pt`
- 每个客户端机器只保留 `clients/<client_id>/train.pt`
- 攻击机保留 `clients/*/train.pt`
- 测试机保留 `server/test.pt` 或各客户端 `test.pt`

如果你选择直接保留 `role_datasets/<task>/server`、`role_datasets/<task>/client/<client_id>` 这种新根目录不变，那也可以，只是运行命令里需要显式覆盖 `data.split_dir`。

## 4. 训练方式

### 4.1 单机直接训练（一般只用集中式训练）

集中式训练：

rare数据

```bash
cd workspace/src
python -m fedlab.entrypoints.train   --config configs/rare/centralized.yaml  --mode centralized --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/centralized --override runtime.seed=4096
```

mnist 数据

```bash
cd workspace/src
python -m fedlab.entrypoints.train   --config configs/mnist/centralized.yaml  --mode centralized --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/centralized --override runtime.seed=4096
```

cifar10 数据

```bash
cd workspace/src
python -m fedlab.entrypoints.train   --config configs/cifar10/centralized.yaml  --mode centralized --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/centralized --override runtime.seed=4096
```

### 4.2 多进程 / 多节点 gRPC 联邦训练

#### 通用说明

多节点运行时：

- 服务端使用 `python -m fedlab.entrypoints.server`
- 客户端使用 `python -m fedlab.entrypoints.client`
- 需要先启动服务端，确认 `grpc.address` 对应端口已经开始监听后，再依次启动各个客户端
- 三端需要使用语义一致的配置
- 服务端 `grpc.address` 需要监听一个实际可访问的地址
- 客户端 `grpc.server_address` 需要指向服务器的实际 `IP:PORT`
- 客户端 `client-id` 必须与 `data.clients` 一致
- 下面示例里的默认地址使用 `127.0.0.1`，表示服务端和客户端都在本机运行；如果跨机器部署，请把它替换成服务器的实际 IP，并根据需要调整端口
- 客户端只需要能访问自己的本地训练数据；图像任务下通常只需要 `split_dir/clients/<client_id>/train.pt`，时间序列任务下通常只需要自己的 `train.csv`
- 服务端不执行攻击；训练阶段到底需要哪些最小数据，请直接以第 3.5 节和第 4.2.0 节为准
- EGA 首轮会先等待服务端准备 codec，然后再开始真正的第 0 轮训练；首次等待时间较长时属于正常现象

推荐把 `experiment.output_dir` 设到 `../outputs/...`，这样实验产物会落在 `src` 外层，避免污染代码目录。客户端会在本地额外写日志到：

```text
../outputs/<run_name>/client_<client_id>/
```

#### 4.2.0 训练阶段的最小数据需求

多节点联邦训练时，可以把“谁最少需要哪些数据”理解成下面这样：

- 单个客户端最少只需要自己的本地训练数据。
  对图像任务，通常就是 `split_dir/clients/<client_id>/train.pt`。
  对 `rare`，通常就是该客户端自己的预处理后 `train.csv`。
- 单个客户端通常不需要别的客户端的训练集，也不需要全局训练集。
- 单个客户端在当前默认训练流程下也不依赖自己的本地验证集和测试集；联邦训练阶段的验证/测试主要由服务端完成。
- 服务端在联邦训练阶段只负责验证和测试，不再读取客户端训练集。
  对 `mnist` / `cifar10`，推荐最少需要 `server/val.pt`；若缺失，服务端也可以退回读取各客户端的 `val.pt` 来拼接验证集。`server/test.pt` 是可选的，缺失时训练结束只保存模型、不执行最终测试。
  对 `rare`，最少需要三份客户端的 `val.csv`；三份 `test.csv` 同样是可选的，缺失时训练结束只保存模型、不执行最终测试。对应的标准化统计由客户端在注册阶段上传给服务端。
- 如果某个客户端和服务端部署在同一台机器上，也仍然只要求该客户端本地能访问自己的训练数据；别的客户端训练集缺失不会影响这个客户端进程本身启动，也不会影响服务端完成初始化，只要服务端自己的验证集齐全即可。

换句话说，当前框架在训练阶段的最小可行部署通常是：

- 服务端：服务端评估所需数据
- 客户端 `i`：自己的 `train`

如果你额外把本地 `val/test` 也放在客户端节点上不会出错，但不是训练必需条件。更具体的最小目录请直接看上面的第 3.5 节。

#### 4.2.0.1 gRPC 外部端口流量监控

正式多节点部署时，建议直接在服务器节点上监听服务端 `grpc.address` 对应的端口。也就是说，谁在跑 `python -m fedlab.entrypoints.server`，就在哪台机器上做监控。

推荐做法是：先在服务器节点启动监控，再启动联邦服务端和各个客户端；训练结束后停止监控。

例如服务端监听 `0.0.0.0:50051`，服务器对外通信网卡是 `eth0`，输出目录是 `../outputs/rare_fedavg_grpc`，可以在服务器节点执行：

```bash
cd workspace/src
bash scripts/monitor_tcp_port_traffic.sh \
  --port 50051 \
  --interface eth0 \
  --output-dir ../outputs/rare_fedavg_grpc/grpc_port_traffic
```

如果你不确定具体走哪块网卡，可以直接用 `any`：

```bash
cd workspace/src
bash scripts/monitor_tcp_port_traffic.sh \
  --port 50051 \
  --interface any \
  --output-dir ../outputs/rare_fedavg_grpc/grpc_port_traffic
```

说明：

- `--port` 应该填服务端实际监听的 gRPC 端口
- `--interface` 应该填服务器实际通信网卡；单机本地联调常见是 `lo`，跨机器部署常见是 `eth0`、`ens*` 或直接 `any`
- 监控结果会输出到 `grpc_port_traffic/` 目录
- 训练结束后在监控所在终端 `Ctrl-C` 即可，脚本会自动汇总统计结果

其中常见文件包括：

- `grpc_port_traffic.summary.json`
- `grpc_port_traffic.pcap`
- `grpc_port_traffic.capture.stderr.log`

`grpc_port_traffic.summary.json` 中：

- `received_bytes` / `sent_bytes`：按抓包 `frame.len` 汇总的端口接收/发送总流量
- `received_tcp_payload_bytes` / `sent_tcp_payload_bytes`：按 `tcp.len` 汇总的 TCP payload 字节，更接近应用层承载数据量
- 这套外部抓包结果适合做网络侧旁证，但不建议直接拿来替代框架内 `parameter_*` 统计，也不建议直接用于计算论文里的通信压缩比
- 抓包口径本身也会有误差：例如抓包进程权限/缓冲限制导致的漏包，或 TCP 重传、回环接口/多视角抓取带来的重复计数。因此它更适合做趋势核对和异常排查，而不是作为唯一通信真值

如果监控异常退出后有残留进程，可以在服务器节点清理：

```bash
ps -efww | grep monitor_tcp_port_traffic | grep -v grep
ps -efww | grep 'tcpdump -n -U -i' | grep -v grep
pkill -f 'scripts/monitor_tcp_port_traffic.sh'
pkill -f 'tcpdump -n -U -i'
```

如果你知道具体监控端口，也可以更精确地只清理该端口对应的 `tcpdump`：

```bash
pkill -f 'tcpdump -n -U -i .* tcp port 50051'
```

#### 4.2.0.2 启动顺序与轮询间隔

推荐启动顺序：

1. 先在服务器节点启动端口监控。
2. 再启动服务端。
3. 确认服务端日志已经显示监听地址，且端口可连通。
4. 最后启动全部客户端。

`grpc.poll_seconds` 控制的是 gRPC 训练过程中客户端和服务端的轮询休眠间隔，也就是“多久重试一次注册、拉取全局模型、提交更新、等待关闭确认”；它不是服务器主动推送消息的频率。

直接用 Python 入口时，服务端和客户端都可以通过 `--override grpc.poll_seconds=...` 控制：

```bash
python -m fedlab.entrypoints.server \
  --config configs/rare/fedavg.yaml \
  --override grpc.address=0.0.0.0:50051 \
  --override grpc.poll_seconds=5.0
```

```bash
python -m fedlab.entrypoints.client \
  --client-id Nd2O3 \
  --config configs/rare/fedavg.yaml \
  --override grpc.server_address=127.0.0.1:50051 \
  --override grpc.poll_seconds=5.0
```

#### 4.2.1 `rare` 的 FedAvg

（可选）启动服务器端流量监控

```bash
cd workspace/src
bash scripts/monitor_tcp_port_traffic.sh \
  --port 50051 \
  --interface any \
  --output-dir ../outputs/exp/rare/multi_sync/4096/fedavg/traffic
```

服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/rare/fedavg.yaml   --override grpc.address=0.0.0.0:50051   --override grpc.server_address=127.0.0.1:50051   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/fedavg --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

客户端 `Nd2O3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id Nd2O3   --config configs/rare/fedavg.yaml   --override grpc.server_address=127.0.0.1:50051   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/fedavg --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

客户端 `CeO2`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id CeO2   --config configs/rare/fedavg.yaml   --override grpc.server_address=127.0.0.1:50051   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/fedavg --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

客户端 `La2O3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id La2O3   --config configs/rare/fedavg.yaml   --override grpc.server_address=127.0.0.1:50051   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/fedavg --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

（可选）如果启动了服务器端流量监控

通过 Ctrl-c 来中断，可以自动从目录中获取 `grpc_port_traffic.summary.json`。其中 `received_bytes` / `sent_bytes` 是抓包口径下的端口总流量，`received_tcp_payload_bytes` / `sent_tcp_payload_bytes` 是更接近应用层承载数据量的 TCP payload 统计；它们适合做网络侧旁证，不建议直接替代框架内 `parameter_*` 统计来计算通信压缩比。


执行 DLG：

```bash
python -m fedlab.tools.replay_saved_update_dlg   ../outputs/exp/rare/multi_sync/4096/fedavg   --output-dir ../outputs/exp/rare/multi_sync/4096/fedavg/offline_attack_replay_dlg --override data.split_dir=../data/attack/rare/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

执行 iDLG：

```bash
python -m fedlab.tools.replay_saved_update_idlg   ../outputs/exp/rare/multi_sync/4096/fedavg   --output-dir ../outputs/exp/rare/multi_sync/4096/fedavg/offline_attack_replay_idlg --override data.split_dir=../data/attack/rare/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

测试：

```bash
python -m fedlab.tools.replay_saved_model_evaluation ../outputs/exp/rare/multi_sync/4096/fedavg/model.pt --config ../outputs/exp/rare/multi_sync/4096/fedavg/config.yaml --data-dir ../data/test/rare/test --evaluation-context ../outputs/exp/rare/multi_sync/4096/fedavg/evaluation_context.json --output-dir ../outputs/exp/rare/multi_sync/4096/fedavg/offline_test_replay --override runtime.device=cpu
```


#### 4.2.2 `rare` 的 Top-k

（可选）启动服务器端流量监控

```bash
cd workspace/src
bash scripts/monitor_tcp_port_traffic.sh \
  --port 50052 \
  --interface any \
  --output-dir ../outputs/exp/rare/multi_sync/4096/topk/traffic
```


服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/rare/topk.yaml   --override grpc.address=0.0.0.0:50052   --override grpc.server_address=127.0.0.1:50052   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/topk --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

客户端 `Nd2O3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id Nd2O3   --config configs/rare/topk.yaml   --override grpc.server_address=127.0.0.1:50052   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/topk --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

客户端 `CeO2`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id CeO2   --config configs/rare/topk.yaml   --override grpc.server_address=127.0.0.1:50052   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/topk --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

客户端 `La2O3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id La2O3   --config configs/rare/topk.yaml   --override grpc.server_address=127.0.0.1:50052   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/topk --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

（可选）如果启动了服务器端流量监控

通过 Ctrl-c 来中断，可以自动从目录中获取 `grpc_port_traffic.summary.json`。其中 `received_bytes` / `sent_bytes` 是抓包口径下的端口总流量，`received_tcp_payload_bytes` / `sent_tcp_payload_bytes` 是更接近应用层承载数据量的 TCP payload 统计；它们适合做网络侧旁证，不建议直接替代框架内 `parameter_*` 统计来计算通信压缩比。

执行 DLG：

```bash
python -m fedlab.tools.replay_saved_update_dlg   ../outputs/exp/rare/multi_sync/4096/topk   --output-dir ../outputs/exp/rare/multi_sync/4096/topk/offline_attack_replay_dlg --override data.split_dir=../data/attack/rare/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

执行 iDLG：

```bash
python -m fedlab.tools.replay_saved_update_idlg   ../outputs/exp/rare/multi_sync/4096/topk   --output-dir ../outputs/exp/rare/multi_sync/4096/topk/offline_attack_replay_idlg --override data.split_dir=../data/attack/rare/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

测试：

```bash
python -m fedlab.tools.replay_saved_model_evaluation ../outputs/exp/rare/multi_sync/4096/topk/model.pt --config ../outputs/exp/rare/multi_sync/4096/topk/config.yaml --data-dir ../data/test/rare/test --evaluation-context ../outputs/exp/rare/multi_sync/4096/topk/evaluation_context.json --output-dir ../outputs/exp/rare/multi_sync/4096/topk/offline_test_replay --override runtime.device=cpu
```

#### 4.2.3 `rare` 的 EGA

（可选）启动服务器端流量监控

```bash
cd workspace/src
bash scripts/monitor_tcp_port_traffic.sh \
  --port 50053 \
  --interface any \
  --output-dir ../outputs/exp/rare/multi_sync/4096/ega/traffic
```

服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/rare/ega.yaml   --override grpc.address=0.0.0.0:50053   --override grpc.server_address=127.0.0.1:50053   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/ega --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

客户端 `Nd2O3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id Nd2O3   --config configs/rare/ega.yaml   --override grpc.server_address=127.0.0.1:50053   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/ega --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

客户端 `CeO2`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id CeO2   --config configs/rare/ega.yaml   --override grpc.server_address=127.0.0.1:50053   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/ega --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

客户端 `La2O3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id La2O3   --config configs/rare/ega.yaml   --override grpc.server_address=127.0.0.1:50053   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/rare/multi_sync/4096/ega --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

（可选）如果启动了服务器端流量监控

通过 Ctrl-c 来中断，可以自动从目录中获取 `grpc_port_traffic.summary.json`。其中 `received_bytes` / `sent_bytes` 是抓包口径下的端口总流量，`received_tcp_payload_bytes` / `sent_tcp_payload_bytes` 是更接近应用层承载数据量的 TCP payload 统计；它们适合做网络侧旁证，不建议直接替代框架内 `parameter_*` 统计来计算通信压缩比。

执行 DLG：

```bash
python -m fedlab.tools.replay_saved_update_dlg   ../outputs/exp/rare/multi_sync/4096/ega   --output-dir ../outputs/exp/rare/multi_sync/4096/ega/offline_attack_replay_dlg --override data.split_dir=../data/attack/rare/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

执行 iDLG：

```bash
python -m fedlab.tools.replay_saved_update_idlg   ../outputs/exp/rare/multi_sync/4096/ega   --output-dir ../outputs/exp/rare/multi_sync/4096/ega/offline_attack_replay_idlg --override data.split_dir=../data/attack/rare/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

测试：

```bash
python -m fedlab.tools.replay_saved_model_evaluation ../outputs/exp/rare/multi_sync/4096/ega/model.pt --config ../outputs/exp/rare/multi_sync/4096/ega/config.yaml --data-dir ../data/test/rare/test --evaluation-context ../outputs/exp/rare/multi_sync/4096/ega/evaluation_context.json --output-dir ../outputs/exp/rare/multi_sync/4096/ega/offline_test_replay --override runtime.device=cpu
```

#### 4.2.4 `mnist` 的 FedAvg / Top-k / EGA

`mnist` 的客户端固定是：`m1`、`m2`、`m3`。下面只替换配置文件、客户端 ID、输出目录和端口。

#### FedAvg

（可选）启动服务器端流量监控

```bash
cd workspace/src
bash scripts/monitor_tcp_port_traffic.sh \
  --port 50061 \
  --interface any \
  --output-dir ../outputs/exp/mnist/multi_sync/4096/fedavg/traffic
```

FedAvg 服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/mnist/fedavg.yaml   --override grpc.address=0.0.0.0:50061   --override grpc.server_address=127.0.0.1:50061   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/mnist/multi_sync/4096/fedavg --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

FedAvg 客户端 `m1` / `m2` / `m3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m1   --config configs/mnist/fedavg.yaml   --override grpc.server_address=127.0.0.1:50061   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/mnist/multi_sync/4096/fedavg --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m2   --config configs/mnist/fedavg.yaml   --override grpc.server_address=127.0.0.1:50061   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/mnist/multi_sync/4096/fedavg --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m3   --config configs/mnist/fedavg.yaml   --override grpc.server_address=127.0.0.1:50061   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/mnist/multi_sync/4096/fedavg --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

（可选）如果启动了服务器端流量监控

通过 Ctrl-c 来中断，可以自动从目录中获取 `grpc_port_traffic.summary.json`。其中 `received_bytes` / `sent_bytes` 是抓包口径下的端口总流量，`received_tcp_payload_bytes` / `sent_tcp_payload_bytes` 是更接近应用层承载数据量的 TCP payload 统计；它们适合做网络侧旁证，不建议直接替代框架内 `parameter_*` 统计来计算通信压缩比。

执行 DLG：

```bash
python -m fedlab.tools.replay_saved_update_dlg   ../outputs/exp/mnist/multi_sync/4096/fedavg   --output-dir ../outputs/exp/mnist/multi_sync/4096/fedavg/offline_attack_replay_dlg --override data.split_dir=../data/attack/mnist/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

执行 iDLG：

```bash
python -m fedlab.tools.replay_saved_update_idlg   ../outputs/exp/mnist/multi_sync/4096/fedavg   --output-dir ../outputs/exp/mnist/multi_sync/4096/fedavg/offline_attack_replay_idlg --override data.split_dir=../data/attack/mnist/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

测试：

```bash
python -m fedlab.tools.replay_saved_model_evaluation ../outputs/exp/mnist/multi_sync/4096/fedavg/model.pt --config ../outputs/exp/mnist/multi_sync/4096/fedavg/config.yaml --data-dir ../data/test/mnist/test --output-dir ../outputs/exp/mnist/multi_sync/4096/fedavg/offline_test_replay --override runtime.device=cpu
```

#### Topk

（可选）启动服务器端流量监控

```bash
cd workspace/src
bash scripts/monitor_tcp_port_traffic.sh \
  --port 50062 \
  --interface any \
  --output-dir ../outputs/exp/mnist/multi_sync/4096/topk/traffic
```


Top-k 服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/mnist/topk.yaml   --override grpc.address=0.0.0.0:50062   --override grpc.server_address=127.0.0.1:50062   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/mnist/multi_sync/4096/topk --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

Top-k 客户端 `m1` / `m2` / `m3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m1   --config configs/mnist/topk.yaml   --override grpc.server_address=127.0.0.1:50062   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/mnist/multi_sync/4096/topk --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m2   --config configs/mnist/topk.yaml   --override grpc.server_address=127.0.0.1:50062   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/mnist/multi_sync/4096/topk --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m3   --config configs/mnist/topk.yaml   --override grpc.server_address=127.0.0.1:50062   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/mnist/multi_sync/4096/topk --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

（可选）如果启动了服务器端流量监控

通过 Ctrl-c 来中断，可以自动从目录中获取 `grpc_port_traffic.summary.json`。其中 `received_bytes` / `sent_bytes` 是抓包口径下的端口总流量，`received_tcp_payload_bytes` / `sent_tcp_payload_bytes` 是更接近应用层承载数据量的 TCP payload 统计；它们适合做网络侧旁证，不建议直接替代框架内 `parameter_*` 统计来计算通信压缩比。

执行 DLG：

```bash
python -m fedlab.tools.replay_saved_update_dlg   ../outputs/exp/mnist/multi_sync/4096/topk   --output-dir ../outputs/exp/mnist/multi_sync/4096/topk/offline_attack_replay_dlg --override data.split_dir=../data/attack/mnist/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

执行 iDLG：

```bash
python -m fedlab.tools.replay_saved_update_idlg   ../outputs/exp/mnist/multi_sync/4096/topk   --output-dir ../outputs/exp/mnist/multi_sync/4096/topk/offline_attack_replay_idlg --override data.split_dir=../data/attack/mnist/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

测试：

```bash
python -m fedlab.tools.replay_saved_model_evaluation ../outputs/exp/mnist/multi_sync/4096/topk/model.pt --config ../outputs/exp/mnist/multi_sync/4096/topk/config.yaml --data-dir ../data/test/mnist/test --output-dir ../outputs/exp/mnist/multi_sync/4096/topk/offline_test_replay --override runtime.device=cpu
```

##### EGA

（可选）启动服务器端流量监控

```bash
cd workspace/src
bash scripts/monitor_tcp_port_traffic.sh \
  --port 50063 \
  --interface any \
  --output-dir ../outputs/exp/mnist/multi_sync/4096/ega/traffic
```

EGA 服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/mnist/ega.yaml   --override grpc.address=0.0.0.0:50063   --override grpc.server_address=127.0.0.1:50063   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/mnist/multi_sync/4096/ega --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

EGA 客户端 `m1` / `m2` / `m3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m1   --config configs/mnist/ega.yaml   --override grpc.server_address=127.0.0.1:50063   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/mnist/multi_sync/4096/ega --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m2   --config configs/mnist/ega.yaml   --override grpc.server_address=127.0.0.1:50063   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/mnist/multi_sync/4096/ega --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m3   --config configs/mnist/ega.yaml   --override grpc.server_address=127.0.0.1:50063   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/mnist/multi_sync/4096/ega --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

（可选）如果启动了服务器端流量监控

通过 Ctrl-c 来中断，可以自动从目录中获取 `grpc_port_traffic.summary.json`。其中 `received_bytes` / `sent_bytes` 是抓包口径下的端口总流量，`received_tcp_payload_bytes` / `sent_tcp_payload_bytes` 是更接近应用层承载数据量的 TCP payload 统计；它们适合做网络侧旁证，不建议直接替代框架内 `parameter_*` 统计来计算通信压缩比。

执行 DLG：

```bash
python -m fedlab.tools.replay_saved_update_dlg   ../outputs/exp/mnist/multi_sync/4096/ega   --output-dir ../outputs/exp/mnist/multi_sync/4096/ega/offline_attack_replay_dlg --override data.split_dir=../data/attack/mnist/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

执行 iDLG：

```bash
python -m fedlab.tools.replay_saved_update_idlg   ../outputs/exp/mnist/multi_sync/4096/ega   --output-dir ../outputs/exp/mnist/multi_sync/4096/ega/offline_attack_replay_idlg --override data.split_dir=../data/attack/mnist/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

测试：

```bash
python -m fedlab.tools.replay_saved_model_evaluation ../outputs/exp/mnist/multi_sync/4096/ega/model.pt --config ../outputs/exp/mnist/multi_sync/4096/ega/config.yaml --data-dir ../data/test/mnist/test --output-dir ../outputs/exp/mnist/multi_sync/4096/ega/offline_test_replay --override runtime.device=cpu
```

#### 4.2.5 `cifar10` 的 FedAvg / Top-k / EGA

`cifar10` 的客户端固定是：`c1`、`c2`、`c3`。

##### FedAvg

（可选）启动服务器端流量监控

```bash
cd workspace/src
bash scripts/monitor_tcp_port_traffic.sh \
  --port 50071 \
  --interface any \
  --output-dir ../outputs/exp/cifar10/multi_sync/4096/fedavg/traffic
```

FedAvg 服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/cifar10/fedavg.yaml   --override grpc.address=0.0.0.0:50071   --override grpc.server_address=127.0.0.1:50071   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/cifar10/multi_sync/4096/fedavg --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

FedAvg 客户端 `c1` / `c2` / `c3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c1   --config configs/cifar10/fedavg.yaml   --override grpc.server_address=127.0.0.1:50071   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/cifar10/multi_sync/4096/fedavg --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c2   --config configs/cifar10/fedavg.yaml   --override grpc.server_address=127.0.0.1:50071   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/cifar10/multi_sync/4096/fedavg --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c3   --config configs/cifar10/fedavg.yaml   --override grpc.server_address=127.0.0.1:50071   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/cifar10/multi_sync/4096/fedavg --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

（可选）如果启动了服务器端流量监控

通过 Ctrl-c 来中断，可以自动从目录中获取 `grpc_port_traffic.summary.json`。其中 `received_bytes` / `sent_bytes` 是抓包口径下的端口总流量，`received_tcp_payload_bytes` / `sent_tcp_payload_bytes` 是更接近应用层承载数据量的 TCP payload 统计；它们适合做网络侧旁证，不建议直接替代框架内 `parameter_*` 统计来计算通信压缩比。

执行 DLG：

```bash
python -m fedlab.tools.replay_saved_update_dlg   ../outputs/exp/cifar10/multi_sync/4096/fedavg   --output-dir ../outputs/exp/cifar10/multi_sync/4096/fedavg/offline_attack_replay_dlg --override data.split_dir=../data/attack/cifar10/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

执行 iDLG：

```bash
python -m fedlab.tools.replay_saved_update_idlg   ../outputs/exp/cifar10/multi_sync/4096/fedavg   --output-dir ../outputs/exp/cifar10/multi_sync/4096/fedavg/offline_attack_replay_idlg --override data.split_dir=../data/attack/cifar10/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

测试：

```bash
python -m fedlab.tools.replay_saved_model_evaluation ../outputs/exp/cifar10/multi_sync/4096/fedavg/model.pt --config ../outputs/exp/cifar10/multi_sync/4096/fedavg/config.yaml --data-dir ../data/test/cifar10/test --output-dir ../outputs/exp/cifar10/multi_sync/4096/fedavg/offline_test_replay --override runtime.device=cpu
```

##### Topk

（可选）启动服务器端流量监控

```bash
cd workspace/src
bash scripts/monitor_tcp_port_traffic.sh \
  --port 50072 \
  --interface any \
  --output-dir ../outputs/exp/cifar10/multi_sync/4096/topk/traffic
```

Top-k 服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/cifar10/topk.yaml   --override grpc.address=0.0.0.0:50072   --override grpc.server_address=127.0.0.1:50072   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/cifar10/multi_sync/4096/topk --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

Top-k 客户端 `c1` / `c2` / `c3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c1   --config configs/cifar10/topk.yaml   --override grpc.server_address=127.0.0.1:50072   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/cifar10/multi_sync/4096/topk --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c2   --config configs/cifar10/topk.yaml   --override grpc.server_address=127.0.0.1:50072   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/cifar10/multi_sync/4096/topk --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c3   --config configs/cifar10/topk.yaml   --override grpc.server_address=127.0.0.1:50072   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/cifar10/multi_sync/4096/topk --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

（可选）如果启动了服务器端流量监控

通过 Ctrl-c 来中断，可以自动从目录中获取 `grpc_port_traffic.summary.json`。其中 `received_bytes` / `sent_bytes` 是抓包口径下的端口总流量，`received_tcp_payload_bytes` / `sent_tcp_payload_bytes` 是更接近应用层承载数据量的 TCP payload 统计；它们适合做网络侧旁证，不建议直接替代框架内 `parameter_*` 统计来计算通信压缩比。

执行 DLG：

```bash
python -m fedlab.tools.replay_saved_update_dlg   ../outputs/exp/cifar10/multi_sync/4096/topk   --output-dir ../outputs/exp/cifar10/multi_sync/4096/topk/offline_attack_replay_dlg --override data.split_dir=../data/attack/cifar10/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

执行 iDLG：

```bash
python -m fedlab.tools.replay_saved_update_idlg   ../outputs/exp/cifar10/multi_sync/4096/topk   --output-dir ../outputs/exp/cifar10/multi_sync/4096/topk/offline_attack_replay_idlg --override data.split_dir=../data/attack/cifar10/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

测试：

```bash
python -m fedlab.tools.replay_saved_model_evaluation ../outputs/exp/cifar10/multi_sync/4096/topk/model.pt --config ../outputs/exp/cifar10/multi_sync/4096/topk/config.yaml --data-dir ../data/test/cifar10/test --output-dir ../outputs/exp/cifar10/multi_sync/4096/topk/offline_test_replay --override runtime.device=cpu
```

##### EGA

（可选）启动服务器端流量监控

```bash
cd workspace/src
bash scripts/monitor_tcp_port_traffic.sh \
  --port 50073 \
  --interface any \
  --output-dir ../outputs/exp/cifar10/multi_sync/4096/ega/traffic
```

EGA 服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/cifar10/ega.yaml   --override grpc.address=0.0.0.0:50073   --override grpc.server_address=127.0.0.1:50073   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/cifar10/multi_sync/4096/ega --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

EGA 客户端 `c1` / `c2` / `c3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c1   --config configs/cifar10/ega.yaml   --override grpc.server_address=127.0.0.1:50073   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/cifar10/multi_sync/4096/ega --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c2   --config configs/cifar10/ega.yaml   --override grpc.server_address=127.0.0.1:50073   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/cifar10/multi_sync/4096/ega --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c3   --config configs/cifar10/ega.yaml   --override grpc.server_address=127.0.0.1:50073   --override runtime.device=cpu --override experiment.output_dir=../outputs/exp/cifar10/multi_sync/4096/ega --override runtime.seed=4096 --override grpc.poll_seconds=10.0
```

（可选）如果启动了服务器端流量监控

通过 Ctrl-c 来中断，可以自动从目录中获取 `grpc_port_traffic.summary.json`。其中 `received_bytes` / `sent_bytes` 是抓包口径下的端口总流量，`received_tcp_payload_bytes` / `sent_tcp_payload_bytes` 是更接近应用层承载数据量的 TCP payload 统计；它们适合做网络侧旁证，不建议直接替代框架内 `parameter_*` 统计来计算通信压缩比。


执行 DLG：

```bash
python -m fedlab.tools.replay_saved_update_dlg   ../outputs/exp/cifar10/multi_sync/4096/ega   --output-dir ../outputs/exp/cifar10/multi_sync/4096/ega/offline_attack_replay_dlg --override data.split_dir=../data/attack/cifar10/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

执行 iDLG：

```bash
python -m fedlab.tools.replay_saved_update_idlg   ../outputs/exp/cifar10/multi_sync/4096/ega   --output-dir ../outputs/exp/cifar10/multi_sync/4096/ega/offline_attack_replay_idlg --override data.split_dir=../data/attack/cifar10/attack --override attack.max_samples=512 --override attack.max_samples_cap=512 --override attack.steps=500 --override attack.seed=4096 --override attack.device=cpu
```

测试：

```bash
python -m fedlab.tools.replay_saved_model_evaluation ../outputs/exp/cifar10/multi_sync/4096/ega/model.pt --config ../outputs/exp/cifar10/multi_sync/4096/ega/config.yaml --data-dir ../data/test/cifar10/test --output-dir ../outputs/exp/cifar10/multi_sync/4096/ega/offline_test_replay --override runtime.device=cpu
```

## 5. 训练完成后的攻击回放与测试回放

### 5.1 训练输出里哪些文件最重要

一次实验运行结束后，常见关键产物包括：

- `summary.json`
- `metrics.json`
- `model.pt`
- `config.yaml`
- `saved_updates/`

其中：

- `saved_updates/` 用于离线重放攻击
- `model.pt` 用于离线重放测试

当前框架本身不在联邦训练过程中执行在线攻击；攻击统一通过离线重放脚本独立完成。

### 5.2 回放全部已配置攻击

`replay_saved_update_attacks.py` 会读取某个实验目录下的 `saved_updates/`，并按当前配置重新执行已启用攻击。

注意：离线攻击回放不会再从训练输出里读取参考样本模板，而是会根据 `config.yaml` 里的数据配置重新加载客户端本地训练集。因此做离线攻击时，攻击机仍然必须能访问被攻击客户端的训练集；如果你已经用第 3.5.4 节的 `prepare_role_datasets` 生成了角色目录，推荐直接把 `role_datasets/<task>/attack/` 作为新的 `data.split_dir` 传入。

#### 5.2.1 离线攻击阶段的最小数据需求

如果只讨论当前这套离线攻击实现，最小需要的数据是：

- 攻击输出目录本身：至少需要 `saved_updates/` 和对应的 `config.yaml`。
- 被攻击目标客户端的本地训练集：因为当前离线攻击实现会重新加载这部分数据，用于重建攻击输入形状，并在攻击完成后做一对一匹配与成功率统计。

当前实现下，不需要的数据包括：

- 被攻击客户端的本地验证集
- 被攻击客户端的本地测试集
- 其他未被攻击客户端的训练集、验证集、测试集
- 服务端的全局验证集、全局测试集

更具体地说：

- 如果你只回放一个目标客户端的攻击，那么最少只需要那个客户端自己的训练集。
- 如果你回放一个实验目录里所有客户端的攻击，那么需要这些被攻击客户端各自的训练集都可访问。
- 如果只是想让攻击优化过程跑起来，从理论上讲只需要模型状态和目标更新；但当前脚本还会做恢复样本匹配与指标统计，所以实际运行时仍要求目标客户端训练集可访问。

因此，当前离线攻击阶段的最小可行数据集合通常是：

- 攻击机：`saved_updates/` + `config.yaml`
- 目标客户端 `i`：自己的 `train`

而不是“所有客户端的 train/val/test 全都放在攻击机上”。

```bash
cd workspace/src
python -m fedlab.tools.replay_saved_update_attacks   ../outputs/rare_fedavg_grpc   --output-dir ../outputs/rare_fedavg_grpc/offline_attack_replay   --override data.split_dir=../data/role_datasets/rare/attack
```

如果想临时修改攻击参数，可以继续追加 `--override`：

```bash
python -m fedlab.tools.replay_saved_update_attacks   ../outputs/rare_fedavg_grpc   --output-dir ../outputs/rare_fedavg_grpc/offline_attack_replay   --override data.split_dir=../data/role_datasets/rare/attack   --override attack.steps=300   --override attack.lr=0.05   --override attack.max_samples=8
```

如果你还想同时固定攻击设备和攻击随机种子，可以继续追加：

```bash
python -m fedlab.tools.replay_saved_update_attacks   ../outputs/rare_fedavg_grpc   --output-dir ../outputs/rare_fedavg_grpc/offline_attack_replay   --override data.split_dir=../data/role_datasets/rare/attack   --override attack.device=cuda:1   --override attack.seed=1234   --override attack.steps=300
```

### 5.3 只回放 DLG 或 iDLG

如果只想评估单一攻击方法，可以使用专门的入口；它们仍然会按当前配置重新加载客户端本地参考数据。

只回放 DLG：

```bash
python -m fedlab.tools.replay_saved_update_dlg   ../outputs/rare_fedavg_grpc   --output-dir ../outputs/rare_fedavg_grpc/offline_attack_replay_dlg --override data.split_dir=../data/role_datasets/rare/attack --override attack.max_samples=512 --override attack.steps=500
```

只回放 iDLG：

```bash
python -m fedlab.tools.replay_saved_update_idlg   ../outputs/rare_fedavg_grpc   --output-dir ../outputs/rare_fedavg_grpc/offline_attack_replay_idlg --override data.split_dir=../data/role_datasets/rare/attack --override attack.max_samples=512 --override attack.steps=500
```

如果你只想回放单一攻击方法，可以直接使用上面的 `replay_saved_update_dlg` 或 `replay_saved_update_idlg` 入口；README 这里不再展开批量脚本调用方式。

### 5.4 对保存好的模型重新做测试回放

离线测试回放使用 `model.pt` 和对应实验配置重新构建数据加载器与模型，然后在共享测试集上评测。

对于 `rare`，训练输出目录下现在会额外保存 `evaluation_context.json`，里面记录了每个客户端测试时需要的 scaler 统计。因此离线测试时，只要：

- 有 `model.pt`、`config.yaml`、`evaluation_context.json`
- 并且新的数据目录里仍然能提供各客户端 `test.csv`

就可以脱离训练集独立测试。

最常用命令：

```bash
cd workspace/src
python -m fedlab.tools.replay_saved_model_evaluation   ../outputs/rare_fedavg_grpc/model.pt   --config ../outputs/rare_fedavg_grpc/config.yaml   --data-dir ../data/rare_earth_test_only   --output-dir ../outputs/rare_fedavg_grpc/offline_test_replay
```

如果你显式把 `evaluation_context.json` 放到了别的位置，也可以继续传：

```bash
python -m fedlab.tools.replay_saved_model_evaluation   ../outputs/rare_fedavg_grpc/model.pt   --config ../outputs/rare_fedavg_grpc/config.yaml   --data-dir ../data/rare_earth_test_only   --evaluation-context ../outputs/rare_fedavg_grpc/evaluation_context.json   --output-dir ../outputs/rare_fedavg_grpc/offline_test_replay
```

对 `mnist` / `cifar10`，离线测试仍然按照原来的预切分目录读取；如果想换一套数据目录重新测试，可以覆盖 `data.split_dir`，或者直接传 `--data-dir`：

```bash
python -m fedlab.tools.replay_saved_model_evaluation   ../outputs/mnist_fedavg_grpc/model.pt   --config ../outputs/mnist_fedavg_grpc/config.yaml   --data-dir ../data/mnist   --output-dir ../outputs/mnist_fedavg_grpc/offline_test_replay
```

离线测试回放的结果通常会写到：

- `test_metrics.json`
- `test_summary.json`

## 6. 结果分析 

### 6.1 直接使用 `analyze_experiment_suite`

如果你已经把多次实验结果整理在同一层级目录下，推荐把分析结果输出到 `../outputs/analysis/...`：

```bash
cd workspace/src
PYTHONPATH=. python -m fedlab.tools.analyze_experiment_suite   ../outputs/exp/rare/multi_sync/4096   --loss mse   --output-dir ../outputs/analysis/rare/multi_sync/4096   --algorithms centralized fedavg topk ega
```

如果你想联合分析多次不同 seed 或不同运行目录，也可以一次传多个根目录：

```bash
PYTHONPATH=. python -m fedlab.tools.analyze_experiment_suite   ../outputs/exp/rare/multi_sync/4096   ../outputs/exp/rare/multi_sync/2026   ../outputs/exp/rare/multi_sync/42   --loss mse   --output-dir ../outputs/analysis/rare/multi_sync/   --algorithms centralized fedavg topk ega
```

当前工具支持：

- `rare`：`mse` / `mae`
- `mnist`：`cross_entropy`
- `cifar10`：`cross_entropy`

分析结果里通常包括：

- 汇总 CSV / Markdown
- 验证集损失随轮数变化图
- 验证集损失随累计上传量变化图
- best-so-far 验证集损失图
- 测试指标对比图
- 相对 `centralized` 的性能损失统计

### 6.2 参数速查表

下面这部分按“用户真正会传入或覆盖的参数”整理。为了避免 README 失控，这里优先列出训练入口和离线攻击入口实际会读取的参数；如果你要看更完整的配置块说明，再查 `docs/config_reference.md`。

#### 6.2.1 训练入口常用 `--override`

这些参数可直接传给 `python -m fedlab.entrypoints.train/server/client`。

| 参数 | 取值范围 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `experiment.output_dir` | 合法路径 | 无固定默认输出目录 | 当前实验输出目录。 |
| `runtime.device` | `cpu`、`cuda:*` | `cpu` | 主训练设备。 |
| `runtime.seed` | 任意整数 | 未显式设置时为空 | 训练随机种子。 |
| `runtime.deterministic` | `true`、`false` | `true` | 是否尽量启用确定性运行。 |
| `training.epochs` | 正整数 | `1` | centralized 模式下表示总训练 epoch；federated 模式下表示每轮本地训练 epoch。 |
| `training.lr` | 正浮点数 | `0.001` | 训练学习率。 |
| `training.loss` | `mse`、`mae`、`smooth_l1`、`huber`、`cross_entropy`、`task_default` | `mse` | 训练损失。分类任务通常改成 `cross_entropy`。 |
| `training.optimizer` | `adam`、`adamw`、`sgd` | `adam` | 训练优化器。 |
| `training.weight_decay` | 非负浮点数 | `0.0` | 权重衰减。 |
| `training.patience` | 非负整数 | `50` | 早停耐心轮数。 |
| `training.min_delta` | 非负浮点数 | `0.0` | 早停最小改进阈值。 |
| `evaluation.metrics` | 任务支持的指标列表 | forecasting 默认 `[mse,mae,mape]`；分类配置通常会覆盖 | 验证和测试时计算哪些指标。 |
| `federated.algorithm` | `fedavg`、`sparse_fedavg`、`randomk_fedavg`、`qsgd_fedavg`、`secure_quantized_fedavg`、`adaptive_clipped_rdp_fedavg`、`ega_fedavg` | `fedavg` | 联邦聚合算法。常用的 Top-k 配置通常对应 `sparse_fedavg`。 |
| `federated.rounds` | 正整数 | `20` | 联邦通信轮数。 |
| `federated.local_steps` | 正整数 | 未设置 | 如果配置，会优先于 `training.epochs`。 |
| `federated.topk_fraction` | `(0,1]` 浮点数 | 稀疏方法未显式设置时通常走各任务 YAML；Top-k 示例配置默认 `0.10` | Top-k / Random-k 保留比例。 |
| `federated.qsgd_levels` | 正整数 | QSGD 方法内部缺省 `127` | QSGD 量化级数。 |
| `federated.quantization_dtype` | 常用 `float16`、`qint8` | 依配置文件 | 安全量化方法的量化类型。 |
| `federated.quantization_seed` | 任意整数 | 未显式设置 | 量化相关随机种子。 |
| `grpc.address` | `host:port` | 运行时默认 `0.0.0.0:50051` | 服务端监听地址。 |
| `grpc.server_address` | `host:port` | 运行时默认 `127.0.0.1:50051` | 客户端连接地址。 |
| `grpc.poll_seconds` | 正浮点数 | 运行时默认 `1.0`；公共 gRPC 配置文件里是 `5.0` | 客户端与服务端的轮询休眠间隔。 |
| `grpc.max_message_mb` | 正浮点数 | `384.0` | gRPC 最大消息大小。 |
| `replay_capture.enabled` | `true`、`false` | `true` | 是否保存可用于离线攻击的 `saved_updates`。 |
| `replay_capture.frequency_rounds` | 正整数 | `30` | 第 0 轮、最后一轮以及每隔多少轮额外保存一次更新。 |
| `tracking.enabled` | `true`、`false` | 运行时默认 `true`；分类 smoke-test 配置常显式设为 `false` | 是否启用 `wandb`。 |
| `tracking.offline` | `true`、`false` | `true` | `wandb` 是否走离线模式。 |
| `tracking.project` | 任意字符串 | `federated-rare-earth` | `wandb` project 名。 |
| `artifacts.config_formats` | `yaml`、`json`、`toml` 组成的列表 | `[yaml]` | 启动时保存哪些格式的配置快照。 |
| `artifacts.save_every_rounds` | 非负整数 | `0` | 是否额外保存按轮快照，`0` 表示关闭。 |

#### 6.2.2 EGA 专用参数

| 参数 | 取值范围 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `ega.artifact_path` | 合法路径 | `artifacts/ega/ega_h240_v1.pt` | EGA codec 文件路径。 |
| `ega.num_clients` | 正整数、`auto` | `auto` | codec 初始化时使用的客户端数。 |
| `ega.block_size` | 正整数 | `256` | 编码块大小。 |
| `ega.encoded_dim` | 正整数 | `128` | 编码后维度。 |
| `ega.hidden_dim` | 正整数 | `2048` | codec 隐层宽度。 |
| `ega.residual_blocks` | 非负整数 | `4` | 残差块数量。 |
| `ega.quantization_level` | 正整数 | `159` | 编码量化级数。 |
| `ega.encode_buffers` | `true`、`false` | `false` | 是否把 buffer 也纳入编码通信。 |
| `ega.buffer_tolerance` | 非负浮点数 | `0.0` | buffer 变化容忍阈值。 |
| `ega.normalization` | 正浮点数 | `0.00025` | 编码归一化尺度。 |
| `ega.initial_normalization` | 正浮点数 | `0.00025` | 初始归一化尺度。 |
| `ega.min_normalization` | 正浮点数 | `1e-6` | 归一化下界。 |
| `ega.normalization_strategy` | 当前常用 `ema_reported_client_max_abs` | `ema_reported_client_max_abs` | 归一化更新策略。 |
| `ega.normalization_ema` | `[0,1]` 浮点数 | `0.98` | 归一化 EMA 系数。 |
| `ega.encoded_dtype` | 常用 `int8` | `int8` | 编码后存储类型。 |
| `ega.encoded_stochastic_rounding` | `true`、`false` | `false` | 编码量化是否随机舍入。 |
| `ega.encoded_noise_std` | 非负浮点数 | `0.0` | 编码噪声强度。 |
| `ega.error_feedback` | `true`、`false` | `true` | 是否使用误差反馈。 |
| `ega.pretrain.epochs` | 正整数 | `220` | codec 预训练轮数。 |
| `ega.pretrain.patience` | 非负整数 | `50` | codec 预训练早停耐心。 |
| `ega.pretrain.batch_size` | 正整数 | `128` | codec 预训练 batch size。 |
| `ega.pretrain.lr` | 正浮点数 | `0.0002` | codec 预训练学习率。 |
| `ega.pretrain.train_groups` | 正整数 | `50000` | codec 预训练训练组数。 |
| `ega.pretrain.val_groups` | 正整数 | `25000` | codec 预训练验证组数。 |
| `ega.pretrain.seed` | 任意整数 | `4096` | codec 预训练随机种子。 |
| `ega.pretrain.device` | `same`、`cpu`、`cuda:*` | `same` | codec 预训练设备。 |

#### 6.2.3 离线攻击回放参数

这些参数由 `python -m fedlab.tools.replay_saved_update_attacks`、`replay_saved_update_dlg`、`replay_saved_update_idlg` 读取，不参与联邦训练本身。

| 参数 | 取值范围 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `attack.target_type` | 当前仅支持 `update_payload` | `update_payload` | 指定攻击对象类型。 |
| `attack.methods` | `dlg`、`idlg`、列表 | `[dlg,idlg]` | 选择回放哪些攻击方法。 |
| `attack.frequency_rounds` | 正整数 | `30` | 统计上每隔多少轮挑选一次攻击轮；真正可回放轮次仍取决于训练阶段保存了哪些 `saved_updates`。 |
| `attack.max_samples` | `auto` 或正整数 | `auto` | 每次联合恢复多少样本。 |
| `attack.max_samples_cap` | 正整数 | `128` | `attack.max_samples=auto` 时的上限。 |
| `attack.steps` | 正整数 | `300` | 每次攻击优化步数。 |
| `attack.lr` | 正浮点数 | `0.001` | 攻击优化学习率。 |
| `attack.optimizer` | 当前仅 `adam` | `adam` | 攻击优化器。 |
| `attack.restarts` | 正整数 | `1` | 随机重启次数。 |
| `attack.client_selection` | `all`、`first`、`round_robin` | `all` | 一轮内选哪些客户端发起攻击。 |
| `attack.clients_per_round` | 正整数 | 配置默认 `3` | 每轮攻击多少个客户端。 |
| `attack.input_clip` | 正浮点数 | `5.0` | 对恢复输入的裁剪上界。 |
| `attack.target_clip` | 正浮点数 | `5.0` | 对恢复标签或目标张量的裁剪上界。 |
| `attack.tv_weight` | 非负浮点数 | `0.0` | 总变差正则权重。 |
| `attack.seed` | 任意整数 | 配置默认 `4096` | 攻击随机种子。 |
| `attack.model_mode` | `train`、`eval` | 配置默认 `eval` | 攻击时模型工作模式。 |
| `attack.local_optimizer` | `adam`、`sgd` | `adam` | 一步局部更新近似使用的优化器。 |
| `attack.local_lr` | 正浮点数 | `0.001` | 一步局部更新近似学习率。 |
| `attack.local_optimizer_eps` | 正浮点数 | `1e-8` | `adam` 近似的 `eps`。 |
| `attack.recovery_match_metric` | `mse`、`psnr`、`ssim` | `mse` | 恢复后做一对一匹配时使用的指标。 |
| `attack.recovery_success_metric` | `mse`、`psnr`、`ssim` | `mse` | 判定恢复成功时使用的指标。 |
| `attack.recovery_success_threshold` | `null` 或浮点数 | `null` | 恢复成功阈值；为空时按指标自动推导。 |
| `attack.success_rate_threshold` | `[0,1]` 浮点数 | `0.05` | 单次攻击成功率阈值。 |
| `attack.overall_success_rate_threshold` | `[0,1]` 浮点数 | `0.05` | 汇总层面成功率阈值。 |
| `attack.data_range` | 正浮点数 | `1.0` | PSNR / SSIM 使用的数据范围。 |
| `attack.async_enabled` | `true`、`false` | `false` | 是否异步执行攻击。 |
| `attack.async_workers` | 正整数 | `1` | 异步攻击 worker 数。 |
| `attack.async_max_pending_rounds` | 正整数 | `5` | 异步模式下允许积压的最大轮数。 |
| `attack.device` | `same`、`cpu`、`cuda:*` | `same` | 攻击设备。 |

### 6.3 `summary.json` 指标说明

训练结束后的 `summary.json` 主要由联邦训练摘要生成逻辑写出；如果后续执行离线攻击回放，攻击工具还会在其自己的输出目录里补充攻击相关字段。

#### 6.3.1 联邦训练摘要字段

| 指标 | 取值范围 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `test` | 指标字典或 `null` | 无 | 最终测试集指标；如果没有提供测试集，则为 `null`。 |
| `protocol_test` | 指标字典或 `null` | 默认与 `test` 相同 | 协议语义下的测试指标；如果没有执行最终测试，则为 `null`。 |
| `final_test_executed` | `true`、`false` | `true` | 是否在训练结束后执行了最终测试。 |
| `rounds` | 非负整数 | `0` | 实际完成的联邦轮数。 |
| `total_time_seconds` | 非负浮点数 | `0.0` | 整个训练总耗时。 |
| `best_round` | 非负整数 | 无 | 最佳验证指标所在轮次。 |
| `test_checkpoint` | 字符串或 `null` | `best_validation` | 说明测试使用的是哪一份 checkpoint；如果没有执行最终测试，则为 `null`。 |
| `best_val` | 指标字典 | 无 | 最佳验证轮上的完整验证指标。 |
| `best_val_metric_name` | 字符串 | 由任务决定 | 用于早停和选最佳 checkpoint 的主指标名。 |
| `best_val_metric_value` | 浮点数 | 无 | 主验证指标在最佳轮的数值。 |
| `best_val_<metric>` | 浮点数 | 按任务生成 | 每个验证指标在最佳轮的数值展开形式，例如 `best_val_mse`。 |
| `last_parameter_upload_bytes` | 非负整数 | `0` | 最后一轮按“模型参数语义”统计的上传字节数。 |
| `last_parameter_download_bytes` | 非负整数 | `0` | 最后一轮按“模型参数语义”统计的下载字节数。 |
| `last_parameter_total_bytes` | 非负整数 | `0` | 最后一轮参数级总通信量。 |
| `last_transport_upload_bytes` | 非负整数 | `0` | 最后一轮按实际序列化传输统计的上传字节数。 |
| `last_transport_download_bytes` | 非负整数 | `0` | 最后一轮按实际序列化传输统计的下载字节数。 |
| `last_transport_total_bytes` | 非负整数 | `0` | 最后一轮实际传输总字节数。 |
| `last_transport_upload_overhead_bytes` | 非负整数 | `0` | 最后一轮上传额外开销字节数，即实际传输减参数语义字节。 |
| `last_transport_download_overhead_bytes` | 非负整数 | `0` | 最后一轮下载额外开销字节数。 |
| `last_parameter_download_compression_ratio` | 非负浮点数 | `0.0` | 最后一轮下载相对 dense baseline 的压缩率。 |
| `last_parameter_upload_compression_ratio` | 非负浮点数 | `0.0` | 最后一轮上传相对 dense baseline 的压缩率。 |
| `last_parameter_total_communication_ratio` | 非负浮点数 | `0.0` | 最后一轮参数语义总通信压缩率。 |
| `last_transport_download_compression_ratio` | 非负浮点数 | `0.0` | 最后一轮实际下载压缩率。 |
| `last_transport_upload_compression_ratio` | 非负浮点数 | `0.0` | 最后一轮实际上传压缩率。 |
| `last_transport_total_communication_ratio` | 非负浮点数 | `0.0` | 最后一轮实际总通信压缩率。 |
| `total_parameter_upload_bytes` | 非负整数 | `0` | 全部轮次累计参数语义上传字节数。 |
| `total_parameter_download_bytes` | 非负整数 | `0` | 全部轮次累计参数语义下载字节数。 |
| `total_parameter_bytes` | 非负整数 | `0` | 全部轮次累计参数语义总通信量。 |
| `total_transport_upload_bytes` | 非负整数 | `0` | 全部轮次累计实际上传字节数。 |
| `total_transport_download_bytes` | 非负整数 | `0` | 全部轮次累计实际下载字节数。 |
| `total_transport_bytes` | 非负整数 | `0` | 全部轮次累计实际总通信量。 |
| `total_transport_upload_overhead_bytes` | 非负整数 | `0` | 全部轮次累计实际上传额外开销。 |
| `total_transport_download_overhead_bytes` | 非负整数 | `0` | 全部轮次累计实际下载额外开销。 |
| `privacy_accountant` | `null` 或字符串 | `null` | 当前隐私 accountant 名称；普通 FedAvg 常为空。 |
| `privacy_epsilon` | `null` 或浮点数 | `null` | DP 隐私预算 epsilon。 |
| `privacy_delta` | `null` 或浮点数 | `null` | DP 隐私预算 delta。 |
| `privacy_rdp_alpha` | `null` 或浮点数 | `null` | RDP accountant 的 alpha。 |
| `privacy_rdp_total` | `null` 或浮点数 | `null` | 累计 RDP 量。 |
| `privacy_sampling_rate` | `null` 或浮点数 | `null` | 隐私会计使用的采样率。 |
| `adaptive_clip_norm` | `null` 或浮点数 | `null` | 自适应裁剪当前阈值。 |
| `adaptive_clip_median_norm` | `null` 或浮点数 | `null` | 自适应裁剪估计的中位范数。 |
| `adaptive_reference_clip_norm` | `null` 或浮点数 | `null` | 自适应裁剪参考阈值。 |
| `adaptive_noise_std` | `null` 或浮点数 | `null` | 自适应 DP 噪声标准差。 |
| `privacy_trust_model` | `null` 或字符串 | `null` | 当前隐私信任模型描述。 |
| `transport` | `null`、`inprocess`、`grpc` 等 | 视运行模式而定 | 本次运行使用的传输后端。 |

#### 6.3.2 离线攻击回放追加字段

这些字段通常写在攻击输出目录自己的 `summary.json` 里，而不是训练目录原始的 `summary.json`。

| 指标 | 取值范围 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `attack_target_type` | 当前为 `update_payload` | 若攻击汇总缺失则回退到已有值或 `update_payload` | 攻击目标类型。 |
| `attack_primary_metric_name` | `mse`、`psnr`、`ssim` 等 | 无 | 汇总攻击效果时使用的主指标名。 |
| `attack_primary_metric_direction` | `min`、`max` | 无 | 主指标是越小越好还是越大越好。 |
| `attack_overall_avg_primary_metric_value` | 浮点数 | 无 | 所有攻击样本整体平均主指标。 |
| `attack_overall_best_primary_metric_value` | 浮点数 | 无 | 所有攻击记录里最优主指标。 |
| `attack_success_rate` | `[0,1]` 浮点数 | 无 | 攻击总体成功率。 |
| `attack_evaluations` | 非负整数 | `0` | 本次离线回放共评估了多少条攻击记录。 |
| `attack_summary` | 字典 | 无 | 更完整的攻击统计汇总原文。 |

## 7. 进一步阅读

更详细的中文说明见：

- `docs/框架使用手册.md`
- `docs/config_reference.md`
