# Federated Rare-Earth and Image Classification Framework

本仓库同时支持三类任务：

- `rare`：三客户端稀土价格时间序列预测
- `mnist`：三客户端图像分类
- `cifar10`：三客户端图像分类

如果你按本文档建议的 `workspace/data + workspace/src` 目录组织工程，仓库默认配置里的相对路径可以直接工作，无需额外修改 `data.split_dir`。

## 1. 工作区组织

推荐先创建一个独立工作区，例如 `workspace/`，然后把原始数据和代码分别放到 `data/` 与 `src/`：

```bash
mkdir -p workspace/data
cd workspace
git clone <your-repo-url> src
```

推荐目录结构如下：

```text
workspace/
├── data/
│   ├── raw_data/                  # rare 的 rawdata2 Excel 原始文件，按客户端分目录存放
│   ├── raw_image_datasets/        # MNIST / CIFAR-10 原始下载目录
│   ├── rare_earth_rawdata2/       # rare 预处理后的联邦数据目录
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

### 2.2 快速检查安装是否成功

```bash
cd workspace/src
python -m fedlab.entrypoints.train --config configs/rare/fedavg.yaml --help
python -m fedlab.tools.prepare_image_classification_data --help
python -m fedlab.tools.analyze_experiment_suite --help
```

## 3. 数据准备

### 3.1 框架最终期望的联邦数据布局

框架不会在训练时动态切分数据，而是直接读取 `data.split_dir` 指向的预切分目录。

`rare` 默认读取：

```yaml
data:
  split_dir: ../data/rare_earth_rawdata2
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

请把原始 Excel 按客户端分目录放到：

```text
workspace/data/raw_data/
├── Nd2O3/
│   ├── *.xls
│   └── *.xlsx
├── CeO2/
│   ├── *.xls
│   └── *.xlsx
└── La2O3/
    ├── *.xls
    └── *.xlsx
```

也支持使用中文目录名：

```text
workspace/data/raw_data/
├── 氧化钕/
├── 氧化铈/
└── 氧化镧/
```

脚本会递归扫描 `raw_data/` 下的所有 `.xls` / `.xlsx` 文件，并优先根据所在子目录推断它属于哪个客户端；同时也会读取文件内容里的 `品名` 列做一致性检查。

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

如果同一个客户端目录下有多个 Excel，预处理脚本会先把它们按日期合并；若日期重复，会对同一天的数值取平均。之后再构造三客户端的日级插值宽表，并按时间顺序切分成 `train/val/test`。

生成联邦训练数据：

```bash
cd workspace/src
python -m fedlab.tools.prepare_rawdata2 \
  --raw-dir ../data/raw_data \
  --output-dir ../data/rare_earth_rawdata2
```

### 3.3 `mnist` / `cifar10` 原始数据如何放置

图像分类数据推荐直接让框架自动下载；也可以手动 `wget` 到 `workspace/data/raw_image_datasets/`。

#### 推荐方式：让框架自动下载并切分

```bash
cd workspace/src
python -m fedlab.tools.prepare_image_classification_data   --dataset all   --raw-root ../data/raw_image_datasets   --output-root ../data   --num-clients 3   --val-ratio 0.1   --seed 2026
```

只准备 MNIST：

```bash
python -m fedlab.tools.prepare_image_classification_data   --dataset mnist   --raw-root ../data/raw_image_datasets   --output-root ../data
```

只准备 CIFAR-10：

```bash
python -m fedlab.tools.prepare_image_classification_data   --dataset cifar10   --raw-root ../data/raw_image_datasets   --output-root ../data
```

#### 可选方式：手动 `wget` 原始数据

MNIST 原始文件可放到 `workspace/data/raw_image_datasets/mnist/MNIST/raw/`：

```bash
mkdir -p ../data/raw_image_datasets/mnist/MNIST/raw
wget -O ../data/raw_image_datasets/mnist/MNIST/raw/train-images-idx3-ubyte.gz   https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
wget -O ../data/raw_image_datasets/mnist/MNIST/raw/train-labels-idx1-ubyte.gz   https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
wget -O ../data/raw_image_datasets/mnist/MNIST/raw/t10k-images-idx3-ubyte.gz   https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
wget -O ../data/raw_image_datasets/mnist/MNIST/raw/t10k-labels-idx1-ubyte.gz   https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
```

CIFAR-10 原始文件可放到 `workspace/data/raw_image_datasets/cifar10/`：

```bash
mkdir -p ../data/raw_image_datasets/cifar10
wget -O ../data/raw_image_datasets/cifar10/cifar-10-python.tar.gz   https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf ../data/raw_image_datasets/cifar10/cifar-10-python.tar.gz   -C ../data/raw_image_datasets/cifar10
```

手动下载后，仍然使用 `prepare_image_classification_data` 生成联邦训练数据。

### 3.4 预处理完成后的目录结构

`rare` 预处理完成后通常会得到：

```text
../data/rare_earth_rawdata2/
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

## 4. 训练方式

### 4.1 单机直接训练

集中式训练：

```bash
cd workspace/src
python -m fedlab.entrypoints.train   --config configs/rare/centralized.yaml   --mode centralized
```

单机顺序联邦训练：

```bash
python -m fedlab.entrypoints.train   --config configs/rare/fedavg.yaml   --mode federated
```

你也可以通过 `--override` 覆盖常见参数：

```bash
python -m fedlab.entrypoints.train   --config configs/mnist/ega.yaml   --mode federated   --override experiment.output_dir=../outputs/mnist_ega_debug   --override federated.rounds=20   --override training.epochs=1   --override runtime.device=cuda:0
```

### 4.2 多进程 / 多节点 gRPC 联邦训练

#### 通用说明

多节点运行时：

- 服务端使用 `python -m fedlab.entrypoints.server`
- 客户端使用 `python -m fedlab.entrypoints.client`
- 三端需要使用语义一致的配置
- 服务端 `grpc.address` 需要监听一个实际可访问的地址
- 客户端 `grpc.server_address` 需要指向服务器的实际 `IP:PORT`
- 客户端 `client-id` 必须与 `data.clients` 一致
- 下面示例里的默认地址使用 `127.0.0.1`，表示服务端和客户端都在本机运行；如果跨机器部署，请把它替换成服务器的实际 IP，并根据需要调整端口

推荐把 `experiment.output_dir` 设到 `../outputs/...`，这样实验产物会落在 `src` 外层，避免污染代码目录。客户端会在本地额外写日志到：

```text
../outputs/<run_name>/client_<client_id>/
```

#### 4.2.1 `rare` 的 FedAvg

服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/rare/fedavg.yaml   --override grpc.address=0.0.0.0:50051   --override grpc.server_address=127.0.0.1:50051   --override experiment.output_dir=../outputs/rare_fedavg_grpc   --override runtime.device=cuda:0
```

客户端 `Nd2O3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id Nd2O3   --config configs/rare/fedavg.yaml   --override grpc.server_address=127.0.0.1:50051   --override experiment.output_dir=../outputs/rare_fedavg_grpc   --override runtime.device=cuda:0
```

客户端 `CeO2`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id CeO2   --config configs/rare/fedavg.yaml   --override grpc.server_address=127.0.0.1:50051   --override experiment.output_dir=../outputs/rare_fedavg_grpc   --override runtime.device=cuda:0
```

客户端 `La2O3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id La2O3   --config configs/rare/fedavg.yaml   --override grpc.server_address=127.0.0.1:50051   --override experiment.output_dir=../outputs/rare_fedavg_grpc   --override runtime.device=cuda:0
```

#### 4.2.2 `rare` 的 Top-k

服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/rare/topk.yaml   --override grpc.address=0.0.0.0:50052   --override grpc.server_address=127.0.0.1:50052   --override experiment.output_dir=../outputs/rare_topk_grpc   --override runtime.device=cuda:0
```

客户端 `Nd2O3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id Nd2O3   --config configs/rare/topk.yaml   --override grpc.server_address=127.0.0.1:50052   --override experiment.output_dir=../outputs/rare_topk_grpc   --override runtime.device=cuda:0
```

客户端 `CeO2`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id CeO2   --config configs/rare/topk.yaml   --override grpc.server_address=127.0.0.1:50052   --override experiment.output_dir=../outputs/rare_topk_grpc   --override runtime.device=cuda:0
```

客户端 `La2O3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id La2O3   --config configs/rare/topk.yaml   --override grpc.server_address=127.0.0.1:50052   --override experiment.output_dir=../outputs/rare_topk_grpc   --override runtime.device=cuda:0
```

#### 4.2.3 `rare` 的 EGA

服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/rare/ega.yaml   --override grpc.address=0.0.0.0:50053   --override grpc.server_address=127.0.0.1:50053   --override experiment.output_dir=../outputs/rare_ega_grpc   --override runtime.device=cuda:0
```

客户端 `Nd2O3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id Nd2O3   --config configs/rare/ega.yaml   --override grpc.server_address=127.0.0.1:50053   --override experiment.output_dir=../outputs/rare_ega_grpc   --override runtime.device=cuda:0
```

客户端 `CeO2`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id CeO2   --config configs/rare/ega.yaml   --override grpc.server_address=127.0.0.1:50053   --override experiment.output_dir=../outputs/rare_ega_grpc   --override runtime.device=cuda:0
```

客户端 `La2O3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id La2O3   --config configs/rare/ega.yaml   --override grpc.server_address=127.0.0.1:50053   --override experiment.output_dir=../outputs/rare_ega_grpc   --override runtime.device=cuda:0
```

#### 4.2.4 `mnist` 的 FedAvg / Top-k / EGA

`mnist` 的客户端固定是：`m1`、`m2`、`m3`。下面只替换配置文件、客户端 ID、输出目录和端口。

FedAvg 服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/mnist/fedavg.yaml   --override grpc.address=0.0.0.0:50061   --override grpc.server_address=127.0.0.1:50061   --override experiment.output_dir=../outputs/mnist_fedavg_grpc   --override runtime.device=cuda:0
```

FedAvg 客户端 `m1` / `m2` / `m3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m1   --config configs/mnist/fedavg.yaml   --override grpc.server_address=127.0.0.1:50061   --override experiment.output_dir=../outputs/mnist_fedavg_grpc   --override runtime.device=cuda:0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m2   --config configs/mnist/fedavg.yaml   --override grpc.server_address=127.0.0.1:50061   --override experiment.output_dir=../outputs/mnist_fedavg_grpc   --override runtime.device=cuda:0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m3   --config configs/mnist/fedavg.yaml   --override grpc.server_address=127.0.0.1:50061   --override experiment.output_dir=../outputs/mnist_fedavg_grpc   --override runtime.device=cuda:0
```

Top-k 服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/mnist/topk.yaml   --override grpc.address=0.0.0.0:50062   --override grpc.server_address=127.0.0.1:50062   --override experiment.output_dir=../outputs/mnist_topk_grpc   --override runtime.device=cuda:0
```

Top-k 客户端 `m1` / `m2` / `m3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m1   --config configs/mnist/topk.yaml   --override grpc.server_address=127.0.0.1:50062   --override experiment.output_dir=../outputs/mnist_topk_grpc   --override runtime.device=cuda:0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m2   --config configs/mnist/topk.yaml   --override grpc.server_address=127.0.0.1:50062   --override experiment.output_dir=../outputs/mnist_topk_grpc   --override runtime.device=cuda:0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m3   --config configs/mnist/topk.yaml   --override grpc.server_address=127.0.0.1:50062   --override experiment.output_dir=../outputs/mnist_topk_grpc   --override runtime.device=cuda:0
```

EGA 服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/mnist/ega.yaml   --override grpc.address=0.0.0.0:50063   --override grpc.server_address=127.0.0.1:50063   --override experiment.output_dir=../outputs/mnist_ega_grpc   --override runtime.device=cuda:0
```

EGA 客户端 `m1` / `m2` / `m3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m1   --config configs/mnist/ega.yaml   --override grpc.server_address=127.0.0.1:50063   --override experiment.output_dir=../outputs/mnist_ega_grpc   --override runtime.device=cuda:0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m2   --config configs/mnist/ega.yaml   --override grpc.server_address=127.0.0.1:50063   --override experiment.output_dir=../outputs/mnist_ega_grpc   --override runtime.device=cuda:0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id m3   --config configs/mnist/ega.yaml   --override grpc.server_address=127.0.0.1:50063   --override experiment.output_dir=../outputs/mnist_ega_grpc   --override runtime.device=cuda:0
```

#### 4.2.5 `cifar10` 的 FedAvg / Top-k / EGA

`cifar10` 的客户端固定是：`c1`、`c2`、`c3`。

FedAvg 服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/cifar10/fedavg.yaml   --override grpc.address=0.0.0.0:50071   --override grpc.server_address=127.0.0.1:50071   --override experiment.output_dir=../outputs/cifar10_fedavg_grpc   --override runtime.device=cuda:0
```

FedAvg 客户端 `c1` / `c2` / `c3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c1   --config configs/cifar10/fedavg.yaml   --override grpc.server_address=127.0.0.1:50071   --override experiment.output_dir=../outputs/cifar10_fedavg_grpc   --override runtime.device=cuda:0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c2   --config configs/cifar10/fedavg.yaml   --override grpc.server_address=127.0.0.1:50071   --override experiment.output_dir=../outputs/cifar10_fedavg_grpc   --override runtime.device=cuda:0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c3   --config configs/cifar10/fedavg.yaml   --override grpc.server_address=127.0.0.1:50071   --override experiment.output_dir=../outputs/cifar10_fedavg_grpc   --override runtime.device=cuda:0
```

Top-k 服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/cifar10/topk.yaml   --override grpc.address=0.0.0.0:50072   --override grpc.server_address=127.0.0.1:50072   --override experiment.output_dir=../outputs/cifar10_topk_grpc   --override runtime.device=cuda:0
```

Top-k 客户端 `c1` / `c2` / `c3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c1   --config configs/cifar10/topk.yaml   --override grpc.server_address=127.0.0.1:50072   --override experiment.output_dir=../outputs/cifar10_topk_grpc   --override runtime.device=cuda:0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c2   --config configs/cifar10/topk.yaml   --override grpc.server_address=127.0.0.1:50072   --override experiment.output_dir=../outputs/cifar10_topk_grpc   --override runtime.device=cuda:0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c3   --config configs/cifar10/topk.yaml   --override grpc.server_address=127.0.0.1:50072   --override experiment.output_dir=../outputs/cifar10_topk_grpc   --override runtime.device=cuda:0
```

EGA 服务端：

```bash
cd workspace/src
python -m fedlab.entrypoints.server   --config configs/cifar10/ega.yaml   --override grpc.address=0.0.0.0:50073   --override grpc.server_address=127.0.0.1:50073   --override experiment.output_dir=../outputs/cifar10_ega_grpc   --override runtime.device=cuda:0
```

EGA 客户端 `c1` / `c2` / `c3`：

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c1   --config configs/cifar10/ega.yaml   --override grpc.server_address=127.0.0.1:50073   --override experiment.output_dir=../outputs/cifar10_ega_grpc   --override runtime.device=cuda:0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c2   --config configs/cifar10/ega.yaml   --override grpc.server_address=127.0.0.1:50073   --override experiment.output_dir=../outputs/cifar10_ega_grpc   --override runtime.device=cuda:0
```

```bash
cd workspace/src
python -m fedlab.entrypoints.client   --client-id c3   --config configs/cifar10/ega.yaml   --override grpc.server_address=127.0.0.1:50073   --override experiment.output_dir=../outputs/cifar10_ega_grpc   --override runtime.device=cuda:0
```

## 5. 训练完成后的攻击回放与测试回放

### 5.1 训练输出里哪些文件最重要

一次实验运行结束后，常见关键产物包括：

- `summary.json`
- `metrics.json`
- `model.pt`
- `config.yaml`
- `saved_updates/`
- `attack_results.json`（如果在线攻击开启）

其中：

- `saved_updates/` 用于离线重放攻击
- `model.pt` 用于离线重放测试

### 5.2 回放全部已配置攻击

`replay_saved_update_attacks.py` 会读取某个实验目录下的 `saved_updates/`，并按当前配置重新执行已启用攻击：

```bash
cd workspace/src
python -m fedlab.tools.replay_saved_update_attacks   ../outputs/rare_fedavg_grpc   --output-dir ../outputs/rare_fedavg_grpc/offline_attack_replay
```

如果想临时修改攻击参数，可以继续追加 `--override`：

```bash
python -m fedlab.tools.replay_saved_update_attacks   ../outputs/rare_fedavg_grpc   --output-dir ../outputs/rare_fedavg_grpc/offline_attack_replay   --override attack.steps=300   --override attack.lr=0.05
```

### 5.3 只回放 DLG 或 iDLG

只回放 DLG：

```bash
python -m fedlab.tools.replay_saved_update_dlg   ../outputs/rare_fedavg_grpc   --output-dir ../outputs/rare_fedavg_grpc/offline_attack_replay_dlg
```

只回放 iDLG：

```bash
python -m fedlab.tools.replay_saved_update_idlg   ../outputs/rare_fedavg_grpc   --output-dir ../outputs/rare_fedavg_grpc/offline_attack_replay_idlg
```

### 5.4 对保存好的模型重新做测试回放

离线测试回放使用 `model.pt` 和对应实验配置重新构建数据加载器与模型，然后在共享测试集上评测。

最常用命令：

```bash
cd workspace/src
python -m fedlab.tools.replay_saved_model_evaluation   ../outputs/rare_fedavg_grpc/model.pt   --config ../outputs/rare_fedavg_grpc/config.yaml   --output-dir ../outputs/rare_fedavg_grpc/offline_test_replay
```

如果想换一套数据目录重新测试，可以覆盖 `data.split_dir`，或者直接传 `--data-dir`：

```bash
python -m fedlab.tools.replay_saved_model_evaluation   ../outputs/mnist_fedavg_grpc/model.pt   --config ../outputs/mnist_fedavg_grpc/config.yaml   --data-dir ../data/mnist   --output-dir ../outputs/mnist_fedavg_grpc/offline_test_replay
```

离线测试回放的结果通常会写到：

- `test_metrics.json`
- `test_summary.json`

## 6. 结果分析

### 6.1 直接使用 `analyze_experiment_suite`

如果你已经有一个实验套件目录，推荐也把分析结果输出到 `../outputs/analysis/...`：

```bash
cd workspace/src
PYTHONPATH=. python -m fedlab.tools.analyze_experiment_suite   ../outputs/exp/rare/single_sync/42/noattack_mse   --loss mse   --output-dir ../outputs/analysis/rare_single_sync_42_noattack_mse   --algorithms centralized fedavg topk ega
```

多 seed 联合分析时，可以一次传多个根目录：

```bash
PYTHONPATH=. python -m fedlab.tools.analyze_experiment_suite   ../outputs/exp/rare/single_sync/42/noattack_mse   ../outputs/exp/rare/single_sync/4096/noattack_mse   ../outputs/exp/rare/single_sync/2026/noattack_mse   ../outputs/exp/rare/single_sync/8192/noattack_mse   --loss mse   --output-dir ../outputs/analysis/rare_single_sync_multiseed_noattack_mse   --algorithms centralized fedavg topk ega
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

### 6.2 使用批处理脚本分析 `outputs/exp`

如果你的实验目录遵循现在的批量实验结构：

```text
../outputs/exp/<task>/<mode>/<seed>/<profile>_<loss>
```

那么可以直接用：

```bash
cd workspace/src
bash scripts/run_analyze_experiment_suite_batch.sh
```

默认等价于：

```bash
INPUT_ROOT=../outputs/exp OUTPUT_ROOT=../outputs/analysis/exp TASKS="rare mnist cifar10" TASK_LOSS_MAP="rare=mse,mae;mnist=cross_entropy;cifar10=cross_entropy" MODE=single_sync PROFILE=noattack SEEDS="42 4096 2026 8192" ALGORITHMS="centralized fedavg topk ega" bash scripts/run_analyze_experiment_suite_batch.sh
```

只分析某一个任务和模式：

```bash
TASKS="mnist" MODE=multi_sync PROFILE=attack SEEDS="42 4096" bash scripts/run_analyze_experiment_suite_batch.sh
```

也可以通过命令行参数传入：

```bash
bash scripts/run_analyze_experiment_suite_batch.sh   --input-root ../outputs/exp   --output-root ../outputs/analysis/exp   --tasks "rare mnist cifar10"   --task-loss-map "rare=mse,mae;mnist=cross_entropy;cifar10=cross_entropy"   --mode multi_sync   --profile attack   --seeds "42 4096 2026 8192"   --algorithms "centralized fedavg topk ega"
```

### 6.3 批量实验的运行与分析建议

如果你准备使用仓库里的批量实验脚本，推荐遵循下面的顺序：

1. 先准备 `../data/rare_earth_rawdata2`、`../data/mnist`、`../data/cifar10`
2. 再生成 `../outputs/exp/...` 实验目录
3. 最后用 `scripts/run_analyze_experiment_suite_batch.sh` 做单 seed 与多 seed 汇总分析

## 7. 进一步阅读

更详细的中文说明见：

- `docs/框架使用手册.md`
- `docs/config_reference.md`
