---
title: 在公司集群上用 SLURM 跑多机多卡训练
date: 2025-10-09
tags:
  - SLURM
  - 分布式训练
  - HPC
  - PyTorch
categories:
  - 工程实践
index_img: /img/slurm.png
math: false
---

相信大家一开始最头疼的任务之一，就是把一个已经支持分布式训练的模型跑到多台机器上。理论上听起来很简单——代码已经写好了，只需要run就行。但真正上手之后才发现，集群环境、网络配置、进程同步……每一个环节都有坑。遂出现这篇博客记录一下完整的流程。

## 为什么要用 SLURM？
![slurm的介绍](../../../../img/slurm_intro.png)
公司的 GPU 集群统一由 **SLURM**（Simple Linux Utility for Resource Management）管理。SLURM 本质上是一个作业调度系统，负责在多个用户之间公平地分配计算节点上的 CPU、GPU 和内存资源。

相比直接 `ssh` 到机器上手动启动进程，SLURM 有几个显著优势：

- **免守护终端**：任务提交后即在后台运行，不需要 `tmux` 或 `screen` 保持会话。所有输出会写入 `.out` 日志文件。
- **多节点一键分发**：多机任务时，SLURM 可以一次性向所有指定节点提交，不需要手动逐台 `ssh`。
- **集群状态透明**：通过 `squeue` 查看所有正在运行的任务，通过 `sinfo -N` 查看节点状态，便于协调资源使用。

## 分布式训练基础

在进入脚本细节之前，先简单梳理一下分布式训练的核心思路。

由于单张 GPU 显存有限，既无法容纳超大模型，也难以使用足够大的 batch size，因此需要将计算分散到多张乃至多台机器的 GPU 上。主流的并行方式分为两类：

- **数据并行（Data Parallelism）**：每张 GPU 上加载完整模型副本，输入 batch 被切分后分别计算梯度，再汇总更新参数。优点是实现简单、吞吐量大；缺点是每张 GPU 的显存占用并不减少，当模型本身就放不进单卡时无能为力。
- **模型并行（Model Parallelism）**：将模型本身拆开，不同 GPU 负责不同部分。包括张量并行（切分权重矩阵）、流水线并行（切分层）和序列并行（Sequence Parallel，切分 token 序列）等。其中序列并行能显著降低单卡显存，但通信开销大，速度约慢一倍，建议只在显存确实不足时启用。

实际工程中两者往往结合使用。本文以 `torchrun` 作为启动器，ColossalAI 作为并行框架为例。

## 编写 SLURM 脚本

一个完整的 SLURM 训练脚本（`train.sh`）由以下几个部分组成。

### 资源分配

脚本开头以 `#SBATCH` 指令向调度系统申请计算资源：

```bash
#!/bin/bash

#SBATCH --account=your_account       # 作业所属账户
#SBATCH --job-name=my_train          # 任务名称，显示在 squeue 中
#SBATCH --nodes=2                    # 请求节点数
#SBATCH --nodelist=node110,node111   # 指定节点（可选，不指定则由调度器分配）
#SBATCH --ntasks-per-node=1          # 每节点启动 1 个 SLURM 任务（对应 1 个 torchrun 进程）
#SBATCH --cpus-per-task=64           # 每任务分配的 CPU 核心数
#SBATCH --gres=gpu:8                 # 每节点请求 8 张 GPU
#SBATCH --partition=gpu_A6000        # 目标分区（即搭载特定 GPU 的节点组）
#SBATCH --time=20160                 # 最大运行时间（分钟），约 14 天
#SBATCH --open-mode=append           # 日志追加写入，避免重启时覆盖历史记录
#SBATCH --signal=USR2@120            # 任务结束前 120 秒发送 USR2 信号，可用于触发 checkpoint 保存
```

几个值得注意的地方：

- `--ntasks-per-node=1` 是多机分布式训练的惯用设置。每个节点只有一个 SLURM 任务，但该任务内部通过 `torchrun` 再启动多个进程（每张 GPU 对应一个）。
- `--gres=gpu:8` 中的 `gres` 是 *Generic RESources* 的缩写，用于申请 GPU 等非标准资源。
- `--signal=USR2@120` 是一个优雅退出的机制，可以在程序中捕获该信号，在任务被强制终止前保存训练状态。

### 环境配置

资源声明之后，需要手动设置运行环境。集群上的节点环境通常不会自动继承 master 终端的配置，因此需要显式指定。

**配置 CUDA 路径**时，`PATH` 采用追加方式，但 `LD_LIBRARY_PATH` 和 `CUDA_HOME` 必须**覆盖**而非追加——否则系统可能混用多个 CUDA 版本的库，导致难以排查的运行时错误：

```bash
# CUDA 11.8
export PATH=/path/to/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/path/to/cuda-11.8/lib64   # 覆盖，非追加
export CUDA_HOME=/path/to/cuda-11.8               # 覆盖，非追加
```

**激活 conda 虚拟环境**：

```bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate my_env
```

在脚本中显式激活环境有一个额外好处：提交任务时 master 节点无需预先激活同名环境。

**其他关键环境变量**：

```bash
export OMP_NUM_THREADS=8                              # 每进程 OpenMP 线程数
export SUBMITIT_EXECUTOR=slurm                        # 告知 submitit 使用 SLURM 后端
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 减少显存碎片
```

`OMP_NUM_THREADS` 的设置需要计算：若每节点 64 个 CPU 核心、8 张 GPU，则 8 个进程各分配 8 个线程恰好用满 CPU。NumPy、MKL、OpenBLAS 等底层计算库均依赖 OpenMP 进行 CPU 并行加速，该参数若不设置，默认值可能导致线程数远超核心数，产生大量上下文切换开销。

### NCCL 通信配置

多机训练最容易出问题的环节是**节点间通信**。我们的集群使用 **InfiniBand（IB）** 网络，这是一种专为 HPC 场景设计的高带宽、低延迟互联标准，相比以太网有数量级的性能优势。GPU 间的集合通信（all-reduce、broadcast 等）由 NVIDIA 的 **NCCL**（NVIDIA Collective Communications Library）负责调度。

启用 InfiniBand：

```bash
export NCCL_IB_DISABLE=0   # 0 = 启用 IB；1 = 禁用（退回以太网，调试时可用）
```

多机通信失败的最常见原因是 NCCL 选错了网卡。用 `ip addr` 先查清楚每台节点的网口，再决定哪些需要排除：

```bash
export NCCL_SOCKET_IFNAME=^lo   # ^ 表示排除，这里排除 loopback 接口
```

针对 InfiniBand 的深度调优：

```bash
# ==== NCCL 通用 ====
export NCCL_DEBUG=WARN               # 日志级别，排查问题时可改为 INFO 或 VERSION
export NCCL_ASYNC_ERROR_HANDLING=1   # 异步错误处理，提高健壮性
export NCCL_LAUNCH_MODE=GROUP        # 减少通信延迟

# ==== IB / RoCE 网络 ====
export NCCL_IB_HCA=mlx5_0           # 指定 IB 网卡（具体名称通过 ibstat 查看）
export NCCL_IB_GID_INDEX=0          # 使用 IB / RoCE v1
export NCCL_IB_TC=106               # 流量优先级，提升带宽服务质量
export NCCL_IB_TIMEOUT=22           # 超时阈值，大规模训练时适当调大

# ==== GPUDirect RDMA ====
export NCCL_NET_GDR_LEVEL=2         # 强制启用 GPUDirect，跳过 CPU 中转，显著提速

# ==== Ring 拓扑 ====
export NCCL_MIN_NRINGS=4            # 最小 Ring 数，NCCL 会自动适配实际拓扑
```

> **注意**：`NCCL_IB_HCA`、`NCCL_IB_GID_INDEX` 等参数的具体值依赖于机器的硬件配置，需要在目标节点上执行 `ibstat` 或联系集群管理员确认，不能照搬。

### 获取节点信息

SLURM 在运行时会将当前作业分配到的节点列表存入环境变量 `$SLURM_JOB_NODELIST`，但格式是压缩的（如 `node[110-111]`），需要用 `scontrol` 展开，再提取头节点（master）地址供 `torchrun` 使用：

```bash
NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
MASTER_NODE=$(echo "$NODELIST" | head -n 1)
MASTER_ADDR=$(echo "$NODELIST" | head -n 1)   # torchrun 需要 master 的可达地址
LAST_NODE=$(echo "$NODELIST" | tail -n 1)
NODE_NUM=$(echo "$NODELIST" | wc -l)
NODE_COUNT=0
```

打印关键信息到日志，便于事后核查实验配置：

```bash
echo "NODE_NUM=$NODE_NUM"
echo "NODELIST:"
echo "$NODELIST"
echo "MASTER_NODE=$MASTER_NODE, LAST_NODE=$LAST_NODE"
echo "MASTER_ADDR=$MASTER_ADDR"
```

### 启动多节点训练

最后，遍历节点列表，在每个节点上通过 `srun` 启动 `torchrun`。关键在于**除最后一个节点外，所有 `srun` 命令都加 `&` 放入后台**，最后一个节点的命令在前台运行并等待所有节点完成：

```bash
for NODE in $NODELIST; do
    echo "Launching on $NODE, node_rank=$NODE_COUNT"

    if [ "$NODE" = "$LAST_NODE" ]; then
        # 最后一个节点：前台运行，等待所有进程汇合
        srun --nodes=1 --ntasks=1 -w $NODE \
            torchrun \
                --nproc_per_node=8 \
                --nnodes=$NODE_NUM \
                --node_rank=$NODE_COUNT \
                --master_addr=$MASTER_ADDR \
                --master_port=34567 \
                train.py
    else
        # 非最后节点：后台挂起，继续循环
        srun --nodes=1 --ntasks=1 -w $NODE \
            torchrun \
                --nproc_per_node=8 \
                --nnodes=$NODE_NUM \
                --node_rank=$NODE_COUNT \
                --master_addr=$MASTER_ADDR \
                --master_port=34567 \
                train.py &
    fi

    ((NODE_COUNT++))
done
```

`torchrun` 参数说明：

| 参数 | 含义 |
|------|------|
| `--nproc_per_node` | 每节点启动的进程数，通常等于每节点 GPU 数 |
| `--nnodes` | 总节点数 |
| `--node_rank` | 当前节点的排名（0 为 master） |
| `--master_addr` | master 节点的 IP 或主机名 |
| `--master_port` | 通信端口，确保未被占用 |

这种逐节点循环提交的方式，虽然比某些"一行命令多节点"的方案稍显繁琐，但可靠性更高——每个节点的 `node_rank` 明确指定，不依赖 SLURM 的隐式行为。

## 提交任务

脚本写好后，一行命令提交：

```bash
sbatch train.sh
```

提交后可用 `squeue -u $USER` 查看任务状态，用 `tail -f <job_id>.out` 实时跟踪日志输出。

## 小结

完整的 SLURM 多机训练脚本可以拆解为五个层次：**资源申请 → 环境初始化 → 通信配置 → 节点发现 → 进程启动**。尤其是 NCCL 的网卡配置，要根据实际硬件逐项排查。
