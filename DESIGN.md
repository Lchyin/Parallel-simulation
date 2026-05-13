# 模型框架并行仿真系统设计文档

## 1. 目标与范围
本项目构建一个参考 Calculon / Vidur 思路的并行仿真框架，用于在不真实执行训练/推理时，评估不同硬件、并行、调度与参数组合下的性能与瓶颈。

## 2. 已实现能力（对应需求1~10）
1. ONNX 模型输入 + 多模型格式扩展接口（`LoaderRegistry`）。
2. 输出计算量、访存模式、通信需求（`ModelMetrics`）。
3. 支持自定义硬件（设备/节点/集群）。
4. 支持多并行策略探索（TP/PP/DP/SP）。
5. 支持多调度策略探索（重叠通信、微批大小）。
6. 支持参数网格搜索。
7. 支持多节点部署探索（跨节点带宽/时延）。
8. 支持模拟执行并生成多资源甘特图。
9. 支持不同硬件环境性能预测。
10. 支持瓶颈分析与优化建议。

## 3. 升级后的成本模型

### 3.1 计算成本（Compute）
- 算子级 FLOPs 估计（MatMul/Conv 特化）。
- 集群有效算力：`sum(peak_tflops * compute_efficiency)`。
- 计算时间：`compute_time = flops / effective_peak_flops`。

### 3.2 访存成本（Memory）
- 从输出字节推导激活字节近似（`activation_bytes`）。
- 集群有效带宽：`sum(memory_bw * memory_efficiency)`。
- 访存时间：`memory_time = activation_bits / effective_memory_bw`。

### 3.3 通信成本（Communication）
- TP/DP 通信量粗分解。
- 通信时间采用 α-β 模型 ring all-reduce 近似：
  `2*(p-1)*(alpha + n/(p*beta))`。
- 区分节点内通信（高带宽低时延）与节点间通信（低带宽高时延）。
- 调度策略可通过 overlap 因子降低通信暴露时间。

### 3.4 Pipeline 气泡惩罚
- 引入 `bubble_penalty = 1 + (pp-1)/(pp*micro_batch)`。
- 总时延：`(compute + memory + comm) * bubble_penalty`。

## 4. 架构模块
- `parallel_sim/io`: 模型输入与扩展。
- `parallel_sim/models`: 图表示与算子估算。
- `parallel_sim/hardware`: 硬件配置与效率参数。
- `parallel_sim/parallel`: 并行策略空间。
- `parallel_sim/scheduler`: 调度策略空间。
- `parallel_sim/analysis`: 指标与瓶颈分析。
- `parallel_sim/simulation`: 核心仿真引擎。
- `parallel_sim/optimization`: 联合搜索。
- `parallel_sim/report`: 摘要与甘特图。

## 5. 后续可增强点
- 细分 collective（all-gather/reduce-scatter）和拓扑（ring/tree）。
- 按算子类别接入经验效率模型与 profile 校准。
- 增加多目标优化（吞吐/时延/成本/能耗）。
- 输出 Chrome tracing/可视化前端。


## 6. 本轮新增
- 新增配置系统：`parallel_sim/config/schema.py` + `parallel_sim/config/loader.py`，支持JSON配置和基础校验。
- 新增内存时间线：`parallel_sim/simulation/memory.py`，输出weights/grads/optimizer/activations快照。
- 新增示例配置：`examples/simulation_config.json`。
- 新增基础测试：`tests/test_config_and_memory.py`。
