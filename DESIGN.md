# 模型框架并行仿真系统设计文档

## 1. 目标与范围
本项目构建一个“类似 Calculon / Vidur 思路”的并行仿真框架，用于在不实际训练/推理的情况下，快速评估模型在不同硬件、并行策略、调度策略与参数组合下的性能、通信与瓶颈。

### 覆盖能力（对应需求 1~10）
1. **模型输入**：已支持 ONNX 输入，使用 `LoaderRegistry` 预留多格式扩展（TorchScript/MLIR/自定义 IR）。
2. **指标输出**：输出 FLOPs、访存规模/模式、通信需求。
3. **硬件自定义**：支持设备、节点、集群参数化定义。
4. **并行策略探索**：支持 TP/PP/DP/SP 策略空间枚举。
5. **调度策略探索**：支持不同调度策略（重叠通信、微批配置）。
6. **参数组合探索**：支持任意 `param_grid` 网格搜索。
7. **多节点部署探索**：支持多节点集群建模与跨节点带宽建模。
8. **模拟执行 + 甘特图**：输出时间线并可生成 ASCII 甘特图。
9. **跨硬件性能预测**：可替换 `ClusterSpec` 做不同硬件预测对比。
10. **瓶颈分析 + 建议**：按 compute/comm/memory 主导项给出优化建议。

## 2. 架构总览
- `parallel_sim/io`：模型加载层（ONNX + 扩展接口）。
- `parallel_sim/models`：统一模型图抽象 `ModelGraph` / `ModelOp`。
- `parallel_sim/hardware`：硬件抽象（设备/节点/集群）。
- `parallel_sim/parallel`：并行策略定义与搜索空间。
- `parallel_sim/scheduler`：调度策略定义。
- `parallel_sim/analysis`：计算量、通信、访存与瓶颈分析。
- `parallel_sim/simulation`：仿真执行引擎，生成阶段时间分解。
- `parallel_sim/optimization`：联合搜索器（策略 + 调度 + 参数）。
- `parallel_sim/report.py`：摘要和甘特图输出。

## 3. 核心数据模型
### 3.1 模型图
- `ModelOp`：描述算子类型、输入输出、属性、输出张量形状。
- `ModelGraph`：算子列表 + 元数据，支持整体 FLOPs 和输出访存估算。

### 3.2 硬件模型
- `DeviceSpec`：峰值算力、显存带宽、容量、互联带宽。
- `NodeSpec`：单节点多设备拓扑。
- `ClusterSpec`：多节点集群 + 跨节点带宽。

### 3.3 策略模型
- `ParallelStrategy`：TP/PP/DP/SP 组合并做设备可行性校验。
- `SchedulePolicy`：是否通信重叠、微批粒度等调度属性。

## 4. 仿真与估计方法
### 4.1 模型指标
- FLOPs：按算子类型估算（MatMul/Conv 特化 + fallback）。
- Output memory：按输出张量总元素量估算。
- Communication：基于 TP 粗估 all-reduce/all-gather 量级。
- Memory pattern：按图规模判断 `streaming` / `layerwise_reuse`。

### 4.2 执行时间模型
总时延由三部分线性分解：
- `compute_time = FLOPs / total_peak_tflops`
- `memory_time = bytes / total_mem_bandwidth`
- `comm_time = comm_bytes / effective_interconnect_bw`

调度策略通过 `overlap_communication` 改写有效通信带宽，体现通信计算重叠收益。

## 5. 寻优流程
`Explorer.explore()` 执行联合搜索：
1. 枚举并行策略；
2. 枚举调度策略；
3. 枚举参数网格（batch/seq 等）；
4. 过滤不可行策略；
5. 调用仿真引擎得分；
6. 按总时延排序输出 Pareto 候选（当前按单目标 latency）。

## 6. 扩展点设计
- **模型输入扩展**：新增 `ModelLoader` 实现，并注册到 `LoaderRegistry`。
- **算子级精细模型**：扩展 `ModelOp.estimate_flops()` + 引入 kernel-level 成本库。
- **通信模型增强**：按拓扑（ring/tree/dragonfly）选择不同代价函数。
- **调度策略增强**：支持 stage-aware pipeline / EDF / critical-path 优化。
- **搜索增强**：从网格搜索升级到贝叶斯优化/遗传算法/多目标优化。
- **可视化增强**：输出 JSON trace（Chrome tracing）或 matplotlib 甘特图。

## 7. 与 Calculon / Vidur 思路对齐
- 使用**结构化硬件与策略建模**（而非写死脚本）。
- 支持**what-if 仿真探索**（并行维度 + 参数维度 + 调度维度）。
- 使用**阶段化执行估计**并输出性能解释（瓶颈+建议）。

## 8. 当前实现边界
- ONNX shape 推断使用轻量 fallback；生产级建议接入完整 shape inference。
- 通信模型目前为简化模型；实际应区分 collective 类型和分层拓扑。
- 尚未接入真实运行 trace 校准；建议通过 profiling 数据拟合参数。

## 9. 快速使用
```bash
python3 main.py
```
输出包括：模型指标、最优策略、性能摘要、甘特图。
