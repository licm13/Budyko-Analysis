# RFC: Storage-Adjusted Evaporation Engine

## 背景
- 来源：`研究思路.md` 中 **点子三：方法论深化（径流是“参照物”）** 提出需要引入 GRACE 等数据修正水储量项 $\Delta S$，以检验 Budyko 框架假设。
- 现状：仓库中仅支持经典的 $Ea \approx P - Q$，无法插入外部水储量数据，既无法验证假设也无法对多源数据做对比实验。

## 方案概述
1. **MVP API**：`StorageAdjustmentEngine.compute` 接收降水、径流与可选的储量变化序列，返回经 $\Delta S$ 调整后的实际蒸发量及诊断信息。
2. **扩展接口**：
   - `register_provider`/`providers` 参数：允许以策略模式挂载不同数据源（GRACE、陆面模式、同化产品等）。
   - `aggregator` 钩子：用户可自定义多源合成逻辑（均值、加权平均、质量控制等）。
   - `allow_negative_evaporation` 开关：针对极端或对抗性实验允许越界值。
3. **回归与监控**：结果对象暴露 `residual`、`metadata` 便于集成日志或 QA。

## 替代方案
- **在 `DeviationAnalysis` 内部增加布尔开关**：侵入式，导致耦合加深且难以在其他流程重复利用。
- **通过 pandas 管道手写逻辑**：缺乏统一异常处理与扩展钩子，不利于自动化测试和未来的 streaming 实现。

## 取舍
- 选择模块化 Engine + Provider 结构以满足“端口-适配器”要求，牺牲一定初学者友好性，但获得可测试与可扩展的 API 面。
- 默认禁止负蒸发值，确保物理一致性；通过参数允许专家级用户覆盖。

## 风险
- 多源储量数据存在空间/时间分辨率差异，需要额外对齐逻辑（后续扩展）。
- GRACE 数据常含缺测，需要外部质量控制或 aggregator 中实现；当前版本只提供 `NaN` 保留策略。
- 如果 provider 数量过多，逐一调用可能影响性能；后续可通过并行或缓存处理。

## 复杂度与性能
- `compute` 主体为矢量化 `O(n)`，内存占用约为输入数组的 3 倍（P、Q、ΔS 与派生数组）。
- 并行建议：
  - 对大量流域可在 provider 内部并行抓取数据，再一次性交给 Engine。
  - 对 streaming 场景，可扩展 Engine 接收迭代器并逐批处理（保留 TODO）。

## 兼容性说明
- 不破坏现有 API，`DeviationAnalysis` 等模块继续使用原先签名。
- 迁移步骤：
  1. 现有流程保持不变；
  2. 需要 GRACE 校正时，引入 `StorageAdjustmentEngine` 并注册对应 provider；
  3. 用 `compute` 输出替换原先的 `P - Q` 结果。

## TODO
- [ ] Provider 层引入统一的单位/空间分辨率检查。
- [ ] 设计流式接口以支撑长时间序列的低内存计算。
