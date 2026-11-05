# Generic Skills / 通用 Skill

The shared library contains 34 generic Skills, four tools and two primitives. Every task uses the same library.

共享经验库包含 34 份通用 Skill、4 个工具和 2 个原语，在任务间共享。

| ID | Type / 类型 | Description / 描述 |
|---|---|---|
| [behavioral-checkpoints](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/behavioral-checkpoints/SKILL.md) | skill | 公开约束与行为检查点 |
| [bimanual-workspace-coordination](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/bimanual-workspace-coordination/SKILL.md) | skill | 双臂空间与角色协调 |
| [complete-interaction-cycles](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/complete-interaction-cycles/SKILL.md) | skill | 完整离散交互周期 |
| [compound-footprint-fit](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/compound-footprint-fit/SKILL.md) | skill | 复合轮廓的整体适配 |
| [contact-and-fit](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/contact-and-fit/SKILL.md) | skill | 接触、精细适配与旋转配合 |
| [containment-and-transfer](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/containment-and-transfer/SKILL.md) | skill | 开口、边界与受限转移 |
| [deformable-materials](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/deformable-materials/SKILL.md) | skill | 柔性材料与形状变换 |
| [dual-load-passage](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/dual-load-passage/SKILL.md) | skill | 双持载荷的串行让行 |
| [enclosed-object-clearance](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/enclosed-object-clearance/SKILL.md) | skill | 解除包围后再横移 |
| [far-reach-load-retraction](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/far-reach-load-retraction/SKILL.md) | skill | 远端已持载荷的回收路径 |
| [geometry-driven-stage-order](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/geometry-driven-stage-order/SKILL.md) | skill | 按通道占用安排阶段 |
| [goal-interpretation](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/goal-interpretation/SKILL.md) | skill | Use when the instruction contains several requirements, relations or restrictions. |
| [grasp-support-release](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/grasp-support-release/SKILL.md) | skill | 抓取、支撑与释放 |
| [grouping-and-ordering](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/grouping-and-ordering/SKILL.md) | skill | 关系、排序与朝向约束 |
| [held-orientation-and-visibility](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/held-orientation-and-visibility/SKILL.md) | skill | 持有方向与视野取舍 |
| [hooking-and-threading](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/hooking-and-threading/SKILL.md) | skill | 绕越、穿越与受约束结构 |
| [interaction-feedback](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/interaction-feedback/SKILL.md) | skill | 离散交互与因果反馈 |
| [lost-payload-recovery](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/lost-payload-recovery/SKILL.md) | skill | 失去载荷后的重新定位 |
| [moving-target-interception](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/moving-target-interception/SKILL.md) | skill | 运动目标截获 |
| [nonprehensile-contact-selection](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/nonprehensile-contact-selection/SKILL.md) | skill | 选择保持支撑的接触面 |
| [other-arm-observer](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/other-arm-observer/SKILL.md) | tool | 用另一臂的公开视角观察 |
| [planar-translation-and-rotation](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/planar-translation-and-rotation/SKILL.md) | skill | 区分平移接触与旋转力矩 |
| [policy-applicability](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/policy-applicability/SKILL.md) | skill | Policy能力先验与受证据约束的委托 |
| [post-transfer-spill-recovery](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/post-transfer-spill-recovery/SKILL.md) | skill | 完成规定动作后的局部遗漏恢复 |
| [primitive-control](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/primitive-control/SKILL.md) | primitive | RoboProbe 动作原语 |
| [pushing-and-sweeping](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/pushing-and-sweeping/SKILL.md) | skill | 平面接触与推动 |
| [reappearance-evidence](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/reappearance-evidence/SKILL.md) | skill | 可见历史与重新出现的判定 |
| [relative-alignment](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/relative-alignment/SKILL.md) | skill | 相对对齐与支撑几何 |
| [return-home](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/return-home/SKILL.md) | skill | 全场景统一收尾与归位 |
| [robodojo-interface](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/robodojo-interface/SKILL.md) | tool | RoboDojo 官方观测与控制接口 |
| [robot-action-timing](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/robot-action-timing/SKILL.md) | skill | 净空航段合并位置姿态，接触/夹爪/反馈边界分开；不重复试抬、逐轴小步或静态wait。 |
| [robot-action-timing-tool](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/robot-action-timing-tool/SKILL.md) | primitive | 基于 RoboProbe 的动作生成 |
| [support-transfer-selection](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/support-transfer-selection/SKILL.md) | skill | 分清承托换手与直接交接 |
| [supported-unhook-and-regrasp](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/supported-unhook-and-regrasp/SKILL.md) | skill | 支撑后的脱钩与重新夹持 |
| [tabletop-handoff](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/tabletop-handoff/SKILL.md) | skill | 双臂交接与支撑转移 |
| [temporal-observation](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/temporal-observation/SKILL.md) | skill | 时序观察与记忆 |
| [thin-section-insertion](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/thin-section-insertion/SKILL.md) | skill | 薄截面与狭缝方向匹配 |
| [visual-progress-recovery](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/visual-progress-recovery/SKILL.md) | skill | 公开进展与纠偏 |
| [worldscape-wam-memory](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/worldscape-wam-memory/SKILL.md) | tool | WSP 时序记忆动作技能 |
| [worldscape-wam-nomemory](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/worldscape-wam-nomemory/SKILL.md) | tool | WSP flash 当前观测动作技能 |
