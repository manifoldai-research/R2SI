# Generic Skills / 通用 Skill

The shared library contains 11 generic Skills, three tools and two primitives. Every task uses the same library. The eight Skills with `load_mode: always` are in the prompt at every step; the others are listed by id and read on demand. The on-demand tools, primitives and `policy-applicability`, `behavioral-checkpoints` stay because code or configuration refers to them.

共享经验库包含 11 份通用 Skill、3 个工具和 2 个原语，在任务间共享。8 份 `load_mode: always` 的 Skill 每步都在提示词里，其余只列 id、按需读取。按需的工具、原语和 `policy-applicability`、`behavioral-checkpoints` 因代码或配置引用而保留。

| ID | Type / 类型 | Load / 加载 | Description / 描述 |
|---|---|---|---|
| [behavioral-checkpoints](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/behavioral-checkpoints/SKILL.md) | skill | on_demand | 公开约束与行为检查点 |
| [deformable-materials](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/deformable-materials/SKILL.md) | skill | on_demand | 柔性材料与形状变换 |
| [grasp-by-shape](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/grasp-by-shape/SKILL.md) | skill | always | Take and hold things by their shape, check the hold, and change something real after failures. |
| [other-arm-observer](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/other-arm-observer/SKILL.md) | tool | on_demand | 用另一臂的公开视角观察 |
| [place-and-release](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/place-and-release/SKILL.md) | skill | always | Release at the right height on a supported seat; check before and after letting go. |
| [plan-and-save-steps](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/plan-and-save-steps/SKILL.md) | skill | always | Plan the whole job from the end backwards and spend native steps where they count. |
| [policy-applicability](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/policy-applicability/SKILL.md) | skill | on_demand | Policy能力先验与受证据约束的委托 |
| [primitive-control](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/primitive-control/SKILL.md) | primitive | on_demand | RoboProbe 动作原语 |
| [push-press-and-fit](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/push-press-and-fit/SKILL.md) | skill | always | Push, press, strike, fit tightly and drive mechanisms with checked contact. |
| [robot-action-timing-tool](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/robot-action-timing-tool/SKILL.md) | primitive | on_demand | 基于 RoboProbe 的动作生成 |
| [time-others-and-moving-targets](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/time-others-and-moving-targets/SKILL.md) | skill | always | Wait, give others room, note what you will need later, and meet moving targets. |
| [turn-and-tilt](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/turn-and-tilt/SKILL.md) | skill | always | Turn, tip and pour held things within reach, with the grip checked and the flow watched. |
| [two-arm-roles](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/two-arm-roles/SKILL.md) | skill | always | Give each arm a role, pass objects safely and carry things together. |
| [worldscape-wam-memory](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/worldscape-wam-memory/SKILL.md) | tool | on_demand | WSP 时序记忆动作技能 |
| [worldscape-wam-nomemory](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/worldscape-wam-nomemory/SKILL.md) | tool | on_demand | WSP flash 当前观测动作技能 |
| [wsp-learned-policy](../XPolicyLab/policy/WorldScape_R2SI/harness/vendor/policy/RoboRSI_RoboProbe/skills/generic/wsp-learned-policy/SKILL.md) | skill | always | When and how to hand both arms to the learned policy (WSP flash or memory), and how to take over. |
