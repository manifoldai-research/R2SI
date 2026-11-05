# Contact-admissible orientation candidates / 接触允许的姿态候选

The optional `robot-action-timing-tool` operation `move_candidates` compares native
plans for one world grasp point and a small, explicitly reviewed set of flange
orientations. The Agent must establish contact admissibility from current RGB;
the tool does not infer object geometry, symmetry, clearance or grasp success.
It does not introduce camera parameters, object assets or task selectors.

可选操作 `move_candidates` 在同一个世界抓取点比较一组经 Agent 审阅的法兰姿态。
接触方向、夹持截面及净空由当前 RGB 判断；工具不推断物体形状、对称性或抓取成功，
也不增加相机参数、物体资产或按任务名分派。

## Interface / 接口

Use `call_tool` with `id="robot-action-timing-tool"` and arguments:

- `op="move_candidates"`, `arm="left"` or `"right"`.
- `position`: current-scene world **grasp-point** XYZ in metres.
- `quaternions_wxyz`: 2–8 unit world **flange** quaternions, all contact-admissible.
- `contact_evidence`: why the candidates meet the current contact constraints.

The full schema is documented in the existing robot-action-timing-tool Skill.
The 2–8 bound limits one search, not robot speed, episode duration or motion count.
The current orientation and 180-degree alternatives are never added implicitly.
An object need not admit a symmetric grasp. Gripper commands remain separate.

position 使用抓取点；朝向使用世界法兰单位 WXYZ 四元数。工具偏置转换沿用原接口，
因此改变朝向时法兰位置通常也会改变。2–8 上限只控制一次候选搜索规模。
不自动添加当前姿态或对称姿态，不将闭爪合并到移动中。

## Selection and execution / 选择与执行

Every candidate passes through the existing native move builder, including planning,
start continuity, joint-limit checks and command resampling. Known planning rejection
is recorded and another explicitly supplied candidate may be considered. Infrastructure
errors abort the search; a previously planned candidate is not dispatched as fallback.
Target-jitter experiments are rejected because they would break the common-contact-point
comparison. The existing single-target `move` is unchanged.

Plans are ranked lexicographically by maximum raw joint excursion from the current
state, summed absolute joint travel, flange rotation from the current orientation,
planned action count, then input index. Raw angles are not wrapped: a multi-turn
trajectory must not look like a short rotation. This is a deterministic heuristic,
not an optimality claim or proof that the selected path avoids external obstacles.

Only the winning native action sequence is returned to the official evaluator. It is
not replanned. Arrival tracking is committed to the selected target; losing proposals
and failed searches do not change prior target/gripper intent. Selection provenance
is included in the tool result and `ActionChunk.meta.candidate_selection`. No evaluation
executes a robot action, and successful planning is not reported as arrival.

每个候选都经过现有原生规划、起点连续性、关节限位和动作重采样检查。
选择顺序是最大原始关节偏离、累计关节路程、法兰转角、动作数和输入序号；
不将多圈关节角折算为短角差。只有选中的原生动作序列交给官方评测器，
不再次规划，不把规划成功记为实际到位。基础设施异常时放弃整个搜索。

## Validation and limits / 验证与边界

`harness/tests/test_candidate_motion.py` uses CPU planner fixtures with the real native
move construction and tool dispatcher. It checks path reuse, common grasp-point semantics,
raw multi-turn and whole-path costs, quaternion sign equivalence, all-rejected and
infrastructure-error cases, existing joint/start checks and the single-target route.
No CUDA planning, debug closed loop or simulator performance is established by these tests.

For performance evaluation, pair baseline/candidate on the same authorized public layout
identities, seeds, models and budgets using `rsi/validation.py`; separate collection and
validation records. Report official scores/success, native action counts, planning and
wall-clock cost, nonarrival and cleanup. Candidate search adds planner calls and can cost
more wall time even if its selected path is shorter. No throughput or success-rate gain
is claimed until those measurements exist.

CPU 测试通过真实动作构造链路及工具入口，规划器返回路径由测试夹具提供。
尚未进行 CUDA 规划、闭环调试或仿真收益验证。后续须在同布局、seed、模型和预算下
配对验证，分别记录官方表现、动作数、规划时间、实际未到位和清理状态。
多候选会增加规划调用，路径更短不等于总耗时更短。
