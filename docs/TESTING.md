# Testing / 测试

From the repository root:

```bash
python -m pip install -r XPolicyLab/policy/WorldScape_R2SI/requirements-test.txt
export PYTHONPATH="$PWD:$PWD/XPolicyLab:$PWD/XPolicyLab/policy/WorldScape_R2SI/worldscape-policy/src"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/shared/rsi-test-tmp
mkdir -p "$TMPDIR"
python -B -m pytest -p no:cacheprovider -q tests XPolicyLab/policy/WorldScape_R2SI/harness/tests
```

Coverage includes generic library loading, WAM commit/discard, Agent takeover, startup and duplicate RESET, failed cleanup, CALL/INFER trial state, collection/validation separation, process-exit publication and multi-round orchestration.

测试覆盖通用库加载、WAM 提交/丢弃、Agent 接管、启动与重复 RESET、异常清理、CALL/INFER 状态、采集与验证隔离、进程退出后的经验发布及多轮编排。

For the official policy by itself, run the Harness suite from the parent workspace of an official XPolicyLab checkout:

```bash
export PYTHONPATH="$PWD:$PWD/XPolicyLab"
export R2SI_TEST_XPL_ROOT="$PWD/XPolicyLab"
python -B -m pytest -p no:cacheprovider -q XPolicyLab/policy/WorldScape_R2SI/harness/tests
```

Protocol debug and simulator evaluation commands are in the [policy guide](../XPolicyLab/policy/WorldScape_R2SI/README.md). `VALIDATION.json` in the policy directory records test runs by version.

