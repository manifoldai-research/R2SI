# Export the official policy / 导出官方策略

The full repository is the development source. The XPolicyLab PR adds only `policy/WorldScape_R2SI/`.

完整仓库是开发源，官方 PR 只新增 `policy/WorldScape_R2SI/`。

```bash
python -B scripts/export_policy.py --output /shared/releases/WorldScape_R2SI-policy.tar.gz
```

Extract the archive into an official XPolicyLab checkout. It contains the Agent, generic Skills, WAM implementation and standard install/data/train/eval scripts.

将归档解压到官方 XPolicyLab 仓库；其中包含 Agent、通用 Skill、WAM 实现及安装、数据处理、训练和评测脚本。

After changing frozen runtime assets in a candidate copy, refresh its manifest:

```bash
cd XPolicyLab/policy/WorldScape_R2SI
python -B maintain_manifest.py --write-new-release
python -B maintain_manifest.py
```

Run [tests](TESTING.md), then export. The full repository and policy archive should carry the same `harness/release.json`.

修改候选副本中的冻结运行资产后更新清单、运行测试，再导出。完整仓库和官方策略包使用同一份 `harness/release.json`。

Format references: [XPolicyLab submission standard](https://github.com/XPolicyLab/XPolicyLab/blob/main/CONTRIBUTING.md), [HexaShell PR](https://github.com/XPolicyLab/XPolicyLab/pull/172), [PyRUA-Lean PR](https://github.com/XPolicyLab/XPolicyLab/pull/168).

