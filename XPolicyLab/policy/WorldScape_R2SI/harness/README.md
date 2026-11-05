# Harness implementation

See [the policy README](../README.md) for installation, checkpoint bindings,
provider profiles and validation scope.

The single execution path is policy `model.py` → the frozen boundary in
`harness/model.py` → `CodeAgentBackend` → `runtime.AgentRuntime` → the v4
code-writing `Agent`. Standalone deployment and outer RSI use this same path.
`deploy.yml` configures the runtime. There is no alternate Agent selector.

The shared generic library remains in
`vendor/policy/RoboRSI_RoboProbe/skills/generic/` and is read through
`robo.skills()` / `robo.read_skill()`. `memory/*.md` enters the Agent's prompt.
Outer-loop code updates target the actual policy-level Agent and tool files.
`robo.policy(profile="flash" or "memory")` uses the retained WAM implementation
through `wam.WamSkills` and the policy-level `wam.BundledSession` adapter.
`configs/wam_skills.json` binds both WSP variants, their checkpoints and source.

WAM receives only RGB, joint state and an instruction. The bundled path checks
the complete chunk's shape and finite values, then the unified bridge checks
each action before it is exposed. A final
execution observation commits the chunk before it is passed into WSP's next
history window. Both Skills follow the published checkpoint's rolling inference
configuration and preserve the random generator across completed chunks. Flash's
checkpoint capacity is one chunk, so its visual input contains only the latest
frame; Memory uses its 24-chunk capacity. Neither resets independently before
each prediction. Partial/uncertain execution discards the prediction. Switching
to native motion or another WAM resets the old context; inactive models do not
accumulate stale observations. Read-only Skill/tools calls preserve continuity.
Trial end closes every loaded model, including after failures.

`release.py` freezes code, documents, configuration and integration entry points.
`development.py` imports offline Skill-asset revisions with evidence hashes
into a separate candidate; it does not conduct recursive experiments or training.
`verify_checkpoints.py` verifies both external weight directories without GPU
inference. It must run again when binding the release on another machine.
