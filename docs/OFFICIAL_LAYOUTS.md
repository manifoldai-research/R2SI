# Official layouts

The outer RSI loop uses RoboDojo's saved layouts for 34 non-Open base tasks and 12 Generalization random variants, giving 46 task entries.

The eight Open tasks are excluded from outer collection and validation: `align_blocks`, `general_pickup`, `solve_equation`, `stack_blocks_by_language`, `classify_objects_by_language`, `pick_from_conveyor_by_image`, `store_tools_in_toolbox` and `pour_by_language`. The inner Agent remains available through the official evaluation interface.

## Prepare layouts

```bash
python -B scripts/prepare_official_layout_cases.py \
  --robodojo-root /path/to/RoboDojo \
  --output /shared/rsi/official-cases.json
```

Defaults:
- Collection: seed 0, layout ordinal 0.
- Validation: seeds 0, 1 and 2, layout ordinal 1.
- All 46 non-Open entries; use `--tasks` for a subset.
- Each case records the official filename, SeedManager ordinal and file SHA256.

`scripts/configure_rsi.py` incorporates this plan directly into a runnable campaign configuration.

## Collection and validation

Collection and validation use distinct file identities. Some official files contain identical scene geometry; `layout_split` records shared content hashes. Reflection reads collection records only, including when a validation file has the same scene content.

Baseline and candidate run on the same validation files. The comparison uses score, success, actions and capability/seed coverage. Public-layout results are reported as public-layout results.

Layout records use `official_layout_iteration` and retain their original filenames. They supply reflection and validation; model training reads the configured training dataset.

Task categories come from the [official RoboDojo catalog](https://robodojo-benchmark.com/doc/sim-tasks/).

