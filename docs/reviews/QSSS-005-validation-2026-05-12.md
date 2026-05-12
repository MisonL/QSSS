# QSSS-005 Validation - 2026-05-12

## Task

- ID: QSSS-005
- Title: 修正 CLI 参数行为
- Scope: `qsss analyze --start-date` 和 `--limit` 进入策略分析与 CLI 展示路径。

## Verification

Command:

```bash
uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm pytest -q tests/test_cli_analyze_contract.py tests/test_strategy_core.py
```

Result:

- Exit code: 0
- Summary: 3 passed, 1 warning

## Review Notes

- CLI 调用 `QuantStrategy.run_analysis(start_date=start_date, limit=limit)`。
- `calculate_ma15` 使用相同 `start_date`，避免均线读取旧默认日期缓存。
- README 已移除参数为预留接口的说明。
