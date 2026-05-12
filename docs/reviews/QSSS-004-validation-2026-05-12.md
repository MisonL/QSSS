# QSSS-004 Validation - 2026-05-12

## Task

- ID: QSSS-004
- Title: 新增同花顺本地板块成分读取器
- Scope: 只读解析 `block_conception.ini` 和 `block_industry.ini`，不读取登录态、token、cookie 或私有 socket。

## Field Contract

- `board_code`
- `board_name`
- `board_type`
- `symbol`
- `name`
- `source`
- `version`

## Verification

Command:

```bash
uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm pytest -q tests/test_ths_local_board_contract.py
```

Result:

- Exit code: 0
- Summary: 4 passed, 1 warning

## Review Notes

- Parser 仅按显式文件路径读取本地 ini 内容。
- 路径未配置或不存在时抛出 `FileNotFoundError`，错误信息包含对应默认文件名。
- 测试 fixture 覆盖人工智能、CPO、存储芯片、智能电网。
