# QSSS-003 Validation - 2026-05-12

## Task

- ID: QSSS-003
- Title: 恢复 AKShare 板块和资金流数据源
- Scope: AKShare 板块列表、概念资金流、DataManager 统一入口、external smoke 测试。

## Field Contracts

Board list fields:

- `board_code`
- `board_name`
- `board_type`
- `pct_chg`
- `amount`
- `source`
- `fetched_at`

Board flow fields:

- `board_code`
- `board_name`
- `board_type`
- `pct_chg`
- `amount`
- `net_inflow`
- `main_net_inflow`
- `source`
- `fetched_at`

## Verification

Command:

```bash
uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm --with akshare pytest -q tests/test_akshare_board_flow_contract.py
```

Result:

- Exit code: 0
- Summary: 5 passed, 1 warning

Command:

```bash
uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm --with akshare pytest -q tests/test_data_adapters_external.py -m external -k akshare -rs
```

Result:

- Exit code: 0
- Summary: 2 passed, 8 deselected, 12 warnings
- Notes: 当次真实 AKShare smoke 覆盖概念板块列表和概念板块资金流。

Command:

```bash
uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm pytest -q
```

Result:

- Exit code: 0
- Summary: 22 passed, 10 deselected, 1 warning

## Review Notes

- `DataManager` 注册 AKShare 为板块和资金流能力源，实时行情仍由既有实时数据源承担。
- AKShare 板块列表优先使用 Eastmoney 接口；若该接口真实失败，显式记录失败原因后尝试 AKShare 同花顺公开板块列表接口。
- AKShare 板块资金流失败时抛出包含原始错误的 `RuntimeError`，不返回伪成功结果。
