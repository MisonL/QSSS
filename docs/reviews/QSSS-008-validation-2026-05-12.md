# QSSS-008 Validation - 2026-05-12

## Task

- ID: QSSS-008
- Title: 清理装饰性字符存量
- Scope: README、web README、交互脚本、运行脚本和用户可见模板中的 emoji、Unicode 装饰符号、框线字符。

## Verification

Command:

```bash
rg -n -P "[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]|[╔╗╚╝║═┌┐└┘├┤┬┴┼─│]|[✓✗]|[¥©]" README.md web/README.md scripts src/qsss/interactive_cli.py src/qsss/simple_interactive.py web/run.py web/templates
```

Result:

- Exit code: 1
- Summary: no matches

Command:

```bash
uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm pytest -q tests/test_package_imports.py tests/test_cli_analyze_contract.py
```

Result:

- Exit code: 0
- Summary: 4 passed, 1 warning

## Review Notes

- README tree diagrams now use ASCII characters.
- User-visible CLI and run-script status strings no longer depend on emoji or checkmark symbols.
