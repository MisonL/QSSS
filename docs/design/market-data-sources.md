# 免费优先行情数据源替换方案

## 1. Control Contract

- Primary Setpoint: QSSS 使用免费额度优先的数据源组合，按能力获取实时行情、板块资金流、历史日线和板块成分，现有策略输入字段保持稳定。
- Acceptance:
  - 默认 `pytest -q` 通过。
  - 新增非 external contract 测试覆盖实时行情字段、板块资金流字段、历史日线字段和同花顺本地缓存解析字段。
  - external smoke 可分别验证 pytdx、AKShare、BaoStock、小额度 iFinD 的真实连通性，但不作为默认测试前置。
- Guardrail Metrics:
  - 不增加默认测试对真实网络的依赖。
  - 不把 iFinD 免费额度用于全 A 高频轮询。
  - 不读取同花顺登录态、WebKit token 或私有 socket。
  - 不改变策略层当前消费的统一字段名。
- Sampling Plan:
  - L0: 每个 provider 的离线字段契约测试。
  - L1: DataManager capability routing 测试和小集成测试。
  - L2: 带 `external` 标记的真实数据源 smoke，按自选池小样本执行。
- Known Delays / Delay Budget:
  - 免费接口网络抖动和上游限流不可控。
  - 板块资金流 1 到 5 分钟采样即可，不追求秒级。
  - 历史数据收盘后批处理，不进入盘中快回路。
- Recovery Target:
  - 单个 provider 失败时不影响其他能力域。
  - 实时行情主源失败时保留上一版快照并暴露错误状态，不伪造新行情。
- Rollback Trigger:
  - 默认测试失败。
  - 统一字段契约被破坏。
  - 真实接口失败被吞掉或被标成成功。
  - iFinD 调用无法计量单元格预算。
- Constraints:
  - 全程免费优先。
  - 个人自用。
  - 禁止逆向同花顺私有协议。
  - 禁止隐藏降级和 mock 成功路径。
- Boundary:
  - 允许修改 `src/qsss/data/`、`src/qsss/config/`、相关 tests、README/设计文档。
  - 本轮方案不改数据库 schema、不改策略打分、不改 Web 页面交互。
- Coupling Notes:
  - `QuantStrategy` 依赖 `DataManager.get_stock_list` 和 `get_daily_data` 的字段契约。
  - Web 路由和 Celery 任务直接调用 `DataManager`。
  - 依赖声明需要和 provider 注册策略一致。
- Approximation Validity:
  - 离线测试只能证明字段和路由语义，不证明真实网络稳定。
  - external smoke 只能证明当次可用，不代表长期 SLA。
- Actuator Budget:
  - 第一阶段允许做接口抽象、字段契约、provider 注册和测试。
  - 第二阶段才接入板块、历史和缓存成分。
- Risks:
  - 免费源字段变化导致解析失败。缓解方式是 contract 测试和真实 smoke。
  - 全 A 高频请求触发上游限制。缓解方式是采样频率和批大小限制。
  - 多源回退形成双真相。缓解方式是按能力域明确唯一主源和兜底源。

## 2. 当前状态估计

现有主链是“按数据源名选择 adapter”：

- `DataManager` 默认注册 `pytdx`，可选注册 `tushare` 和 `baostock`。
- `get_stock_list` 和 `get_daily_data` 支持主备源降级。
- `get_realtime_data` 只获取单一 adapter，不做能力域路由。
- `src/qsss/data/adapters.py` 注释写 AKShare 已移除，但 `src/qsss/data/sources/akshare_adapter.py` 仍存在。
- `PytdxAdapter.get_realtime_data` 当前读取 `name`、`volume`、`time`，与 pytdx 实测字段 `vol`、`servertime` 存在契约风险。

结论：问题不是单个 adapter 换名字，而是需要从 source-based routing 改为 capability-based routing。

## 3. 项目级控制拓扑

- 总体设计部: `AGENTS.md` 和 `issues.csv` 维护项目级参考输入、冻结边界和任务顺序。
- 模块 owner:
  - 数据源模块: `src/qsss/data/`
  - 配置模块: `src/qsss/config/settings.py`
  - 策略消费方: `src/qsss/core/strategy.py` 和 `src/qsss/core/optimized_strategy.py`
  - Web 消费方: `web/routes.py` 和 `web/tasks.py`
- 共享边界 owner:
  - 统一行情字段。
  - provider 能力接口。
  - 默认测试和 external smoke gate。

本次主落点:

- 数据面: 行情、板块、历史数据获取路径。
- 控制面: provider 选择、采样频率、失败降级、额度预算。
- 状态面: 后续 SQLite 快照和历史表，本轮方案先冻结 schema。

冻结边界:

- 策略层输入字段不顺手改名。
- Web/CLI 调用入口先兼容旧 `DataManager` 方法。
- 同花顺只读本地 ini，不读取登录态。
- iFinD 不作为默认主链，不写死 token。

## 4. 目标能力分层

| 能力域 | 主源 | 备用或辅助 | 说明 |
| --- | --- | --- | --- |
| 实时行情 | pytdx | mootdx 可选后补 | 个股、指数、盘口快照，自选池高频，全 A 低频 |
| 板块和资金流 | AKShare | 同花顺本地成分映射 | 概念、行业、资金流、ETF，不承担秒级实时行情 |
| 历史日线和复权 | BaoStock | pytdx 或 AKShare 小样本补充 | 收盘后批处理，不进入盘中快回路 |
| 板块成分 | 同花顺本地缓存 | AKShare 概念成分 | 只读 ini，增强同花顺概念口径 |
| 增强兜底 | iFinD 免费额度 | 无 | 只查重点池和关键字段，必须有单元格预算 |

## 5. 目标接口形态

保留旧 `DataManager` 方法，内部切到 capability routing。

建议新增能力接口:

```python
class RealtimeQuoteProvider:
    def get_quotes(self, symbols: list[str]) -> pd.DataFrame:
        ...


class HistoryProvider:
    def get_daily_bars(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        ...


class BoardProvider:
    def get_boards(self) -> pd.DataFrame:
        ...

    def get_board_members(self, board_code: str) -> pd.DataFrame:
        ...


class BoardFlowProvider:
    def get_board_flows(self) -> pd.DataFrame:
        ...
```

旧方法映射:

| 旧入口 | 新能力 |
| --- | --- |
| `get_stock_list` | stock universe provider，优先 pytdx，后续可用 AKShare/BaoStock 补全 |
| `get_daily_data` | history provider，优先 BaoStock，pytdx 兼容保留 |
| `get_realtime_data` | realtime quote provider，优先 pytdx |
| 新增 `get_board_flows` | board flow provider，优先 AKShare |
| 新增 `get_board_members` | local THS board membership provider |

## 6. 统一字段契约

实时行情字段:

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | str | 6 位证券代码 |
| name | str | 名称，可缺失时为空字符串，但不得伪造 |
| price | float | 最新价 |
| last_close | float | 昨收 |
| open | float | 今开 |
| high | float | 最高 |
| low | float | 最低 |
| volume | float | 成交量，pytdx 来源映射自 `vol` |
| amount | float | 成交额 |
| quote_time | str | 行情时间，pytdx 来源映射自 `servertime` |
| source | str | 数据源名 |
| fetched_at | str | 本机采集时间 |

历史日线字段沿用当前 `REQUIRED_DAILY_COLUMNS`:

`date, open, close, high, low, volume, amount, amplitude, pct_chg, change, turn`

板块资金流字段:

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| board_code | str | 板块代码或稳定名称散列 |
| board_name | str | 板块名称 |
| board_type | str | concept 或 industry |
| pct_chg | float | 涨跌幅 |
| amount | float | 成交额 |
| net_inflow | float | 净流入 |
| main_net_inflow | float | 主力净流入 |
| source | str | 数据源名 |
| fetched_at | str | 本机采集时间 |

板块成分字段:

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| board_code | str | 板块代码 |
| board_name | str | 板块名称 |
| board_type | str | concept 或 industry |
| symbol | str | 股票代码 |
| name | str | 股票名称，可为空 |
| source | str | `ths_local_cache` 或 `akshare` |
| version | str | 本地缓存版本或数据源日期 |

## 7. 采样与额度策略

| 任务 | 数据源 | 频率 | 控制约束 |
| --- | --- | --- | --- |
| 自选池实时快照 | pytdx | 5 到 15 秒 | 每批不超过 80 只 |
| 全 A 快照 | pytdx | 1 到 5 分钟 | 分批并发，失败进入冷却 |
| 指数行情 | pytdx | 5 到 30 秒 | 小样本优先 |
| 板块和资金流 | AKShare | 1 到 5 分钟 | 失败保留上一版并显式标记 stale |
| 历史日线和复权 | BaoStock | 收盘后一次 | 不进盘中快回路 |
| 同花顺本地成分 | 本地 ini | 启动时或每日一次 | 文件不存在直接报错 |
| iFinD 增强字段 | iFinD | 按需 | 仅重点池，必须估算单元格预算 |

iFinD 单元格预算建议:

- 默认关闭。
- 每次请求前估算 `symbols * fields`。
- 单次默认上限 5000 单元格。
- 月度预算按配置写入，不硬编码 token。
- 超预算时显式失败，不自动回退成其他源的假成功。

## 8. 分阶段实施计划

### Phase 1: Contract first

目标: 先锁字段和路由，不扩大业务面。

- 修复 `PytdxAdapter.get_realtime_data` 字段映射。
- 新增实时行情离线 contract 测试。
- 保留旧 `DataManager.get_realtime_data` 入口。
- 默认测试必须通过。

对应任务: `QSSS-001`

### Phase 2: Dependency and registry cleanup

目标: 依赖声明和 provider 注册一致。

- 统一 `pyproject.toml`、`requirements.txt`、`web/requirements.txt`。
- 明确 AKShare、BaoStock 默认安装或 optional extras。
- 移除“AKShare 已移除”的错误事实。

对应任务: `QSSS-002`

### Phase 3: Board and flow providers

目标: AKShare 正式承担板块和资金流。

- 让 `AkshareAdapter` 实现板块和资金流能力。
- 新增 `DataManager.get_board_flows`。
- external smoke 小样本验证真实 AKShare 输出。

对应任务: `QSSS-003`

### Phase 4: Local THS membership provider

目标: 同花顺本地缓存只作为成分辅助源。

- 新增只读 parser。
- 支持 `block_conception.ini` 和 `block_industry.ini`。
- 文件必须为 UTF-8 编码；GBK/GB18030 旧缓存需先转换，例如：
  `iconv -f GBK -t UTF-8 block_conception.ini > block_conception.utf8.ini`。
- 不读取 WebKit 缓存、token、cookie。

对应任务: `QSSS-004`

变更记录:

- 2026-05-13: 同花顺本地缓存解析改为严格 UTF-8。遇到非法字节会显式报错，不再静默跳过；升级前请转换旧 GBK/GB18030 文件。

### Phase 5: History provider and storage bridge

目标: 把历史日线职责迁到 BaoStock，给 SQLite MVP 留稳定输入。

- BaoStock 作为 history provider。
- pytdx 日线保留为兼容或补充。
- SQLite schema 设计后再进入落库。

对应任务: `QSSS-006`

### Phase 6: Web MVP

目标: 页面展示真实采集数据。

- 行情总览。
- 板块轮动。
- AI 科技观察池。

对应任务: `QSSS-007`

## 9. 复杂性转移账本

| 原位置 | 新位置 | 收益 | 新成本 | 失效模式 |
| --- | --- | --- | --- | --- |
| 单一 DataAdapter 同时承担实时、历史、列表 | capability provider registry | 职责清晰，免费源组合可控 | 注册和路由逻辑增加 | 路由配置错误导致能力不可用 |
| Tushare/BaoStock 作为 source fallback | 按能力指定主源和备用源 | 避免不同数据口径混用 | 需要字段契约测试 | 双真相或字段错位 |
| iFinD 作为潜在增强源 | 额度受控的 premium fallback | 保护免费额度 | 需要预算统计 | 超预算或误用全 A 高频 |
| 同花顺 App 探测结果 | 本地 ini 只读 parser | 避免私有协议风险 | 本地路径和版本要观测 | 文件缺失或版本过旧 |

## 10. 验证矩阵

| 层级 | 命令或动作 | 目标 |
| --- | --- | --- |
| L0 | `pytest -q tests/test_data_manager_health.py` | 主备和降级逻辑不破坏 |
| L0 | `pytest -q tests/test_package_imports.py` | 导入不加载重依赖 |
| L0 | 新增 provider contract tests | 字段契约稳定 |
| L1 | 默认 `pytest -q` | 非 external 回归通过 |
| L2 | `pytest -m external tests/test_data_adapters_external.py` | 真实源小样本可用 |
| L2 | Web smoke | 页面展示真实数据，不用静态假数据 |

## 11. Rollback and recovery

- 若 Phase 1 失败，只回滚 PytdxAdapter 字段映射相关改动，保留方案文档。
- 若 Phase 3 AKShare 不稳定，不接管实时行情，只保留为板块/资金流 optional provider。
- 若同花顺本地缓存路径不可用，不影响 pytdx、AKShare、BaoStock 主链。
- 若 iFinD 预算不可计量，保持默认关闭。
