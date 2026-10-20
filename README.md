# HotMoney A股拉升前特征研究模块

本项目使用 **Tushare Pro** 获取 A 股日线行情和龙虎榜数据，生成拉升事件及事件前特征样本。

## 功能

- 获取当前上市 A 股列表和历史日线行情。
- 计算过去 1/3/5/10/20 日收益、成交额/成交量相对 5 日均值、波动率、均线偏离。
- 定义研究事件：未来 3 个交易日收盘价相对当前收盘价上涨至少 15%。
- 对事件日前 1–5 个交易日生成回溯样本。
- 获取龙虎榜股票级汇总；在账号权限允许时额外获取机构龙虎榜明细。
- 输出 CSV 和 JSON，供后续统计检验、基线模型和滚动回测使用。

**重要限制：Tushare Pro 的 top_list 是股票级龙虎榜汇总，不包含营业部席位名称。因此本版本不能匹配具体游资营业部，也不会伪造席位匹配结果。** 如必须研究具体营业部，需要另找提供席位明细的数据源。

## 安装与配置

~~~bash
python -m pip install -r requirements.txt
~~~

先在 Tushare Pro 注册并获取 Token，然后设置环境变量。

Windows CMD（当前窗口有效）：

~~~bat
set TUSHARE_TOKEN=你的TushareProToken
~~~

PowerShell（当前窗口有效）：

~~~powershell
$env:TUSHARE_TOKEN="你的TushareProToken"
~~~

## 运行

先小范围验证 Token、接口权限和字段：

~~~bash
python hotmoney_analyzer.py --limit 10 --start 2024-10-20 --end 2026-10-20 --output data
~~~

指定股票：

~~~bash
python hotmoney_analyzer.py --symbols 600519,000001 --start 2024-10-20 --end 2026-10-20
~~~

不传 --limit 或 --symbols 会处理当前上市股票列表，耗时和接口调用量较大。Tushare 的接口权限、积分门槛和调用频率取决于账号；首次建议先用少量股票测试。

## 输出

- data/daily_features.csv：每日行情及技术特征
- data/lhb_events.csv：股票级龙虎榜事件
- data/lhb_institutional_events.csv：机构龙虎榜明细（接口权限允许时）
- data/rally_events.csv：符合事件定义的样本
- data/pre_rally_samples.csv：事件前 1–5 日的回溯样本
- data/analysis_summary.json：样本数量与特征均值对照
- data/cache/：行情和龙虎榜缓存

## 重要边界

- 这是历史研究工具，不是自动荐股或买卖系统。
- “未来 3 日涨幅 >= 15%”是研究标签，不等同于真实游资拉升的因果证明。
- pre_rally_1_5d 使用未来行情构造，只能作为训练标签，不能作为当日预测特征，否则会产生数据泄漏。
- 当前股票列表可能漏掉已退市股票，存在幸存者偏差。
- Tushare 日线的成交额单位为千元，成交量单位为手。
- 代码已切换到 Tushare Pro，但尚未在当前环境使用真实 Token 完成端到端联网验证；运行时需检查账号权限、接口配额和输出行数。
- 建模前应加入失败拉升/无拉升对照样本、时间序列滚动验证、交易费用、滑点、涨跌停无法成交及 A 股 T+1 约束。
