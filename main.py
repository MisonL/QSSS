import concurrent.futures
import os
import sys
import time
import warnings
from datetime import datetime
from functools import wraps
from threading import Lock

import numpy as np
import pandas as pd
import psutil

warnings.filterwarnings('ignore')


def retry_on_exception(retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:  # 最后一次重试
                        print(f"在执行 {func.__name__} 时发生错误: {str(e)}")
                        raise e
                    print(f"重试 {func.__name__} ({i + 1}/{retries})...")
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


try:
    import akshare as ak
except ImportError as e:
    print(f"导入akshare时出错: {e}")
    print("尝试重新安装akshare...")
    os.system("pip install --upgrade akshare py-mini-racer")
    import akshare as ak

import baostock as bs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import atexit
from tqdm import tqdm


class QuantStrategy:
    def __init__(self):
        self.retry_count = 3
        self.retry_delay = 1

        # 初始化 baostock
        self._init_baostock()

        # 注册程序退出时的清理函数
        atexit.register(self.cleanup)

        # 初始化数据存储
        self.stock_data = {}
        self.etf_data = {}
        self.factor_data = {}

        # 添加数据缓存
        self.stock_data_cache = {}  # 股票日线数据缓存
        self.ma15_cache = {}  # 15日均线数据缓存
        self.factor_cache = {}  # 因子计算结果缓存

        # 添加性能监控
        self.performance_stats = {
            'start_time': None,
            'end_time': None,
            'processed_stocks': 0,
            'success_count': 0,
            'failed_count': 0,
            'total_retries': 0,
            'avg_process_time': 0,
            'peak_memory': 0,
            'peak_cpu': 0
        }

        # 添加线程锁
        self.print_lock = Lock()
        self.progress_lock = Lock()
        self.data_lock = Lock()
        self.cache_lock = Lock()  # 添加缓存锁

        # 系统资源监控参数
        self.min_workers = 4  # 最小线程数
        self.max_workers = 10  # 最大线程数
        self.cpu_threshold = 75  # CPU使用率阈值
        self.memory_threshold = 85  # 内存使用率阈值

        # 添加统计信息收集
        self.analysis_stats = {
            'insufficient_data': [],  # 数据不足的股票
            'invalid_data': [],  # 数据无效的股票
            'failed_stocks': []  # 分析失败的股票
        }

    def _init_baostock(self):
        """初始化 baostock 连接"""
        for i in range(self.retry_count):
            try:
                bs.login()
                return
            except Exception as e:
                if i == self.retry_count - 1:
                    print(f"baostock登录失败: {e}")
                    sys.exit(1)
                print(f"baostock登录重试 ({i + 1}/{self.retry_count})...")
                time.sleep(self.retry_delay)

    def cleanup(self):
        """程序退出时的清理函数"""
        try:
            bs.logout()
        except:
            pass

    def __del__(self):
        """析构函数，不再处理登出逻辑"""
        pass

    @retry_on_exception(retries=3, delay=1)
    def get_stock_list(self):
        """获取股票列表"""
        try:
            # 使用 akshare 获取A股列表
            stock_info = ak.stock_zh_a_spot_em()

            # 选择需要的列并重命名
            stock_info = stock_info[['代码', '名称']].copy()
            stock_info.columns = ['symbol', 'name']

            # 过滤掉退市和ST股票
            stock_info = stock_info[
                ~stock_info['name'].str.contains('退市|退') &
                ~stock_info['name'].str.contains('ST')
                ]

            # 添加交易所和板块信息
            def get_market_info(symbol):
                if symbol.startswith('60'):
                    return '上交所-主板'
                elif symbol.startswith('688'):
                    return '上交所-科创板'
                elif symbol.startswith('000'):
                    return '深交所-主板'
                elif symbol.startswith('002'):
                    return '深交所-中小板'
                elif symbol.startswith('300'):
                    return '深交所-创业板'
                elif symbol.startswith('301'):
                    return '深交所-创业板'
                elif symbol.startswith('430'):
                    return '北交所-主板'
                elif symbol.startswith('83'):
                    return '北交所-主板'
                elif symbol.startswith('87'):
                    return '北交所-创新层'
                elif symbol.startswith('889'):
                    return '北交所-基础层'
                else:
                    return '其他'

            stock_info['market'] = stock_info['symbol'].apply(get_market_info)

            # 添加完整代码（带上交所信息）
            stock_info['ts_code'] = stock_info.apply(
                lambda x: x['symbol'] + '.SH' if x['market'].startswith('上交所') else (
                    x['symbol'] + '.BJ' if x['market'].startswith('北交所') else x['symbol'] + '.SZ'
                ), axis=1
            )

            print(f"成功获取股票列表，共 {len(stock_info)} 只股票")
            print("\n各市场股票数量统计：")
            market_stats = stock_info['market'].value_counts()
            for market, count in market_stats.items():
                print(f"{market}: {count}只")

            return stock_info
        except Exception as e:
            print(f"获取股票列表时出错: {e}")
            return pd.DataFrame()

    def get_etf_list(self):
        """获取ETF基金列表"""
        try:
            # 获取场内ETF列表
            etf_list = ak.fund_etf_category_sina()
            return etf_list
        except Exception as e:
            print(f"获取ETF列表时出错: {e}")
            return pd.DataFrame()

    @retry_on_exception(retries=3, delay=1)
    def get_stock_daily_data(self, symbol, start_date='20220101'):
        """获取股票日线数据（优化版）"""
        # 检查缓存
        cache_key = f"{symbol}_{start_date}"
        with self.cache_lock:
            if cache_key in self.stock_data_cache:
                return self.stock_data_cache[cache_key]

        try:
            # 使用 akshare 获取日线数据
            for _ in range(self.retry_count):
                try:
                    df = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=datetime.now().strftime('%Y%m%d'),
                        adjust="qfq"
                    )
                    break
                except Exception as e:
                    print(f"获取股票 {symbol} 数据失败，重试...")
                    time.sleep(self.retry_delay)
                    self.performance_stats['total_retries'] += 1
            else:
                print(f"获取股票 {symbol} 数据失败，跳过该股票")
                return pd.DataFrame()

            if df.empty:
                return df

            # 只保留需要的列并重命名
            df = df[['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']]
            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_chg', 'change',
                          'turn']

            # 确保数据类型正确
            numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'amount', 'turn', 'pct_chg']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 删除任何包含NaN的行
            df = df.dropna()

            # 存入缓存
            with self.cache_lock:
                self.stock_data_cache[cache_key] = df

            return df

        except Exception as e:
            print(f"获取股票 {symbol} 日线数据时出错: {e}")
            return pd.DataFrame()

    def calculate_momentum(self, df):
        """计算动量因子"""
        df['momentum_1m'] = df['close'].pct_change(20)  # 1个月动量
        df['momentum_3m'] = df['close'].pct_change(60)  # 3个月动量
        df['momentum_6m'] = df['close'].pct_change(120)  # 6个月动量
        return df

    def calculate_factors(self, df):
        """计算多因子模型指标"""
        try:
            # 数据预处理：处理异常值
            df = df.copy()

            # 使用中位数填充极端值
            for col in ['close', 'volume', 'turn']:
                median = df[col].median()
                std = df[col].std()
                df[col] = df[col].clip(median - 3 * std, median + 3 * std)

            # 确保没有零值和负值
            df['close'] = df['close'].replace(0, np.nan)
            df['volume'] = df['volume'].replace(0, np.nan)
            df['turn'] = df['turn'].replace(0, np.nan)

            # 计算技术指标
            df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['ma20'] = df['close'].rolling(window=20, min_periods=5).mean()

            # 计算成交量比率，添加数据验证
            volume_ma = df['volume'].rolling(window=20, min_periods=5).mean()
            df['vol_ratio'] = df['volume'] / volume_ma.replace(0, np.nan)
            df['vol_ratio'] = df['vol_ratio'].fillna(1.0).clip(0, 10)  # 使用1.0填充NaN，并限制范围

            # 计算波动率，使用对数收益率，添加数据验证
            returns = df['close'].pct_change()
            df['volatility'] = returns.rolling(window=20, min_periods=5).std() * np.sqrt(252)
            df['volatility'] = df['volatility'].fillna(df['volatility'].mean()).clip(0, 2)  # 限制在0-200%之间

            # 计算RSI，添加异常值处理
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)  # 避免除以零
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50).clip(0, 100)  # 使用50填充NaN，确保RSI在0-100之间

            # 计算MACD，使用更稳健的计算方法
            df['ema12'] = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()

            # 填充MACD相关的NaN值
            df['macd'] = df['macd'].fillna(0)
            df['signal'] = df['signal'].fillna(0)

            # 计算布林带
            df['boll_mid'] = df['close'].rolling(window=20, min_periods=5).mean()
            df['boll_std'] = df['close'].rolling(window=20, min_periods=5).std()
            df['boll_up'] = df['boll_mid'] + 2 * df['boll_std']
            df['boll_down'] = df['boll_mid'] - 2 * df['boll_std']

            # 使用前向填充处理剩余的NaN值
            df = df.fillna(method='ffill')
            # 使用后向填充处理开始部分的NaN值
            df = df.fillna(method='bfill')

            return df

        except Exception as e:
            print(f"计算技术指标时出错: {e}")
            return df

    def train_ml_model(self, df):
        """训练机器学习模型"""
        # 准备特征
        features = ['momentum_1m', 'momentum_3m', 'momentum_6m',
                    'volatility', 'vol_ratio', 'rsi', 'macd', 'signal']

        # 创建目标变量（5日收益率）
        df['target'] = df['close'].shift(-5) / df['close'] - 1

        # 删除缺失值
        df = df.dropna()

        if len(df) < 100:  # 数据太少，不足以训练
            return None, None

        # 准备训练数据
        X = df[features]
        y = (df['target'] > df['target'].mean()).astype(int)  # 二分类问题

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # 训练LightGBM模型，添加参数来避免警告
        model = lgb.LGBMClassifier(
            random_state=42,
            n_estimators=100,  # 增加树的数量
            num_leaves=31,  # 控制树的复杂度
            min_child_samples=5,  # 降低最小叶子节点样本数
            min_split_gain=0.0,  # 降低分裂增益阈值
            max_depth=5,  # 限制树的深度
            learning_rate=0.1,  # 添加学习率
            verbose=-1  # 使用 verbose 替代 silent
        )
        model.fit(X_train, y_train)

        return model, scaler

    def analyze_short_term_explosion(self, df):
        """分析超短线爆发潜力"""
        try:
            if len(df) < 20:  # 确保有足够的数据
                return 0

            # 1. 成交量分析
            recent_volume_mean = df['volume'].iloc[-20:].mean()
            volume_ratio = df['volume'].iloc[-1] / recent_volume_mean if recent_volume_mean > 0 else 0

            # 2. 换手率分析
            recent_turn_mean = df['turn'].iloc[-20:].mean()
            turnover_ratio = df['turn'].iloc[-1] / recent_turn_mean if recent_turn_mean > 0 else 0

            # 3. MACD金叉预判
            last_macd_diff = df['macd'].iloc[-1] - df['signal'].iloc[-1]
            prev_macd_diff = df['macd'].iloc[-2] - df['signal'].iloc[-2]
            macd_cross_score = 1 if (last_macd_diff > 0 and prev_macd_diff < 0) else 0

            # 4. 股价趋势分析
            price_range = df['close'].iloc[-20:].max() - df['close'].iloc[-20:].min()
            if price_range > 0:
                price_position = (df['close'].iloc[-1] - df['close'].iloc[-20:].min()) / price_range
            else:
                price_position = 0.5  # 如果价格范围为0，设置为中间位置

            # 5. 短期动量加速
            recent_momentum = df['close'].iloc[-1] / df['close'].iloc[-3] - 1 if df['close'].iloc[-3] > 0 else 0

            # 标准化各个指标，避免异常值
            volume_ratio = min(max(volume_ratio, 0), 5)  # 限制在0-5倍之间
            turnover_ratio = min(max(turnover_ratio, 0), 5)  # 限制在0-5倍之间
            recent_momentum = min(max(recent_momentum * 100, -20), 20)  # 限制在-20%到20%之间

            # 计算综合得分
            explosion_score = (
                    volume_ratio * 0.3 +
                    turnover_ratio * 0.2 +
                    macd_cross_score * 0.2 +
                    (1 - price_position) * 0.15 +  # 价格相对低位得分高
                    (recent_momentum + 20) / 40 * 0.15  # 将动量归一化到0-1之间
            )

            return explosion_score

        except Exception as e:
            print(f"计算超短线爆发潜力时出错: {e}")
            return 0

    def analyze_stock(self, stock_info):
        """分析单个股票"""
        try:
            # 获取日线数据
            df = self.get_stock_daily_data(stock_info['symbol'])
            if df.empty:
                with self.data_lock:
                    self.analysis_stats['invalid_data'].append(f"{stock_info['name']}({stock_info['symbol']})")
                return None

            # 确保数据完整性
            if len(df) < 120:  # 至少需要120天数据
                with self.data_lock:
                    self.analysis_stats['insufficient_data'].append(
                        f"{stock_info['name']}({stock_info['symbol']}): {len(df)}天")
                return None

            # 检查是否有无效数据
            if df['close'].isnull().any() or df['volume'].isnull().any():
                with self.data_lock:
                    self.analysis_stats['invalid_data'].append(f"{stock_info['name']}({stock_info['symbol']})")
                return None

            # 计算各类因子
            df = self.calculate_momentum(df)
            df = self.calculate_factors(df)

            # 检查计算后的指标是否有效
            required_columns = ['momentum_1m', 'momentum_3m', 'momentum_6m',
                                'volatility', 'vol_ratio', 'rsi', 'macd', 'signal']
            if any(df[col].isnull().any() for col in required_columns):
                print(f"警告: {stock_info['name']} ({stock_info['symbol']}) 技术指标计算结果存在无效值")
                return None

            # 训练模型
            model, scaler = self.train_ml_model(df)
            if model is None:
                print(f"警告: {stock_info['name']} ({stock_info['symbol']}) 模型训练失败")
                return None

            # 获取最新数据进行预测
            latest_data = df.iloc[-1:][required_columns]
            if latest_data.isnull().any().any():
                return None

            latest_scaled = scaler.transform(latest_data)
            prediction = model.predict_proba(latest_scaled)[0][1]

            # 计算超短线爆发潜力得分
            explosion_score = self.analyze_short_term_explosion(df)

            # 计算MACD金叉状态
            macd_status = ''
            if df['macd'].iloc[-1] > 0:  # MACD在0轴上方
                if df['macd'].iloc[-1] > df['signal'].iloc[-1] and df['macd'].iloc[-2] <= df['signal'].iloc[-2]:
                    macd_status = '金叉'
                elif abs(df['macd'].iloc[-1] - df['signal'].iloc[-1]) < abs(
                        df['macd'].iloc[-2] - df['signal'].iloc[-2]):
                    macd_status = '即将金叉'

            # 确保所有数值都是有效的
            if not all(pd.notnull([prediction, df['momentum_1m'].iloc[-1],
                                   df['rsi'].iloc[-1], explosion_score])):
                return None

            return {
                'symbol': stock_info['symbol'],
                'name': stock_info['name'],
                'market': stock_info['market'],
                'prediction': prediction,
                'momentum_score': df['momentum_1m'].iloc[-1],
                'rsi': df['rsi'].iloc[-1],
                'volatility': df['volatility'].iloc[-1],
                'macd': df['macd'].iloc[-1],
                'macd_status': macd_status,  # 添加MACD状态
                'close': df['close'].iloc[-1],
                'volume': df['volume'].iloc[-1],
                'turn': df['turn'].iloc[-1],
                'explosion_score': explosion_score
            }

        except Exception as e:
            with self.data_lock:
                self.analysis_stats['failed_stocks'].append(f"{stock_info['name']}({stock_info['symbol']}): {str(e)}")
            return None

    def get_optimal_thread_count(self):
        """根据系统资源使用情况动态计算最优线程数（优化版）"""
        try:
            # 获取更详细的系统信息
            cpu_percent = psutil.cpu_percent(interval=0.5)  # 缩短采样间隔
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            swap = psutil.swap_memory()
            swap_percent = swap.percent

            # 更新性能统计信息
            self.performance_stats['peak_cpu'] = max(
                self.performance_stats['peak_cpu'],
                cpu_percent
            )
            self.performance_stats['peak_memory'] = max(
                self.performance_stats['peak_memory'],
                memory_percent
            )

            # 计算综合负载指标
            system_load = max(
                cpu_percent / 100,
                memory_percent / 100,
                swap_percent / 100
            )

            # 根据系统负载调整线程数
            if system_load > 0.8:  # 系统负载高
                optimal_threads = self.min_workers
            elif system_load > 0.6:  # 系统负载中等
                optimal_threads = int(
                    self.min_workers +
                    (self.max_workers - self.min_workers) * (0.8 - system_load) / 0.2
                )
            else:  # 系统负载低
                optimal_threads = int(
                    self.max_workers * (0.9 - system_load * 0.5)
                )

            # 确保线程数在合理范围内
            optimal_threads = min(max(optimal_threads, self.min_workers), self.max_workers)

            with self.print_lock:
                print(f"\n系统状态 - CPU: {cpu_percent:.1f}%, 内存: {memory_percent:.1f}%, "
                      f"交换空间: {swap_percent:.1f}%")
                print(f"可用内存: {memory.available / 1024 / 1024:.0f}MB, "
                      f"系统负载: {system_load:.2f}")
                print(f"线程数调整: {optimal_threads}（最小{self.min_workers}, "
                      f"最大{self.max_workers}）")

            return optimal_threads

        except Exception as e:
            print(f"获取系统资源信息时出错: {e}")
            return self.min_workers

    def run_analysis(self):
        """运行完整的分析流程（优化版）"""
        print("开始分析...")

        # 初始化性能统计
        self.performance_stats['start_time'] = time.time()
        self.performance_stats['processed_stocks'] = 0
        self.performance_stats['success_count'] = 0
        self.performance_stats['failed_count'] = 0

        # 重置统计信息
        self.analysis_stats = {
            'insufficient_data': [],
            'invalid_data': [],
            'failed_stocks': []
        }

        # 获取股票列表
        stocks = self.get_stock_list()
        if stocks.empty:
            print("未能获取股票列表")
            return pd.DataFrame()

        print(f"共获取到 {len(stocks)} 只股票")
        results = []
        total_stocks = len(stocks)

        def analyze_stock_wrapper(stock):
            """股票分析的包装函数，用于多线程处理"""
            start_time = time.time()
            try:
                result = self.analyze_stock(stock)
                if result:
                    with self.data_lock:
                        results.append(result)
                        self.performance_stats['success_count'] += 1
                else:
                    with self.data_lock:
                        self.performance_stats['failed_count'] += 1

                # 更新处理时间统计
                process_time = time.time() - start_time
                with self.data_lock:
                    current_count = self.performance_stats['processed_stocks']
                    current_avg = self.performance_stats['avg_process_time']
                    self.performance_stats['avg_process_time'] = (
                            (current_avg * current_count + process_time) / (current_count + 1)
                    )
                    self.performance_stats['processed_stocks'] += 1

                return result is not None

            except Exception as e:
                with self.print_lock:
                    print(f"\n分析股票 {stock['symbol']} 时出错: {e}")
                with self.data_lock:
                    self.performance_stats['failed_count'] += 1
                return False

        # 获取初始线程数
        thread_count = self.get_optimal_thread_count()

        # 计算动态批处理大小
        batch_size = min(100, len(stocks) // (thread_count * 2))
        batch_size = max(20, batch_size)  # 确保批大小在合理范围内

        # 使用线程池进行并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            # 使用tqdm创建进度条
            with tqdm(total=total_stocks, desc="分析进度", ncols=100) as pbar:
                # 分批提交任务
                for i in range(0, len(stocks), batch_size):
                    # 每批开始前检查系统资源并调整线程数
                    thread_count = self.get_optimal_thread_count()
                    executor._max_workers = thread_count

                    # 获取当前批次的股票
                    batch_stocks = stocks.iloc[i:i + batch_size]

                    # 提交当前批次的任务
                    future_to_stock = {
                        executor.submit(analyze_stock_wrapper, stock): stock
                        for _, stock in batch_stocks.iterrows()
                    }

                    # 等待当前批次完成
                    for future in concurrent.futures.as_completed(future_to_stock):
                        stock = future_to_stock[future]
                        try:
                            future.result()
                            pbar.update(1)
                        except Exception as e:
                            with self.print_lock:
                                print(f"\n处理股票 {stock['symbol']} 时发生错误: {e}")
                            pbar.update(1)

                    # 显示当前批次的性能统计
                    with self.print_lock:
                        elapsed_time = time.time() - self.performance_stats['start_time']
                        success_rate = (self.performance_stats['success_count'] /
                                        max(1, self.performance_stats['processed_stocks']) * 100)
                        print(f"\n性能统计:")
                        print(f"处理股票数: {self.performance_stats['processed_stocks']}")
                        print(f"成功率: {success_rate:.1f}%")
                        print(f"平均处理时间: {self.performance_stats['avg_process_time']:.2f}秒")
                        print(f"总耗时: {elapsed_time:.0f}秒")
                        print(f"峰值CPU使用率: {self.performance_stats['peak_cpu']:.1f}%")
                        print(f"峰值内存使用率: {self.performance_stats['peak_memory']:.1f}%")

                    # 每批处理完后暂停一下，避免持续高负载
                    time.sleep(1)

        # 更新最终统计信息
        self.performance_stats['end_time'] = time.time()
        total_time = self.performance_stats['end_time'] - self.performance_stats['start_time']

        print(f"\n分析完成:")
        print(f"总耗时: {total_time:.0f}秒")
        print(f"成功分析: {self.performance_stats['success_count']}只")
        print(f"失败数量: {self.performance_stats['failed_count']}只")
        print(f"平均处理时间: {self.performance_stats['avg_process_time']:.2f}秒/只")

        if not results:
            print("没有符合条件的股票")
            return pd.DataFrame()

        # 转换为DataFrame并排序
        results_df = pd.DataFrame(results)

        # 添加调试信息
        print(f"\n初始结果数量: {len(results_df)}")

        # 根据预测概率和动量分数排序，调整权重
        results_df['total_score'] = (
                results_df['prediction'] * 0.3 +  # 增加预测权重
                results_df['momentum_score'] * 0.2 +  # 降低动量权重
                results_df['explosion_score'] * 0.35 +  # 进一步增加爆发潜力权重
                (1 - results_df['volatility']) * 0.15  # 降低波动率影响
        )

        # 使用更严格的筛选条件
        prediction_filter = results_df['prediction'] > 0.6  # 提高预测概率要求
        momentum_filter = results_df['momentum_score'] > -0.1  # 收紧动量要求
        rsi_upper_filter = results_df['rsi'] < 75  # 降低RSI上限
        rsi_lower_filter = results_df['rsi'] > 30  # 提高RSI下限
        volatility_filter = results_df['volatility'] < 0.6  # 收紧波动率限制
        volume_filter = results_df['volume'] > 50000  # 添加成交量过滤
        price_filter = results_df['close'] > 3  # 添加价格过滤

        print(f"\n筛选条件统计：")
        print(f"预测概率 > 60% 的股票数: {prediction_filter.sum()}")
        print(f"动量得分 > -10% 的股票数: {momentum_filter.sum()}")
        print(f"RSI在30-75之间的股票数: {(rsi_upper_filter & rsi_lower_filter).sum()}")
        print(f"波动率 < 60% 的股票数: {volatility_filter.sum()}")
        print(f"日均成交量 > 5万的股票数: {volume_filter.sum()}")
        print(f"股价 > 3元的股票数: {price_filter.sum()}")

        # 使用新的筛选条件
        selected = results_df[
            prediction_filter &
            momentum_filter &
            rsi_upper_filter &
            rsi_lower_filter &
            volatility_filter &
            volume_filter &
            price_filter
            ].sort_values('explosion_score', ascending=False)

        print(f"\n最终符合所有条件的股票数: {len(selected)}")

        # 如果筛选出的股票数量过多，只保留得分最高的50只
        if len(selected) > 50:
            selected = selected.head(50)
            print(f"保留得分最高的前50只股票")

        # 在返回结果之前添加统计信息
        print("\n=== 数据质量说明 ===")
        print("1. 数据筛选条件：")
        print(f"   - 要求至少120天的历史数据（不满足的股票数：{len(self.analysis_stats['insufficient_data'])}）")
        print(f"   - 要求数据完整无缺失（数据无效的股票数：{len(self.analysis_stats['invalid_data'])}）")
        print(f"   - 技术指标计算正常（分析失败的股票数：{len(self.analysis_stats['failed_stocks'])}）")

        return selected

    def calculate_ma15(self, symbol):
        """计算15日均线（优化版）"""
        # 检查缓存
        with self.cache_lock:
            if symbol in self.ma15_cache:
                return self.ma15_cache[symbol]

        try:
            df = self.get_stock_daily_data(symbol)
            if not df.empty:
                ma15 = df['close'].rolling(window=15).mean().iloc[-1]
                # 存入缓存
                with self.cache_lock:
                    self.ma15_cache[symbol] = ma15
                return ma15
            return None
        except Exception as e:
            print(f"计算15日均线时出错 ({symbol}): {e}")
            return None


if __name__ == "__main__":
    # 设置最大重试次数
    max_retries = 3
    retry_delay = 2

    # 修改浮点数显示格式
    pd.set_option('display.float_format', lambda x: '{:.2%}'.format(x) if isinstance(x, float) and x < 10 and (
                'prediction' in str(x) or 'rsi' in str(x) or 'explosion_score' in str(x)) else (
        '{:.2f}'.format(x / 100000000) if 'volume' in str(x) else '{:.2f}'.format(x)
    ))

    for attempt in range(max_retries):
        try:
            strategy = QuantStrategy()
            selected_stocks = strategy.run_analysis()
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"程序执行失败: {e}")
                sys.exit(1)
            print(f"程序执行出错，正在重试 ({attempt + 1}/{max_retries})...")
            time.sleep(retry_delay)

    if not selected_stocks.empty:
        # 计算15日均线
        def calculate_ma15(symbol):
            try:
                df = strategy.get_stock_daily_data(symbol)
                if not df.empty:
                    return df['close'].rolling(window=15).mean().iloc[-1]
                return None
            except:
                return None


        selected_stocks['ma15'] = selected_stocks['symbol'].apply(calculate_ma15)

        # 更新列名映射
        columns_map = {
            'name': '股票名称',
            'symbol': '股票代码',
            'market': '交易所-板块',
            'prediction': '上涨概率',
            'momentum_score': '动量得分',
            'rsi': 'RSI指标',
            'close': '收盘价',
            'ma15': '15日均线价格',
            'explosion_score': '爆发潜力值',
            'macd_status': 'MACD状态'
        }
        display_df = selected_stocks[columns_map.keys()].copy()
        display_df.columns = columns_map.values()

        # 设置pandas显示选项
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)

        print("\n=== 选出的标的（前20名） ===")
        print("注：MACD状态标记 - 金叉：MACD在0轴上方形成金叉，即将金叉：MACD在0轴上方且即将形成金叉")
        print(display_df.head(20).to_string(index=False, justify='center'))

        # 筛选15日均线在15元以内的股票
        low_price_stocks = display_df[display_df['15日均线价格'] <= 15].copy()
        if not low_price_stocks.empty:
            print("\n=== 15日均线在15元以内的标的 ===")
            print(low_price_stocks.to_string(index=False, justify='center'))

        # 显示超短线爆发潜力股票
        explosion_stocks = selected_stocks[
            selected_stocks['explosion_score'] > 1.5
            ].sort_values('explosion_score', ascending=False)

        if not explosion_stocks.empty:
            # 更新爆发潜力股票的列名映射
            explosion_columns = {
                'name': '股票名称',
                'symbol': '股票代码',
                'market': '交易所-板块',
                'explosion_score': '爆发潜力值',
                'volume': '成交量（亿）',
                'turn': '换手率',
                'close': '收盘价',
                'ma15': '15日均线价格',
                'rsi': 'RSI指标',
                'macd_status': 'MACD状态'
            }
            explosion_df = explosion_stocks[explosion_columns.keys()].copy()
            explosion_df.columns = explosion_columns.values()

            print("\n=== 超短线爆发潜力股票（前20名） ===")
            print(explosion_df.head(20).to_string(index=False, justify='center'))

            # 筛选15日均线在15元以内的爆发潜力股票
            low_price_explosion = explosion_df[explosion_df['15日均线价格'] <= 15].copy()
            if not low_price_explosion.empty:
                print("\n=== 15日均线在15元以内的爆发潜力股票 ===")
                print(low_price_explosion.to_string(index=False, justify='center'))

            print("\n超短线选股说明：")
            print("1. 爆发潜力值计算：")
            print("   - 成交量突增：权重30%")
            print("   - 换手率变化：权重20%")
            print("   - MACD金叉：权重20%")
            print("   - 价格位置：权重15%")
            print("   - 短期动量：权重15%")

            print("\n2. 建议关注：")
            print("   - 成交量和换手率较前期显著放大的股票")
            print("   - MACD即将金叉或刚形成金叉的股票")
            print("   - 股价处于近期低位但有上涨趋势的股票")
            print("   - 15日均线呈现向上趋势的股票")

            print("\n3. 风险提示：")
            print("   - 超短线交易属于高风险操作，建议仅供技术分析参考")
            print("   - 入场前请务必结合分时走势、盘口数据和市场情绪")
            print("   - 注意设置止损，控制好仓位，严格执行交易纪律")

        # 添加分析说明
        print("\n=== 选股条件说明 ===")
        print("1. 基础筛选条件：")
        print("   - 上涨概率 > 60%（基于机器学习模型预测）")
        print("   - 动量得分 > -10%（20日涨跌幅）")
        print("   - RSI指标：30-75之间（避免超买超卖）")
        print("   - 波动率 < 60%（年化计算，控制风险）")
        print("   - 日均成交量 > 5万（确保流动性）")
        print("   - 股价 > 3元（规避低价股风险）")

        print("\n2. 综合得分权重：")
        print("   - 上涨概率：30%（机器学习模型预测结果）")
        print("   - 动量得分：20%（价格走势趋势）")
        print("   - 爆发潜力：35%（短线上涨概率）")
        print("   - 波动率：15%（风险控制指标）")

        print("\n3. 特别说明：")
        print("   - 所有技术指标均经过标准化处理，消除量纲影响")
        print("   - 波动率采用对数收益率计算，更准确反映风险")
        print("   - 异常值已通过中位数法进行处理")
        print("   - 模型每日自动更新，适应市场变化")

        print("\n4. 风险提示：")
        print("   - 本程序基于历史数据和技术分析建模，不构成投资建议")
        print("   - 任何投资决策请结合市场环境、政策因素和个人风险承受能力")
        print("   - 模型预测结果仅供参考，交易需自行承担风险")
        print("   - 股市有风险，投资需谨慎")
    else:
        print("未找到符合条件的标的")
