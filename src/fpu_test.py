from datetime import date, timedelta
import akshare as ak
import pandas as pd

from src.backtester import Backtester
from src.main import run_hedge_fund


def get_all_stocks():
    try:
        df = ak.stock_info_a_code_name()
        print(f"成功获取A股所有股票代码数据，共 {len(df)} 条记录\n")
        return df
    except Exception as e:
        print(f"获取A股所有股票代码数据时发生错误: {str(e)}")
        return pd.DataFrame()


if __name__ == "__main__":
    today = date.today()
    stocks = get_all_stocks()
    for _, stock in stocks.iterrows():
        ticker = stock['code']
        print(f"开始收集[{ticker}]{stock['name']}的数据...")
        df = ak.stock_zh_a_hist(symbol=ticker,
                                period="daily",
                                start_date=today.strftime("%Y%m%d"),
                                end_date=today.strftime("%Y%m%d"))
        if df.empty:
            print(f"未收集到[{ticker}]{stock['name']}的今日交易数据，跳过")
            continue
        if df.iloc[0]['收盘'] > 10:
            print(f"[{ticker}]{stock['name']}的今日收盘价格高于10元，跳过")
            continue
        
        print(df)

        # 创建回测器实例
        backtester = Backtester(
            agent=run_hedge_fund,
            ticker=ticker,
            start_date=(today - timedelta(days=90)).strftime('%Y-%m-%d'),
            end_date=today.isoformat(),
            initial_capital=10000,
            num_of_news=5
        )
        # 运行回测
        backtester.run_backtest()
        # 分析性能
        performance_df = backtester.analyze_performance()
