import statsmodels.api as sm
from statsmodels import regression
import numpy as np
import pandas as pd
import time 
from datetime import date
from jqdata import *


'''
================================================================================
总体回测前
================================================================================
'''
#总体回测前要做的事情
def initialize(context):
    set_params()        #1设置策参数
    set_variables()     #2设置中间变量
    set_backtest()      #3设置回测条件
    
#1
#设置策略参数
def set_params():
    g.Change_Frequnecy=20  # 调仓频率
    g.Text_Length=63  # 样本长度
    g.Own_Number=10   # 持仓数目
    g.Factor_Number=5  # 五因子模型 
    g.K_SMB = 0 # 打分中的SMB系数
    g.K_HMI = -7 # 打分中的HMI系数
    g.K_RMW = -256 # 打分中的RMW系数
    g.K_CMA = 233 # 打分中的CMA系数 
    g.K_Random = 1 # 打分中模拟随机的系数 
    g.Is_Regression = 1 # 是否是回归求系数
    g.Is_Average = 1 # 是否是等额买入
    
#2
#设置中间变量
def set_variables():
    g.Day_Clock=0               #记录连续回测天数
    g.Risk_Free=0.04           #无风险利率
    g.If_Trade=False    #当天是否交易 
    today=date.today()     #取当日时间
    a=get_all_trade_days() #取所有交易日
    today = max([i for i in range(0,len(a)) if a[i] < today ])
    g.Day_List = [a[i].isoformat() for i in range(0,today+1)] # 取出所有可以交易的天数
#3
#设置回测条件
def set_backtest():
    set_option('use_real_price', True) #用真实价格交易
    log.set_level('order', 'error') 




'''
================================================================================
每天开盘前
================================================================================
'''
#每天开盘前要做的事情
def before_trading_start(context):
    if g.Day_Clock%g.Change_Frequnecy==0: #每g.Change_Frequnecy天，交易一次行
        g.If_Trade=True
        set_slip_fee(context)  # 设置手续费
        # 设置可行股票池：获得当前开盘的沪深300股票池并剔除当前或者计算样本期间停牌的股票
        g.all_stocks = set_feasible_stocks(get_index_stocks('000300.XSHG'),g.Text_Length,context)
    g.Day_Clock+=1

#4 根据不同的时间段设置滑点与手续费
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(FixedSlippage(0)) 
    # 根据不同的时间段设置手续费
    dt=context.current_dt 
    if dt>datetime.datetime(2013,1, 1):
        set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5)) 
        
    elif dt>datetime.datetime(2011,1, 1):
        set_commission(PerTrade(buy_cost=0.001, sell_cost=0.002, min_cost=5))
            
    elif dt>datetime.datetime(2009,1, 1):
        set_commission(PerTrade(buy_cost=0.002, sell_cost=0.003, min_cost=5))
                
    else:
        set_commission(PerTrade(buy_cost=0.003, sell_cost=0.004, min_cost=5))


#5
# 设置可行股票池：
# 过滤掉当日停牌的股票,且筛选出前days天未停牌股票 
def set_feasible_stocks(stock_list,days,context):
    # 得到是否停牌信息的dataframe，停牌的1，未停牌得0
    suspened_info_df = get_price(list(stock_list), start_date=context.current_dt, end_date=context.current_dt, frequency='daily', fields='paused')['paused'].T
    # 得到当日未停牌股票的代码list:
    unsuspened_stocks = suspened_info_df[suspened_info_df.iloc[:,0]<1].index
    # 进一步，筛选出前days天未曾停牌的股票list:
    return [stock for stock in unsuspened_stocks if sum(attribute_history(stock, days, unit='1d',fields=('paused'),skip_paused=False))[0]==0]
  
'''
================================================================================
每天交易时
================================================================================
'''

#每天交易时要做的事情
def handle_data(context, data):
    if g.If_Trade==True: 
        todayStr=str(context.current_dt)[0:10]
        # 计算每个股票的打分情况
        Stock_List=Get_Scores(g.all_stocks,getDay(todayStr,-g.Text_Length),getDay(todayStr,-1),g.Risk_Free) 
        
        # 依打分排序，当前需要持仓的股票 
        Stock_Sort=Stock_List.sort_values('score') 
        if(g.Is_Regression == 1) :
            Stock_Sort = Stock_Sort[Stock_Sort.iloc[:,1]<0]
        # 把多余的股票去掉
        if(len(Stock_Sort) > g.Own_Number) : 
            Stock_Sort = Stock_Sort[:g.Own_Number]
        stock_sellOrbuy(context,Stock_Sort)       
    g.If_Trade=False


#6 交易股票
def stock_sellOrbuy(context,Stock_List):
    # 对于不需要持仓的股票，全仓卖出
    Stock_Code = Stock_List["code"]
    for stock in context.portfolio.positions:
        if stock not in Stock_Code: 
            order_target_value(stock, 0)
    # 买入之前没买过的
    Need_Buy = Stock_List[~Stock_List["code"].isin(context.portfolio.positions)]
    Length = len(Need_Buy["code"])
    if(Length == 0):
        return
    Need_Buy["score"] = Need_Buy["score"] / Need_Buy["score"].sum()
    Need_stock = list(Need_Buy["code"])
    if (g.Is_Average == 1) :
        Every_Money = [context.portfolio.total_value / Length] * Length
    else :
        Every_Money = list(Need_Buy["score"] * context.portfolio.total_value)
    for i in range(Length) :  
        order_target_value(Need_stock[i], Every_Money[i])

 

#7 这里我们算出分数情况
def Get_Scores (stocks,begin,end,Risk_Free):
    Length_Of_Stocks=len(stocks)
    q = query(
        valuation.code,
        valuation.market_cap,
        (balance.total_owner_equities/valuation.market_cap/100000000.0).label("BTM"),
        indicator.roe,
        balance.total_assets.label("Inv")
    ).filter(
        valuation.code.in_(stocks)
    )
    #查询相关因子的语句
    
    df = get_fundamentals(q,begin)
    
    #计算5因子再投资率的时候需要跟一年前的数据比较，所以单独取出计算
    ldf=get_fundamentals(q,getDay(begin,-252))
    # 若前一年的数据不存在，则暂且认为Inv=0
    if len(ldf)==0:
        ldf=df
    df["Inv"]=np.log(df["Inv"]/ldf["Inv"])  
    stocks = df["code"]
    Length_Of_Stocks=len(df["code"])
    #查询相关因子
    SMB_scores = df["market_cap"]
    HMI_scores = df["BTM"]
    RMW_scores = df["roe"]
    CMA_scores = df["Inv"]  
    # 选出特征股票组合
    S=df.sort_values('market_cap')['code'][:int(Length_Of_Stocks/3)]
    B=df.sort_values('market_cap')['code'][Length_Of_Stocks-int(Length_Of_Stocks/3):]
    L=df.sort_values('BTM')['code'][:int(Length_Of_Stocks/3)]
    H=df.sort_values('BTM')['code'][Length_Of_Stocks-int(Length_Of_Stocks/3):]
    W=df.sort_values('roe')['code'][:int(Length_Of_Stocks/3)]
    R=df.sort_values('roe')['code'][Length_Of_Stocks-int(Length_Of_Stocks/3):]
    C=df.sort_values('Inv')['code'][:int(Length_Of_Stocks/3)]
    A=df.sort_values('Inv')['code'][Length_Of_Stocks-int(Length_Of_Stocks/3):]
    
    # 获得样本期间的股票价格并计算日收益率
    df2 = get_price(list(stocks),begin,end,'1d')
    df3=df2['close'][:]
    df4=np.diff(np.log(df3),axis=0)+0*df3[1:]
    #求因子的值
    SMB=sum(df4[S].T)/len(S)-sum(df4[B].T)/len(B)
    HMI=sum(df4[H].T)/len(H)-sum(df4[L].T)/len(L)
    RMW=sum(df4[R].T)/len(R)-sum(df4[W].T)/len(W)
    CMA=sum(df4[C].T)/len(C)-sum(df4[A].T)/len(A)
    
    #用沪深300作为大盘基准 
    RM=diff(np.log(get_price('000300.XSHG',begin,end,'1d')['close']))-Risk_Free/252
    
    #将因子们计算好并且放好
    X=pd.DataFrame({"RM":RM,"SMB":SMB,"HMI":HMI,"RMW":RMW,"CMA":CMA})
    #取前g.Factor_Number个因子为策略因子
    factor_flag=["RM","SMB","HMI","RMW","CMA"][:g.Factor_Number] 
    X = X[factor_flag]
    X ['alpha'] = 1 
    final_scores = [0.0] * Length_Of_Stocks
    if (g.Is_Regression == 0) :
        for i in range(Length_Of_Stocks) :
            final_scores[i] = g.K_Random * np.random.rand() + g.K_SMB * SMB_scores[i] +g.K_HMI * HMI_scores[i] + g.K_RMW * RMW_scores[i] + g.K_CMA * CMA_scores[i] 
    else :
        for i in range(Length_Of_Stocks):
            final_scores[i] = np.linalg.lstsq(X,df4[stocks[i]]-Risk_Free/252, rcond=None)[0][g.Factor_Number]
    scores=pd.DataFrame({'code':stocks,'score':final_scores})
    return scores

 

#8 日期计算之获得某个日期之前或者之后dt个交易日的日期 
def getDay(dt, delta):
    idx = next(i for i in range(0, len(g.Day_List)) if dt <= g.Day_List[i])
    target_idx = max(idx + delta, 0)
    return g.Day_List[target_idx]




'''
================================================================================
每天收盘后
================================================================================
'''
#每天收盘后要做的事情
def after_trading_end(context):
    return
# 进行长运算（本模型中不需要）


