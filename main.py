import sys
import numpy as np
import pandas as pd
import seaborn as sns
from eigen import eigen
from hrp import mvp_hrp
from nco import nco
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials

YEARS = 3
DOW = [
    'UNH', 'GS', 'HD', 'MSFT', 'MCD',
    'CAT', 'AMGN', 'V', 'BA', 'HON', 'TRV',
    'AXP', 'CRM', 'CVX', 'JPM', 'PG', 'IBM',
    'NKE', 'MMM', 'DIS', 'MRK', 'KO', 'DOW',
    'CSCO', 'VZ', 'WBA', 'INTC'
    ]

def show_corr(data):
    correlation = data.corr()
    plt.figure(figsize=(15,10))
    plt.title('Correlation Matrix')
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
    plt.show()

def get_data(tickers, start_date, end_date):
    print('Pulling Data from Yahoo....')
    yahoo_financials = YahooFinancials(tickers)
    price_data = yahoo_financials.get_historical_price_data(
        start_date,
        end_date, 
        'daily'
        )
    ticker_list = price_data.keys() 
    
    all_comps = pd.DataFrame()
    for ticker in ticker_list:
        df = pd.DataFrame(price_data[ticker]['prices'])
        df = df[['adjclose']].set_index(df['formatted_date'])
        df.rename(columns={'adjclose': ticker}, inplace=True)
        all_comps = pd.concat([all_comps, df], axis=1)
    
    return all_comps

def plot_pf(df):
    if False in (np.array(df['Eigen']>0)):
        print('Eigen Portfolio as negative values')
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(15,10))
        ax1.pie(df['MVP'], labels=df['Ticker']);
        ax1.set_title('MVP', fontsize = 20)
        ax2.pie(df['HRP'], labels=df['Ticker']);
        ax2.set_title('HRP', fontsize = 20)
        ax3.pie(df['HRP_NCO'], labels=df['Ticker'])
        ax3.set_title('NCO-HRP', fontsize = 20)
        ax4.pie(df['MVP_NCO'], labels=df['Ticker'])
        ax4.set_title('NCO-MVP', fontsize = 20)
    else:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5,figsize=(15,10))
        ax1.pie(df['MVP'], labels=df['Ticker']);
        ax1.set_title('MVP', fontsize = 20)
        ax2.pie(df['Eigen'], labels=df['Ticker']);
        ax2.set_title('EIGEN', fontsize=20)
        ax3.pie(df['HRP'], labels=df['Ticker']);
        ax3.set_title('HRP', fontsize = 20)
        ax4.pie(df['HRP_NCO'], labels=df['Ticker'])
        ax4.set_title('NCO-HRP', fontsize = 20)
        ax5.pie(df['MVP_NCO'], labels=df['Ticker'])
        ax5.set_title('NCO-MVPcl', fontsize = 20)

    plt.show()

if __name__=="__main__":
    # dataset = pd.read_csv('./datasets/Dow_adjcloses.csv', index_col=0)
    end_epoch = datetime.now()
    start_epoch = end_epoch - timedelta(days=YEARS*365)
    end_date = end_epoch.strftime('%Y-%m-%d')
    start_date = start_epoch.strftime('%Y-%m-%d')
    
    dataset = get_data(DOW, start_date, end_date)
    if dataset.isna().values.any():
        print('Dataset contains nan values!')
    #show_corr(dataset)
    
    if len(sys.argv) == 2:
        if sys.argv[1] == 'backtest':
            backtest = True
        else: 
            print("Enter 'python main.py backtest' to run in backtest mode")
        print('RUNNING IN BACKTEST MODE')
        eigen_pf = eigen(dataset, backtest)
        hrp_pf = mvp_hrp(dataset, backtest)
        nco_pf = nco(dataset, backtest)
        
    else:
        eigen_pf = eigen(dataset)
        hrp_pf = mvp_hrp(dataset)
        nco_pf = nco(dataset)

    all_pf = pd.merge(eigen_pf, hrp_pf, how='outer', on='Ticker')
    all_pf = pd.merge(all_pf, nco_pf, how='outer', on='Ticker')
    print('\n===============All Portfolios==============')
    print(all_pf)
    plot_pf(all_pf)   