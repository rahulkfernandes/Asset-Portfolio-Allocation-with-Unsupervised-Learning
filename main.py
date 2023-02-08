import sys
import numpy as np
import pandas as pd
import seaborn as sns
from hrp import mvp_hrp
import matplotlib.pyplot as plt
from eigen_portfolio import eigen

def show_corr(data):
    correlation = data.corr()
    plt.figure(figsize=(15,10))
    plt.title('Correlation Matrix')
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
    plt.show()

def plot_pf(df):
    if False in (np.array(df['Eigen']>0)):
        print('Eigen Portfolio as negative values')
        fig, (ax1, ax3) = plt.subplots(1, 2,figsize=(15,10))
        ax1.pie(df['MVP'], labels=df['Ticker']);
        ax1.set_title('MVP', fontsize = 20)
        ax3.pie(df['HRP'], labels=df['Ticker']);
        ax3.set_title('HRP', fontsize = 20)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,10))
        ax1.pie(df['MVP'], labels=df['Ticker']);
        ax1.set_title('MVP', fontsize = 20)
        ax2.pie(df['Eigen'], labels=df['Ticker']);
        ax2.set_title('EIGEN', fontsize=20)
        ax3.pie(df['HRP'], labels=df['Ticker']);
        ax3.set_title('HRP', fontsize = 20)
    plt.show()

if __name__=="__main__":
    dataset = pd.read_csv('./datasets/Dow_adjcloses.csv',index_col=0)
    # show_corr(dataset)

    if len(sys.argv) == 2:
        if sys.argv[1] == 'backtest':
            backtest = True
        else: 
            print("Enter 'python main.py backtest' to run in backtest mode")
        print('RUNNING IN BACKTEST MODE')
        eigen_pf = eigen(dataset, backtest)
        hrp_pf = mvp_hrp(dataset, backtest)
        
    else:
        eigen_pf = eigen(dataset)
        hrp_pf = mvp_hrp(dataset)

    all_pf = pd.merge(eigen_pf, hrp_pf, how='outer', on='Ticker')
    print('\n===============All Portfolios==============')
    print(all_pf)
    plot_pf(all_pf)    