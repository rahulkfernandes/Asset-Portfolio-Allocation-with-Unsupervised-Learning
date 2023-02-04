import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def show_corr(data):
    #correlation
    correlation = data.corr()
    plt.figure(figsize=(15,10))
    plt.title('Correlation Matrix')
    sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
    plt.show()

def preprocessing(data):
    missing_fractions = data.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
    data.drop(labels=drop_list, axis=1, inplace=True)

    # Fill the missing values with the last value available in the dataset. 
    data=data.fillna(method='ffill')
    data = data.dropna(axis=0)

    # Daily Linear Returns (%)
    datareturns = data.pct_change(1)

    #Remove Outliers beyong 3 standard deviation
    datareturns = datareturns[datareturns.apply(lambda x :(x-x.mean()).abs()<(3*x.std()) ).all(1)]
    scaler = StandardScaler().fit(datareturns)
    rescaledDataset = pd.DataFrame(scaler.fit_transform(datareturns),columns = datareturns.columns, index = datareturns.index)
    # summarize transformed data
    datareturns.dropna(how='any', inplace=True)
    rescaledDataset.dropna(how='any', inplace=True)
    return rescaledDataset, datareturns

def show_returns(ticker, rescaledData):
    # Visualizing Log Returns for the DJIA 
    plt.figure(figsize=(16, 5))
    plt.title(f'{ticker} Return')
    plt.ylabel('Return')
    rescaledData.AAPL.plot()
    plt.grid(True);
    plt.legend()
    plt.show()

def exp_variance(model):
    NumEigenvalues=10
    fig, axes = plt.subplots(ncols=2, figsize=(14,4))
    Series1 = pd.Series(model.explained_variance_ratio_[:NumEigenvalues]).sort_values()*100
    Series2 = pd.Series(model.explained_variance_ratio_[:NumEigenvalues]).cumsum()*100
    Series1.plot.barh(ylim=(0,9), label="woohoo",title='Explained Variance Ratio by Top 10 factors',ax=axes[0]);
    Series2.plot(ylim=(0,100),xlim=(0,9),ax=axes[1], title='Cumulative Explained Variance by factor');
    # explained_variance
    # variance = pd.Series(np.cumsum(model.explained_variance_ratio_)).to_frame('Explained Variance').head(NumEigenvalues).style.format('{:,.2%}'.format)
    # print(variance)
    plt.show()

def PCWeights(model):
    # Principal Components (PC) weights for each 28 PCs
    weights_df = pd.DataFrame()

    for i in range(len(model.components_)):
        weights_df["weights_{}".format(i)] = model.components_[i] / sum(model.components_[i])

    weights_df = weights_df.values.T
    return weights_df

def eigen_portfolios(model, columns_names):
    NumComponents=5
        
    topPortfolios = pd.DataFrame(model.components_[:NumComponents], columns=columns_names)
    eigen_portfolios = topPortfolios.div(topPortfolios.sum(1), axis=0)
    eigen_portfolios.index = [f'Portfolio {i}' for i in range( NumComponents)]
    np.sqrt(model.explained_variance_)
    eigen_portfolios.T.plot.bar(subplots=True, layout=(int(NumComponents),1), figsize=(14,10), legend=False, sharey=True, ylim= (-1,1))
    plt.show()
    
    sns.heatmap(topPortfolios)
    plt.show() 

def sharpe_ratio(ts_returns, periods_per_year=252):
    # Sharpe ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk.
    # It calculares the annualized return, annualized volatility, and annualized sharpe ratio.
    # ts_returns are  returns of a signle eigen portfolio.
    
    n_years = ts_returns.shape[0]/periods_per_year
    annualized_return = np.power(np.prod(1+ts_returns),(1/n_years))-1
    annualized_vol = ts_returns.std() * np.sqrt(periods_per_year)
    annualized_sharpe = annualized_return / annualized_vol

    return annualized_return, annualized_vol, annualized_sharpe

def optimizedPortfolio(model, rescaledDataset, X_raw):
    n_portfolios = len(model.components_)
    annualized_ret = np.array([0.] * n_portfolios)
    sharpe_metric = np.array([0.] * n_portfolios)
    annualized_vol = np.array([0.] * n_portfolios)
    highest_sharpe = 0 
    stock_tickers = rescaledDataset.columns.values
    n_tickers = len(stock_tickers)
    pcs = model.components_
    
    for i in range(n_portfolios):
        
        pc_w = pcs[i] / sum(pcs[i])
        eigen_prtfi = pd.DataFrame(data ={'weights': pc_w.squeeze()*100}, index = stock_tickers)
        eigen_prtfi.sort_values(by=['weights'], ascending=False, inplace=True)
        eigen_prti_returns = np.dot(X_raw.loc[:, eigen_prtfi.index], pc_w)
        eigen_prti_returns = pd.Series(eigen_prti_returns.squeeze(), index=X_raw.index)
        er, vol, sharpe = sharpe_ratio(eigen_prti_returns)
        annualized_ret[i] = er
        annualized_vol[i] = vol
        sharpe_metric[i] = sharpe
        
        sharpe_metric= np.nan_to_num(sharpe_metric)
    highest_sharpe = np.argmax(sharpe_metric)

    print('Eigen portfolio #%d with the highest Sharpe. Return %.2f%%, vol = %.2f%%, Sharpe = %.2f' % 
          (highest_sharpe,
           annualized_ret[highest_sharpe]*100, 
           annualized_vol[highest_sharpe]*100, 
           sharpe_metric[highest_sharpe]))

    # fig, ax = plt.subplots()
    # fig.set_size_inches(12, 4)
    # ax.plot(sharpe_metric, linewidth=3)
    # ax.set_title('Sharpe ratio of eigen-portfolios')
    # ax.set_ylabel('Sharpe ratio')
    # ax.set_xlabel('Portfolios')

    # results = pd.DataFrame(data={'Return': annualized_ret, 'Vol': annualized_vol, 'Sharpe': sharpe_metric})
    # results.dropna(inplace=True)
    # results.sort_values(by=['Sharpe'], ascending=False, inplace=True)
    # print(results.head(20))
    # plt.show()
    return highest_sharpe

def plotEigen(weights, plot=False, tickers=[]):
    portfolio = pd.DataFrame(data ={'weights': weights.squeeze()*100}, index = tickers) 
    portfolio.sort_values(by=['weights'], ascending=False, inplace=True)
    if plot:
        print('Sum of weights of current eigen-portfolio: %.2f' % np.sum(portfolio))
        portfolio.plot(title='Current Eigen-Portfolio Weights', 
            figsize=(12,6), 
            xticks=range(0, len(stock_tickers),1), 
            rot=45, 
            linewidth=3
            )
        plt.show()
    return portfolio

def backtest(eigen, model, tickers):
    eigen_prtfi = pd.DataFrame(data ={'weights': eigen.squeeze()}, index = tickers)
    eigen_prtfi.sort_values(by=['weights'], ascending=False, inplace=True)    

    eigen_prti_returns = np.dot(X_test_raw.loc[:, eigen_prtfi.index], eigen)
    eigen_portfolio_returns = pd.Series(eigen_prti_returns.squeeze(), index=X_test_raw.index)
    returns, vol, sharpe = sharpe_ratio(eigen_portfolio_returns)
    print('=================')  
    print('Current Eigen-Portfolio:\nReturn = %.2f%%\nVolatility = %.2f%%\nSharpe = %.2f\n' % (returns*100, vol*100, sharpe))
    equal_weight_return=(X_test_raw * (1/len(model.components_))).sum(axis=1)    
    df_plot = pd.DataFrame({'EigenPorfolio Return': eigen_portfolio_returns, 'Equal Weight Index': equal_weight_return}, index=X_test.index)
    np.cumprod(df_plot + 1).plot(title='Returns of the equal weighted index vs. eigen-portfolio' , 
                          figsize=(12,6), linewidth=3)
    plt.show()
    

if __name__=="__main__":
    dataset = pd.read_csv('./datasets/Dow_adjcloses.csv',index_col=0)
    #show_corr(dataset) # Plot Correlation Matix
    rescaled, returns = preprocessing(dataset)
    #show_returns('AAPL',rescaled) # Plot rescaled returns
    
    percentage = int(len(rescaled) * 0.8)
    X_train = rescaled[:percentage]
    X_test = rescaled[percentage:]

    X_train_raw = returns[:percentage]
    X_test_raw = returns[percentage:]


    stock_tickers = rescaled.columns.values
    n_tickers = len(stock_tickers)

    pca = PCA()
    PrincipalComponent = pca.fit(X_train)

    #exp_variance(pca)

    weights = PCWeights(PrincipalComponent)
    #eigen_portfolios(pca, dataset.columns) # Plot Eigen Portolios
   
    pf_high_sharpe = optimizedPortfolio(PrincipalComponent, rescaled, X_train_raw)
    print(pf_high_sharpe)

    portfolio = plotEigen(weights=weights[pf_high_sharpe], plot=False, tickers=stock_tickers)
    print(portfolio)
    portfolio.plot.pie(y='weights', legend=False, figsize=(15, 10))
    plt.show()

    # Backtest(eigen=weights[5], model=PrincipalComponent, tickers=stock_tickers)
    # backtest(eigen=weights[1], model=PrincipalComponent, tickers=stock_tickers)
    # Backtest(eigen=weights[14], model=PrincipalComponent, tickers=stock_tickers)