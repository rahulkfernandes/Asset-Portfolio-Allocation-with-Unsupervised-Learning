import numpy as np
import pandas as pd
import cvxopt as opt
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

import warnings
warnings.filterwarnings('ignore')

def preprocess(data):
    missing_fractions = data.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
    data.drop(labels=drop_list, axis=1, inplace=True)
    data=data.fillna(method='ffill')
    return data

def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist = ((1 - corr) / 2.)**.5
    return dist

def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()

def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems]
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar

def getMVP(cov):
    cov = cov.T.values
    n = len(cov)
    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(cov)
    pbar = opt.matrix(np.ones(cov.shape[0]))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    opt.solvers.options['show_progress'] = False
    portfolios = [opt.solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER    
    returns = [opt.blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(opt.blas.dot(x, S * x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO    
    wt = opt.solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']

    return list(wt)

def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getHRP(cov, corr):
    # Construct a hierarchical portfolio
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    # plt.figure(figsize=(20, 10))
    # dn = sch.dendrogram(link, labels=cov.index.values)
    # plt.show()
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()

def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), 
            (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w

def get_all_portfolios(returns):
    cov, corr = returns.cov(), returns.corr()
    hrp = getHRP(cov, corr)
    mvp = getMVP(cov)
    
    mvp_df = pd.DataFrame({'Ticker': cov.index, 'MVP': mvp})
    hrp_df = pd.DataFrame({'Ticker': hrp.index, 'HRP': hrp.values})
    portfolios = pd.merge(mvp_df, hrp_df, how='outer', on='Ticker')
    portfolios['HRP'] = portfolios['HRP'].apply(lambda x: x * 100)
    portfolios['MVP'] = portfolios['MVP'].apply(lambda x: x * 100) 
    return portfolios

def plot_pf(pf):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,10))
    ax1.pie(pf.iloc[:,0], );
    ax1.set_title('MVP',fontsize = 30)
    ax2.pie(pf.iloc[:,1]);
    ax2.set_title('HRP',fontsize = 30)
    plt.show()

def compare(pf,returns, returns_test):
    bar_width = 0.25
    x_axis = np.arange(len(pf['Ticker']))
    plt.figure(figsize=(15,6))
    plt.bar(x_axis-bar_width, pf['MVP'], width=bar_width*2)
    plt.bar(x_axis+bar_width, pf['HRP'], width=bar_width*2)
    plt.xticks(x_axis, pf['Ticker'])
    plt.title('Current MVP and HRP Portfolio Weights')
    plt.legend(['MVP', 'HRP'])

    print('Backtesting HRP portfolio against MVP portfolio...')
    pf['Equal_Weight'] = np.full((len(pf['Ticker'])), 100/len(pf['Ticker']))
    Insample_Result=pd.DataFrame(
        np.dot(returns,np.array(pf.drop(['Ticker'],axis=1))),
        columns=['MVP', 'HRP', 'Equal_Weight'],
        index = returns.index
        )
    OutOfSample_Result=pd.DataFrame(
        np.dot(returns_test,np.array(pf.drop(['Ticker'],axis=1))),
        columns=['MVP', 'HRP', 'Equal_Weight'],
        index = returns_test.index
        )
    Insample_Result.cumsum().plot(
        figsize=(12, 6),
        title ="MVP vs HRP In-Sample Results"
        )
    OutOfSample_Result.cumsum().plot(
        figsize=(12, 6),
        title ="MVP vs HRP Out Of Sample Results"
        )

    stddev = Insample_Result.std() * np.sqrt(252)
    sharp_ratio = (Insample_Result.mean()*np.sqrt(252))/(Insample_Result).std()
    Results = pd.DataFrame(dict(stdev=stddev, sharp_ratio = sharp_ratio))
    print('In Sample Results:')
    print(Results)
    stddev_oos = OutOfSample_Result.std() * np.sqrt(252)
    sharp_ratio_oos = (OutOfSample_Result.mean()*np.sqrt(252))/(OutOfSample_Result).std()
    Results_oos = pd.DataFrame(dict(stdev_oos=stddev_oos,sharp_ratio_oos=sharp_ratio_oos))
    print('Out of Sample Results:')
    print(Results_oos)
    plt.show()

def mvp_hrp(dataset, backtest=False):
    processed_data = preprocess(dataset)
    if backtest == True:
        row = len(processed_data)
        train_len = int(row * 0.8)
        X_train = processed_data.head(train_len)
        X_test = processed_data.tail(row-train_len)
        returns = X_train.pct_change().dropna()
        returns_test = X_test.pct_change().dropna()

        portfolios = get_all_portfolios(returns)
        compare(portfolios, returns, returns_test)
    else:
        returns_all = processed_data.pct_change().dropna()
        portfolios = get_all_portfolios(returns_all)
        print('MVP and HRP Portfolios Created!')
    return portfolios