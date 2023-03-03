import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import linkage, ClusterWarning
import matplotlib.pyplot as plt
import cvxopt as opt

from warnings import simplefilter
simplefilter('ignore', ClusterWarning)

def preprocess(data):
    missing_fractions = data.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
    data.drop(labels=drop_list, axis=1, inplace=True)
    data=data.fillna(method='ffill')
    return data

def cov2corr(cov):
    # Derive correlation matrix from covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std,std)
    corr[corr<-1], corr[corr>1] = -1, 1 # numerical error
    return corr

def getPCA(matrix):
    # Get eVal, eVec from a Hermitian matrix
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1] # args for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:,indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec

def mpPDF(var, q, pts):
    # Marcenko- Pastur pdf
    # q=T/N
    eMin, eMax = var * (1-(1./q)**.5)**2, var*(1+(1./q)**.5)**2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf2 = pd.Series(pdf.reshape(pdf.shape[0],), index=eVal.reshape(eVal.shape[0],))
    return pdf2

def fitKDE(obs, bWidth=.25, kernel='gaussian', x=None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the arraymof values on which the fit KDE will be evaluated
    if len(obs.shape)==1: obs=obs.reshape(-1,1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    if x is None: x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1: x=x.reshape(-1,1)
    logProb = kde.score_samples(x) # log (density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf

def errPDFs(var, eVal, q, bWidth, pts=1000):
    # Fit error
    pdf0 = mpPDF(var, q, pts) # theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values) # empirical pdf
    sse = np.sum((pdf1-pdf0) ** 2)
    return sse 

def findMaxEval(eVal, q, bWidth):
    # Find max random eVal by fitting Marcenko's dist
    out = minimize(
        lambda *x: errPDFs(*x), .5,args=(eVal, q, bWidth),
        bounds=((1E-5, 1-1E-5),)
    )
    if out['success']: var = out['x'][0]
    else: var = 1
    eMax = var * (1+(1./q)) ** 2
    return eMax, var

def denoisedCorr(eVal, eVec, nFacts):
    # Remove noise from corr by fixing random eigenvalues
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:]=eVal_[nFacts].sum()/float(eVal_.shape[0]-nFacts)
    eVal_ = np.diag(eVal_)
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)
    corr1 = cov2corr(corr1)
    return corr1

def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov
 
def deNoiseCov(cov0, q, bWidth):
    corr0=cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth)
    nFacts0 = eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1 = denoisedCorr(eVal0, eVec0, nFacts0)
    cov1 = corr2cov(corr1, np.diag(cov0) ** .5)
    return cov1

def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10):
    x = ((1-corr0.fillna(0))/2.)**.5
    silh = pd.Series(dtype=float) # observation matrix
    for init in range(n_init):
        for i in range(2, maxNumClusters+1):
            kmeans_= KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
            stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
            if np.isnan(stat[1]) or stat[0]>stat[1]:
                silh, kmeans = silh_, kmeans_
    newIdx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[newIdx] # reorder rows

    corr1=corr1.iloc[:,newIdx] # reorder columns
    clstrs = {i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist()\
              for i in np.unique(kmeans.labels_)
            } # cluster members
    silh = pd.Series(silh, index=x.index)
    return corr1, clstrs, silh

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
        sortIx =pd.concat([sortIx,df0])  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()

def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems]
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar

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

def HRP(cov, corr):
    # Construct a hierarchical portfolio
    dist = correlDist(corr)
    link = linkage(dist, 'ward', metric='euclidean')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    hrp = getRecBipart(cov, sortIx)
    return hrp

def MVP(cov, mu=None):
    cov = pd.DataFrame(cov)
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

    return np.array(list(wt))

def HRP_nco(cov, mu=None, maxNumClusters=None):
    cov = pd.DataFrame(cov)
    if mu is not None: mu = pd.Series(mu[:,0])
    corr1 = cov2corr(cov)
    corr1, clstrs,_ = clusterKMeansBase(corr1, maxNumClusters, n_init=10)
    
    wIntra = pd.DataFrame(0, index = cov.index, columns=clstrs.keys())
    for i in clstrs:
        cov_ = cov.loc[clstrs[i], clstrs[i]]
        if mu is None: mu_=None
        else: mu_ = mu.loc[clstrs[i]].values.reshape(-1,1)
        wIntra.loc[clstrs[i], i] = HRP(cov_, cov2corr(cov_))
    cov_ = wIntra.T.dot(np.dot(cov, wIntra)) # reduce covariance matrix
    mu_ = (None if mu is None else wIntra.T.dot(mu))
    wInter = pd.Series(HRP(cov_, cov2corr(cov_)), index=cov_.index)
    nco = wIntra.mul(wInter, axis=1).sum(axis=1)
    return nco

def MVP_nco(cov, mu=None, maxNumClusters=None):
    cov = pd.DataFrame(cov)
    if mu is not None: mu = pd.Series(mu[:,0])
    corr1 = cov2corr(cov)
    corr1, clstrs,_ = clusterKMeansBase(corr1, maxNumClusters, n_init=10)
    
    wIntra = pd.DataFrame(0, index = cov.index, columns=clstrs.keys())
    for i in clstrs:
        cov_ = cov.loc[clstrs[i], clstrs[i]].values
        if mu is None: mu_=None
        else: mu_ = mu.loc[clstrs[i]].values.reshape(-1,1)
        wIntra.loc[clstrs[i],i] = MVP(cov_, mu_).flatten()
    cov_ = wIntra.T.dot(np.dot(cov, wIntra)) # reduce covariance matrix
    mu_ = (None if mu is None else wIntra.T.dot(mu))
    wInter = pd.Series(MVP(cov_,mu_).flatten(), index=cov_.index)
    nco = wIntra.mul(wInter, axis=1).sum(axis=1)
    return nco

def getNCO(returns):
    cov0 = returns.cov()
    cols = cov0.columns
    q = returns.shape[0]/returns.shape[1]
    cov1 = deNoiseCov(cov0, q, bWidth=0.1)
    cov1 = pd.DataFrame(cov1, index=cols, columns=cols)

    HRP_pf = HRP_nco(cov1, mu=None, maxNumClusters=int(cov1.shape[0]/2))
    MVP_pf = MVP_nco(cov1, mu=None, maxNumClusters=int(cov1.shape[0]/2))
    pf = pd.DataFrame({'HRP_NCO': HRP_pf.values, 'MVP_NCO': MVP_pf.values}, index=HRP_pf.index)
    return pf

def compare(pf, returns, returns_test):
    plt.figure(figsize=(15,6))
    plt.bar(pf['Ticker'], pf['HRP_NCO'])
    plt.title('Current NCO-HRP Portfolio Weights')

    print('\nBacktesting NCO-HRP portfolio...')
    pf['Equal_Weight'] = np.full((len(pf['Ticker'])), 100/len(pf['Ticker']))
    Insample_Result=pd.DataFrame(
        np.dot(returns,np.array(pf.drop(['Ticker'],axis=1))),
        columns=['HRP_NCO', 'MVP_NCO', 'Equal_Weight'],
        index = returns.index
        )
    OutOfSample_Result=pd.DataFrame(
        np.dot(returns_test,np.array(pf.drop(['Ticker'],axis=1))),
        columns=['HRP_NCO','MVP_NCO' ,'Equal_Weight'],
        index = returns_test.index
        )
    Insample_Result.cumsum().plot(
        figsize=(12, 6),
        title ="NCO-HRP vs Equal Weight In-Sample Results"
        )
    OutOfSample_Result.cumsum().plot(
        figsize=(12, 6),
        title ="NCO-HRP vs Equal Weight Out Of Sample Results"
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

def sharpe_ratio(ts_returns, periods_per_year=252):
    n_years = ts_returns.shape[0]/periods_per_year
    annualized_return = np.power(np.prod(1+ts_returns),(1/n_years))-1
    annualized_vol = ts_returns.std() * np.sqrt(periods_per_year)
    annualized_sharpe = annualized_return / annualized_vol
    return annualized_sharpe

def nco(dataset, backtest=False):
    dataset = preprocess(dataset)

    if backtest == True:
        row = len(dataset)
        train_len = int(row * 0.8)
        X_train = dataset.head(train_len)
        X_test = dataset.tail(row-train_len)
        returns = X_train.pct_change().dropna()
        returns_test = X_test.pct_change().dropna()

        portfolio = getNCO(returns)
        portfolio['HRP_NCO'] = portfolio['HRP_NCO'].apply(lambda x: x*100)
        portfolio['MVP_NCO'] = portfolio['MVP_NCO'].apply(lambda x: x*100)

        portfolio.reset_index(inplace=True)
        portfolio.rename(columns = {'index':'Ticker'},inplace=True)
        compare(portfolio, returns, returns_test)
        portfolio.drop(['Equal_Weight'], axis=1, inplace=True)
    else:
        returns = dataset.pct_change().dropna()

        portfolio = getNCO(returns)
        hrp_returns = np.dot(returns, portfolio['HRP_NCO'].values)
        hrp_sharpe = sharpe_ratio(hrp_returns)
        mvp_returns = np.dot(returns, portfolio['MVP_NCO'].values)
        mvp_sharpe =  sharpe_ratio(mvp_returns)
        portfolio['HRP_NCO'] = portfolio['HRP_NCO'].apply(lambda x: x*100)
        portfolio['MVP_NCO'] = portfolio['MVP_NCO'].apply(lambda x: x*100)

        portfolio.reset_index(inplace=True)
        portfolio.rename(columns = {'index':'Ticker'},inplace=True)

        print('NCO-HRP Sharpe Ratio =', hrp_sharpe)
        print('NCO-MVP Sharpe Ratio =', mvp_sharpe)
    return portfolio