import pandas as pd
from eigen_portfolio import eigen
import matplotlib.pyplot as plt

dataset = pd.read_csv('./datasets/Dow_adjcloses.csv',index_col=0)

eigen_pf = eigen(dataset)
eigen_pf.plot.pie(y='weights', legend=False, figsize=(15, 10))
plt.show()