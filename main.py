import pandas as pd
from eigen_portfolio import eigen
from hrp import mvp_hrp
import matplotlib.pyplot as plt


def plot_pf(hrp, eigen):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,10))
    ax1.pie(hrp['MVP'],);
    ax1.set_title('MVP', fontsize = 30)
    ax2.pie(eigen['weights'],);
    ax2.set_title('EIGEN', fontsize=30)
    ax3.pie(hrp['HRP']);
    ax3.set_title('HRP', fontsize = 30)
    plt.show()

if __name__=="__main__":
    dataset = pd.read_csv('./datasets/Dow_adjcloses.csv',index_col=0)

    eigen_pf = eigen(dataset)

    hrp_pf = mvp_hrp(dataset)

    plot_pf(hrp_pf, eigen_pf)    