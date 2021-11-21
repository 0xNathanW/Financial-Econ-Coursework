import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./CleanData.csv")
depVar = df["Gearing (%)"]


def run_regression(*independentVars):
    """
    Runs a regression on the data.
    Prints summary

    """
    indVars = [var for var in independentVars]
    results = sm.OLS(endog=depVar, exog=df[indVars]).fit()

    print("\n\n", results.summary())

    fig = sm.graphics.plot_partregress_grid(results)
    fig.tight_layout(pad=1.0)
    plt.show()
    

run_regression(
    "Asset Tangibility", 
    "Log Sales", 
    "Profit margin (%)", 
    "Directors' Remuneration",
    "BvD Independence Indicator"
    )

    
