import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./RegressionData.csv")
df = add_constant(df)
depVar = df["Gearing (%)2018"]


def run_regression(interestVar):
    """
    Runs a regression on the data.
    Prints summary

    """
    # Add a constant to the data
    indVars = [
        "const",
        interestVar,
        "Asset Tangability Avg",
        "Log Sales Avg",
        "TobinsQ Avg",
        "Profitability Avg",
        "NDTS Avg",
    ]
    results = sm.OLS(endog=depVar, exog=df[indVars]).fit()

    print("\n\n", results.summary())

    fig = sm.graphics.plot_partregress_grid(results)
    fig.tight_layout(pad=1.0)
    plt.show()
    

run_regression("Directors' Fees Avg")

run_regression("Directors Remuneration Avg")

run_regression("Highest Paid Director Avg")    
