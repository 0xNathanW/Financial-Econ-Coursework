import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.regressionplots as smgr

df = pd.read_csv("./RegressionData.csv")


def run_regression(targetVar, influence=False, plot=False):
    """
    Runs a regression on the data.
    Prints summary

    """
    results = smf.ols(
        formula=f"""Gearing
        ~ {targetVar}
        + Size
        + Asset_Tangability
        + Growth_Opportunities
        + Profitability
        + Non_Debt_Tax_Shields""", data=df).fit()

    print("\n\n", results.summary())
    if plot: 
        fig = sm.graphics.plot_partregress_grid(results)
        fig.tight_layout(pad=1.0)
        plt.show()
    #print(results.get_influence().summary_table())
    if influence:
        smgr.influence_plot(results)
        plt.show()


run_regression("Ownership_Concentrated", plot=True)
