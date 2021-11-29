import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.regressionplots as smgr
from statsmodels.stats.diagnostic import het_breuschpagan

df = pd.read_csv("./RegressionData.csv")

pd.options.display.width = 0
print("\n\n")
print(df.rename(columns=
{
    "Asset_Tangability": "Asset_Tang",
    "Growth_Opportunities": "Growth_Opp",
    "Non_Debt_Tax_Shields": "NDTS",
    "Ownership_Concentrated": "Owner_Conc"

}).describe().round(2))
print("\n\n")

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
        + Non_Debt_Tax_Shields""", data=df).fit(cov_type='HC1')

    print("\n\n", results.summary2())

    # Heteroskedasticity Breusch-Pagan Test on residuals.
    print("\n\n", het_breuschpagan(results.resid, results.model.exog))


    # F-test for significance of target variable.
    #print("\n\n", results.f_test(targetVar))

    if plot: 
        fig = sm.graphics.plot_partregress_grid(results)
        fig.tight_layout()
        plt.show()
    #print(results.get_influence().summary_table())
    if influence:
        smgr.influence_plot(results)
        plt.show()


run_regression("Ownership_Concentrated")
