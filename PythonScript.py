import numpy as np #deal with arrays in python
import pandas as pd #python data frames
import statsmodels.api as sm  # GLM
import statsmodels.formula.api as smf  # formula based GLM
import matplotlib.pyplot as plt #standard pythonic approach for plots
import seaborn as sns #nice library for plots

msdata = pd.read_csv(
    "glms_meyershi.csv",
    dtype = {
        'acc_year': int,
        'dev_year': int,
        'incremental': float,
        'cumulative': float},
        delimiter=';'
    )
print(msdata)

# Drop the first column ('team')
msdata.drop(columns=msdata.columns[0], axis=1, inplace=True)

fig, axs = plt.subplots(1, 2)
plt.subplots_adjust(wspace=1)
g=sns.lineplot(x='dev_year',y='cumulative',data=msdata,hue='acc_year',palette=sns.color_palette("deep"), ax=axs[0])
g.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 6)
g.set_title("Cumulative payments")

g1=sns.lineplot(x='dev_year',y='incremental',data=msdata,hue='acc_year',palette=sns.color_palette("deep"), ax=axs[1])
g1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 6)
g1.set_title("Incremental payments")

plt.show()

#  add some additional variables to the data set

cat_type = pd.api.types.CategoricalDtype(categories=list(range(1,11)), ordered=True)

msdata["acc_year_factor"] = msdata["acc_year"].astype(cat_type)
msdata["dev_year_factor"] = msdata["dev_year"].astype(cat_type)
msdata["cal_year"] = msdata["acc_year"] + msdata["dev_year"] - 1  # subtract 1 so that first cal_year is 1 not 2

print(msdata.head(6))
print(msdata.dtypes)


######################################################################
####################### CHAIN LADDER MODEL ###########################
######################################################################

# The specific model that replicates the chain ladder result is the Over-dispersed Poisson (ODP) cross classified (cc) model 
#The cross-classified model requires separate levels for each of accident and development year 
# so we use the categorical versions of these variates.

######################### FITTING THE MODEL #########################

glm_fit1 = smf.glm('incremental~ -1 + acc_year_factor + dev_year_factor',
                   data = msdata,
                   family = sm.families.Poisson() ).fit(scale='X2')
print(glm_fit1.summary())

# write a function to produce a data set from a vectors of acc_year and dev_year variables
# The function below takes in numpy 1-d arrays of acc and dev and the categorisation we created above 
# and produces a pandas dataframe with the various covariates that we have created so far.

