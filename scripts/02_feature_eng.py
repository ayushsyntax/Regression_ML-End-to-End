#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
from category_encoders import TargetEncoder

# ============================
# 1. Load cleaned datasets
# ============================
train_df = pd.read_csv("data/processed/cleaning_train.csv")
eval_df = pd.read_csv("data/processed/cleaning_eval.csv")
holdout_df = pd.read_csv("data/processed/cleaning_holdout.csv")

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows

print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
print("Eval date range:", eval_df["date"].min(), "to", eval_df["date"].max())
print("Holdout date range:", holdout_df["date"].min(), "to", holdout_df["date"].max())

# Ensure datetime
train_df["date"] = pd.to_datetime(train_df["date"])
eval_df["date"] = pd.to_datetime(eval_df["date"])
holdout_df["date"] = pd.to_datetime(holdout_df["date"])


# In[35]:


train_df.head(2)


# In[20]:


# ============================
# 2. Date Features
# ============================
def add_date_features(df):
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month

    # Reorder columns
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "quarter", df.pop("quarter"))
    df.insert(3, "month", df.pop("month"))
    return df

train_df = add_date_features(train_df)
eval_df = add_date_features(eval_df)
holdout_df = add_date_features(holdout_df)


# In[21]:


print(train_df.shape)
train_df.head(1)


# In[22]:


print(eval_df.shape)
eval_df.head(1)


# In[23]:


print(holdout_df.shape)
holdout_df.head(1)


# The key rule:
#
# - Fit encoders/transformers on train only
#
# - Apply the learned mappings to eval

# 🎯 Why this matters
#
# - If we had fit the encoders/scalers on both train + eval together:
#
# - Eval would leak information into training.
#
# - Our metrics would look artificially good, because the model would unknowingly benefit from seeing the future.
#
# By strictly fitting on train and applying to eval:
#
# - Eval stays a true proxy for unseen future data.
#
# - The performance you see is realistic for when you deploy the model in the wild.

# In[37]:


train_df.head()


# In[24]:


# ============================
# 3. Frequency Encoding (zipcode)
# ============================
zip_counts = train_df["zipcode"].value_counts()

train_df["zipcode_freq"] = train_df["zipcode"].map(zip_counts)
eval_df["zipcode_freq"] = eval_df["zipcode"].map(zip_counts).fillna(0)
holdout_df["zipcode_freq"] = holdout_df["zipcode"].map(zip_counts).fillna(0)

print("Zip freq example (train):")
print(train_df[["zipcode", "zipcode_freq"]].head())


'''
🔍 Nuance:

- On train, we build the frequency dictionary (zip_counts).
- On eval, we never compute new counts → we only look up in the dictionary.
- If eval has an unseen zipcode, it gets NaN, which we replace with 0.
'''


# In[25]:


# ============================
# 4. Target Encoding (city_full)
# ============================
te = TargetEncoder(cols=["city_full"])

train_df["city_encoded"] = te.fit_transform(train_df["city_full"], train_df["price"])
eval_df["city_encoded"] = te.transform(eval_df["city_full"])
holdout_df["city_encoded"] = te.transform(holdout_df["city_full"])

print("City encoding example (train):")
print(train_df[["city_full", "city_encoded"]].head())

'''
🔍 Nuance:

- On train, we call fit_transform() → this computes the mapping from city → avg price using training targets.
- On eval, we only call transform() → it applies the train mapping. It never looks at eval’s price column.
'''


# In[26]:


# ============================
# 5. Drop unused columns
# ============================
# Drop leakage column "median_sale_price"
drop_cols = ["date", "city_full", "city", "zipcode", "median_sale_price"]
train_df.drop(columns=drop_cols, inplace=True)
eval_df.drop(columns=drop_cols, inplace=True)
holdout_df.drop(columns=drop_cols, inplace=True)


# In[27]:


print(train_df.shape)
train_df.head(1)


# In[28]:


print(eval_df.shape)
eval_df.head(1)


# In[29]:


# ============================
# 6. Save feature-engineered datasets
# ============================
train_df.to_csv("data/processed/fe_train.csv", index=False)
eval_df.to_csv("data/processed/fe_eval.csv", index=False)
holdout_df.to_csv("data/processed/fe_holdout.csv", index=False)

print("✅ Feature engineering complete.")
print("Train shape:", train_df.shape)
print("Eval shape:", eval_df.shape)
print("Holdout shape:", holdout_df.shape)


# - Fit frequency encoding on train only, apply to eval (with fillna(0) in case eval has unseen zipcodes).
#
# - Fit target encoding on train only, apply to eval with the same mapping.
#
# - Added a helper add_date_features so date feature logic isn’t duplicated.
#
# - Dropped unused columns consistently from both datasets
#
# - Droped high correlated column "median_sale_price"
#
# - Saved two separate outputs:
#
# - feature_engineered_train.csv
#
# - feature_engineered_eval.csv

# ✅ So the nuance is:
#
# - Train → fit transformations (learn rules from the past).
#
# - Eval → transform with those rules (apply them to future data).
#
# - Never re-fit on eval. That’s what keeps eval “unseen” and leakage-free.

# ## Multicolinearity

# to look at multicolinearity we can use:
# - VIF
# - Correlation matrix

# In[30]:


df = train_df


# In[31]:


import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def calculate_vif(df, target_col=None):
    """
    Calculate Variance Inflation Factor (VIF) for each numeric column in df.
    """
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64']).copy()

    # Drop target if provided
    if target_col and target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])

    # Add constant for statsmodels
    X = add_constant(numeric_df)

    # Compute VIF (skip the first column = constant)
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i + 1)  # skip constant at index 0
        for i in range(len(numeric_df.columns))
    ]

    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


# Look at VIF and correlation with original training dataset
df = pd.read_csv("data/raw/train.csv")
vif_table = calculate_vif(df, target_col="price")
print(vif_table)


# How to interpret
#
# - IF > 10 → serious multicollinearity.
#
# - VIF > 100 → catastrophic (drop/re-engineer immediately).
#
# - VIF = ∞ → perfect linear redundancy (drop one).

# In[32]:


offenders = ["Total Population", "Total Labor Force", "Total Families Below Poverty"]
corr_matrix = df[offenders].corr()
corr_matrix


# In[33]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric features (exclude the target 'price')
numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=["price"], errors="ignore")

# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Display full matrix as heatmap
plt.figure(figsize=(16,12))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    annot=False,   # set to True if you want numbers on cells
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Matrix of Numeric Features", fontsize=16)
plt.show()


# In[34]:


# Compute correlations with price
num_cols = df.select_dtypes("number").columns
corr_vec  = df[num_cols].corr(method="pearson")["price"].sort_values(ascending=False)
sns.set_theme(style="white")
sns.set(font_scale=1.1)
plt.figure(figsize=(6,10))
ax = sns.heatmap(
        corr_vec.to_frame(),
        annot=True, fmt=".2f",
        vmin=-1, vmax=1,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0,
        linewidths=.5, linecolor="white",
        cbar_kws={"shrink":0.8, "pad":0.02}
     )
ax.set_title("Price Pearson r", pad=20)
ax.set_ylabel("")
ax.set_xlabel("")
plt.tight_layout()
plt.show()


# In[ ]:




