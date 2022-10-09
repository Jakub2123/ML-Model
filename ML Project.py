#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import eli5 
from collections import Counter
import gc


# In[6]:


#Get Data
df = pd.read_hdf(r"C:\Users\Jakub\Desktop\python\data.h5")


# In[8]:


df_customers = (
    df[ ["price_total", "customer_id"] ]
    .groupby("customer_id")
    .agg("sum")
    .reset_index()
    .sort_values(by="price_total", ascending=False)
    .rename(columns={"price_total": "customer_price_total"})
)


df_customers["cumsum"] = df_customers["customer_price_total"].cumsum()
value_80prc = int(df["price_total"].sum() * 0.8)
df_customers["most_revenue_customer"] = df_customers["cumsum"] < value_80prc


top_customers = set(df_customers[ df_customers["most_revenue_customer"] ]["customer_id"].unique())

del df_customers
gc.collect()


# In[9]:


def feature_engineering(df):
    df_customers = (
        df
        .groupby("customer_id")
        .agg(
            count_orders=("order_id", lambda x: len(set(x))),
            count_unq_products=("product_id", lambda x: len(set(x))),
            sum_quantity=("quantity", np.sum),
            sum_price_unit=("price_unit", np.sum),
            sum_price_total=("price_total", np.sum),
            count_unq_countries=("country_id", lambda x: len(set(x))),
            prob_canceled=("is_canceled", np.mean)
        ).reset_index()
    )
    
    
    
    return df_customers


# In[10]:


def get_feats(df_customers, black_list=["most_revenue_customer"]):
    feats = list(df_customers.select_dtypes([np.number, bool]).columns)
    return [x for x in feats if x not in black_list]


# In[11]:


def get_X_y(df_customers, top_customers, feats):
    df_customers["most_revenue_customer"] = df_customers["customer_id"].map(lambda x: x in top_customers)
    
    X = df_customers[feats].values
    y = df_customers["most_revenue_customer"].values
    
    return X, y


# In[12]:


def train_and_get_scores(model, X, y, scoring="accuracy", cv=5):

    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    return np.mean(scores), np.std(scores)


# In[13]:


# 1st model

df_customers = feature_engineering(df)
feats = get_feats(df_customers)
X, y = get_X_y(df_customers, top_customers, feats)
model = DecisionTreeClassifier(max_depth=5)

train_and_get_scores(model, X, y)


# In[14]:


# 2 nd Model. Exclude Sum Price total
df_customers = feature_engineering(df)
feats = get_feats(df_customers, black_list=["most_revenue_customer", "sum_price_total"])
X, y = get_X_y(df_customers, top_customers, feats)
model = DecisionTreeClassifier(max_depth=5)

train_and_get_scores(model, X, y)


# In[15]:


# 3rd model using xgboost
df_customers = feature_engineering(df)
feats = get_feats(df_customers, black_list=["most_revenue_customer", "sum_price_total"])
X, y = get_X_y(df_customers, top_customers, feats)
model = xgb.XGBClassifier(max_depth=5, n_estimators=50, learning_rate=0.3)

train_and_get_scores(model, X, y)


# In[16]:


#Check feature importance

model.fit(X, y)

eli5.show_weights(model, feature_names=feats)


# In[17]:


def feature_engineering(df):
    
    def counter(vals):
        cntr = Counter()
        cntr.update(vals)
        return cntr
    
    df_customers = (
        df
        .groupby("customer_id")
        .agg(
            count_orders=("order_id", lambda x: len(set(x))),
            count_unq_products=("product_id", lambda x: len(set(x))),
            count_by_products=("product_id", lambda x:  counter(x) ),
            sum_quantity=("quantity", np.sum),
            sum_price_unit=("price_unit", np.sum),
            sum_price_total=("price_total", np.sum),
            count_unq_countries=("country_id", lambda x: len(set(x))),
            prob_canceled=("is_canceled", np.mean)
        ).reset_index()
    )
    
    
    return df_customers


# In[18]:


df_customers  = feature_engineering(df)
df_customers.head()


# In[21]:


df_customers["count_by_products"]

df_count_products = df_customers["count_by_products"].apply(pd.Series).fillna(-1)
df_count_products.columns = ["product_{}".format(x) for x in df_count_products.columns]

df_count_products.head(5)

# -1 = no order


# In[22]:


#Concatanate fields `df_count_products` & `df_customers` 
df_customers = pd.concat([df_customers, df_count_products], axis=1)
df_customers.shape


# In[23]:


#Train
feats = get_feats(df_customers, black_list=["most_revenue_customer", "sum_price_total"])
X, y = get_X_y(df_customers, top_customers, feats)
model = xgb.XGBClassifier(max_depth=5, n_estimators=50, learning_rate=0.3)

train_and_get_scores(model, X, y)


# In[24]:


model.fit(X, y)
eli5.show_weights(model, feature_names=feats, top=50)
#Prod_545
#Prod_99


# In[27]:


#CNT Clients Orders
customer_ids_by_product = set(df[ df["product_id"] == 545 ]["customer_id"].unique())
len(customer_ids_by_product )


# In[26]:


df_customers[ df_customers.customer_id.isin(customer_ids_by_product) ]["most_revenue_customer"].mean()


# In[31]:


df


# In[30]:


df[ df.product_id == 545 ]["price_unit"].value_counts()


# In[ ]:




