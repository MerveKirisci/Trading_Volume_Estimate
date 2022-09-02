##########################################
# Üye İş yerlerinden Beklenilen İşlem Hacmi Tahmini
##########################################

# İş Problemi

# Iyzico internetten alışveriş deneyimini hem alıcılar hem de satıcılar için kolaylaştıran bir
# finansal teknolojiler şirketidir. E-ticaret firmaları, pazaryerleri ve bireysel kullanıcılar
# için ödeme altyapısı sağlamaktadır. 2021 yılının ilk 3 ayı için merchant_id ve gün bazında toplam
# işlem hacmi tahmini yapılması beklenmekte.


# Import işlemleri
import re
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings('ignore')

# Görev 1: Keşifçi Veri Analizi

# Adım 1: Iyzico_data.csv dosyasını okutunuz. transaction_date değişkeninin tipini date'e çeviriniz.

df = pd.read_csv("datasets/iyzico_data.csv", index_col=0)

def check_df(df, head=5):
    print("##################### Shape #####################")
    print(df.shape)

    print("##################### Types #####################")
    print(df.dtypes)

    print("##################### Head #####################")
    print(df.head(head))

    print("##################### Tail #####################")
    print(df.tail(head))

    print("##################### is null? #####################")
    print(df.isnull().sum())

    print("##################### Quantiles #####################")
    print(df.quantile([0, 0.25, 0.50, 0.75, 0.99, 1]).T)
    print(df.describe().T)

check_df(df)

# transaction_date değişkeninin tipi object fakat bizim date tipine dönüştürmemiz gerekiyor.

df["transaction_date"] = pd.to_datetime(df["transaction_date"])

df["transaction_date"].dtypes

# Adım 2: Veri setinin başlangıc ve bitiş tarihleri nedir?

df["transaction_date"].min(), df["transaction_date"].max()

# başlangıç: Timestamp('2018-01-01 00:00:00')
# bitiş: Timestamp('2020-12-31 00:00:00')

# Adım 3: Her üye iş yerindeki toplam işlem sayısı kaçtır?
df["merchant_id"].nunique()
# 7

df.groupby("merchant_id").agg({"Total_Transaction":"sum"}).sort_values("Total_Transaction",ascending=False)
#              Total_Transaction
# merchant_id
# 124381                 1935357
# 46774                  1599559
# 535                    1302725
# 57192                  1146440
# 42616                  1126191
# 86302                   840951
# 129316                  440029


# Adım 4: Her üye iş yerindeki toplam ödeme miktarı kaçtır?

df.groupby("merchant_id").agg({"Total_Paid":"sum"}).sort_values("Total_Paid",ascending=False)

#                 Total_Paid
# merchant_id
# 46774       1567200341.286
# 124381      1158692543.973
# 42616        354583091.808
# 57192        317337137.586
# 535          156601530.234
# 86302          2870446.716
# 129316         1555471.476

# Adım 5: Her üye iş yerininin her bir yıl içerisindeki transaction count grafiklerini gözlemleyiniz.

for id in df.merchant_id.unique():
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1, title = str(id) + ' 2018-2019 Transaction Count')
    df[(df.merchant_id == id) &( df.transaction_date >= "2018-01-01" )& (df.transaction_date < "2019-01-01")]["Total_Transaction"].plot()
    plt.xlabel('')
    plt.subplot(3, 1, 2,title = str(id) + ' 2019-2020 Transaction Count')
    df[(df.merchant_id == id) &( df.transaction_date >= "2019-01-01" )& (df.transaction_date < "2020-01-01")]["Total_Transaction"].plot()
    plt.xlabel('')
    plt.show()

# Görev 2 : Feature Engineering tekniklerini uygulayanız. Yeni feature'lar türetiniz.

# • Date Features

df.head()

def create_date_features(df, date_column):
    df['month'] = df[date_column].dt.month
    df['day_of_month'] = df[date_column].dt.day
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.weekofyear
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['year'] = df[date_column].dt.year
    df["is_wknd"] = df[date_column].dt.weekday // 4
    df['is_month_start'] =df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['quarter'] = df[date_column].dt.quarter
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    return df

df = create_date_features(df, "transaction_date")
df.head()

# Üye iş yerlerinin yıl ve ay bazında işlem sayılarının incelenmesi
df.groupby(["merchant_id","year"]).agg({"Total_Transaction": ["sum", "mean", "median", "std",]}).head(20)

# Üye iş yerlerinin yıl ve ay bazında toplam ödeme miktarlarının incelenmesi
df.groupby(["merchant_id","year","month"]).agg({"Total_Paid": ["sum", "mean", "median", "std",]}).head(20)


# Random Noise

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

# • Lag/Shifted Features

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91,92,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,
                       350,351,352,352,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,
                       538,539,540,541,542,
                       718,719,720,721,722])

check_df(df)

# • Rolling Mean Features


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby("merchant_id")['Total_Transaction']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [91,92,178,179,180,181,182,359,360,361,449,450,451,539,540,541,629,630,631,720])


# • Exponentially Weighted Mean Features

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby("merchant_id")['Total_Transaction'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91,92,178,179,180,181,182,359,360,361,449,450,451,539,540,541,629,630,631,720]

df = ewm_features(df, alphas, lags)

check_df(df)

# • Özel günler, döviz kuru vb.

# Black Friday veya yazın gelmesi gibi sezonlar satış hacmi problemlerinde önemli uç noktalar gösteren dönemlerdir. Bu günler adına değişken oluşturuyoruz.

df["is_black_friday"] = 0
df.loc[df["transaction_date"].isin(["2018-11-22","2018-11-23","2019-11-29","2019-11-30"]) ,"is_black_friday"]=1

df["is_summer_solstice"] = 0
df.loc[df["transaction_date"].isin(["2018-06-19","2018-06-20","2018-06-21","2018-06-22",
                                    "2019-06-19","2019-06-20","2019-06-21","2019-06-22",]) ,"is_summer_solstice"]=1


# Görev 3 : Modellemeye Hazırlık ve Modelleme

# Adım 1: One-hot encoding yapınız.

df = pd.get_dummies(df, columns=['merchant_id','day_of_week', 'month'])

# Converting sales to log(1+sales)

df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)

# Adım 2: Custom Cost Function'ları tanımlayınız.

# MAE, MSE, RMSE, SSE

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: symmsetric mean absolute error (adjusted MAPE)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

# Adım 3: Veri setini train ve validation olarak ayırınız.

import re
df = df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x)) # rakam, harf ve _ dışındaki karakterleri siliyoruz

# 2020'nin 1. ayına kadar olan zamanı train seti olarak belirliyoruz
train = df.loc[(df["transaction_date"] < "2020-01-01"), :]

# 2020'nin ilk 3 ayını validasyon seti olarak belirliyoruz. covid etkisi göz önünde bulundurulmalı
val = df.loc[((df["transaction_date"] > "2020-01-01") & (df["transaction_date"] < "2020-04-01")), :]

cols = [col for col in train.columns if col not in ['transaction_date', 'id', "Total_Transaction","Total_Paid", "year" ]]

Y_train = train['Total_Transaction']
X_train = train[cols]

Y_val = val['Total_Transaction']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape
# ((5105,), (5105, 212), (630,), (630, 212))

# Adım 4: LightGBM Modelini oluşturunuz ve SMAPE ile hata değerini gözlemleyiniz.


lgb_params = {'metric': {'mae'},
              'num_leaves': 6,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)


y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))
# 24.793104156448617

####################################
# Değişken Önem Düzeyleri
####################################

def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30, plot=True)

plot_lgb_importances(model, num=30)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()