#importing libraries
#importing the libraries for data analysis
import numpy as np
import pandas as pd
from pylab import rcParams
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
rcParams['figure.figsize'] = 10,20# plotting for bigger plot size
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

clean_mob_encode=pd.read_csv('encode.csv')
y = clean_mob_encode['approx_price_EUR'].values
clean_mob_encode.drop('approx_price_EUR', axis=1, inplace=True)
X=clean_mob_encode.copy()


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)


std=StandardScaler()
X_train=std.fit_transform(X_train)
X_test=std.fit_transform(X_test)


print("Training model")
reg=RandomForestRegressor()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
print(r2_score(y_test,y_pred)*100)

#saving my model
pickle.dump(reg, open('rf_model.pkl','wb'))
print("dumping complete")
#
# #loading the model

model = pickle.load(open('rf_model.pkl','rb'))






