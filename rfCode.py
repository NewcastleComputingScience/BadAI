import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv("train.csv")
print (data.columns)
del data['year']

X_train, X_test, y_train, y_test = train_test_split(data, data.loc[: ,'actual duration'], test_size=0.3, random_state=123456)

del X_train['actual duration']
del X_test['actual duration']

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, random_state=123456, n_jobs=-1)
rf.fit(X_train, y_train)

r2s = rf.score(X_test, y_test)

print ('R^2 from test set = ', r2s)

print (rf.feature_importances_)

predict=pd.read_csv("predict.csv")

predict_label = predict.loc[: ,'actual duration']

del predict['year']
del predict['actual duration']

score = rf.score(predict, predict_label)

print ('R^2 from predicion set = ', score)
