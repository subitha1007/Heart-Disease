import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#Data Collection
df=pd.read_csv('C:/Users/Ajith/Pictures/day20/Cardiovascular_Disease_Dataset.csv') 
print(df)

##Data processing
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df['gender']=le.fit_transform(df['gender'])
df['chestpain']=le.fit_transform(df['chestpain'])
df['restingBP']=le.fit_transform(df['restingBP'])
df['slope']=le.fit_transform(df['slope'])
df['serumcholestrol']=le.fit_transform(df['serumcholestrol'])
df['noofmajorvessels']=le.fit_transform(df['noofmajorvessels'])
##df['trestbps']=le.fit_transform(df['trestbps'])
##df['chol']=le.fit_transform(df['chol'])



print(df)
x=df.drop(columns=['maxheartrate'])
y=df['maxheartrate']

print("xxxx",x)
print("yyyy",y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)

print("DF",df.shape)
print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)

from sklearn.naive_bayes import GaussianNB
NB=GaussianNB()

NB.fit(x_train,y_train)

##Model Evaluation

y_pred=NB.predict(x_test)


print("y_pred",y_pred)
print("y_test",y_test)

from sklearn.metrics import accuracy_score
print('ACCURACY is', accuracy_score(y_pred,y_test))

testPrediction=NB.predict([[29,0.2,100,106,1.2,80,1,1,1,3,102,45,6]])
if testPrediction==1:
    print("The patient have disease,please consult the doctor")
else:
    print("the patient normal")

