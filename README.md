# Ex:05 Feature Generation
## AIM
To read the given data and perform Feature Generation process and save the data to a file.

## ALGORITHM
#### STEP 1
Read the given Data
#### STEP 2
Clean the Data Set using Data Cleaning Process
#### STEP 3
Apply Feature Generation techniques to all the feature of the data set
#### STEP 4
Save the data to the file

## CODE AND OUTPUT FOR FEATURE ENCODING AND FEATURE SCALING:
```
import pandas as pd
import numpy as np
```
```
from google.colab import files
upload = files.upload()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/e095ccc7-f48b-450d-965f-c1a3ad7b25ce)
```
df = pd.read_csv("bmi.csv")
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/4ce008e8-f1db-4130-9d28-1696d6bc2713)
```
from category_encoders import BinaryEncoder
e1 = BinaryEncoder()
bn = e1.fit_transform(df['Gender'])
df = pd.concat([df,bn],axis = 1)
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/0b787bcc-eff9-4ae4-a823-1ee13e66a414)
```
from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
df[['Height','Weight']] = rs.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/6edec110-486f-4e03-97bf-545af1c663a0)
```
from google.colab import files
upload = files.upload()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/cd0b07c0-36a9-4d86-b033-4eb658cff9bb)
```
df = pd.read_csv("data1.csv")
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/399ac14b-9b05-4681-95d9-180fda96e9da)
```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder
data = ['Very Hot','Hot','Warm','Cold']
e1 = OrdinalEncoder(categories = [data])
df['Ord_1'] = e1.fit_transform(df[['Ord_1']])
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/822853dc-3d4e-4108-a822-47cb30f83b66)
```
data1 = ['High School','Diploma','Bachelors','Masters','PhD']
e1 = LabelEncoder()
df['Ord_2'] = e1.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/88fed322-d748-4ff4-be83-cf779707c8f7)
```
e2 = OneHotEncoder(sparse = False)
enc = pd.DataFrame(e2.fit_transform(df[['City']]))
df = pd.get_dummies(df,columns = ['City'])
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/d67d8f7e-9d35-4639-8ae3-bae47ea4c200)
```
pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
e3 = BinaryEncoder()
bn = e3.fit_transform(df[['bin_1','bin_2']])
df = pd.concat([df,bn],axis = 1)
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/c0117950-6cb4-4095-8b83-1a96ae37c3f6)
```
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
df[['Ord_1','Ord_2']] = mm.fit_transform(df[['Ord_1','Ord_2']])
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/08a9ce1b-b673-4163-ac41-9f8d8c9a5e35)
```
from google.colab import files
upload = files.upload()
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/6c9e1bc2-9ed9-4e45-bf73-01efe9b75074)
```
df = pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/786efbd4-29b3-4f45-a9cd-3423421809ff)
```
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
data1 = ['Hot','Warm','Cold']
e1 = LabelEncoder()
df['ord_2'] = e1.fit_transform(df['ord_2'])
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/3c3064db-48fc-46eb-a5ab-8a8194ead282)
```
e2 = OneHotEncoder(sparse = False)
enc = pd.DataFrame(e2.fit_transform(df[['nom_0']]))
df = pd.get_dummies(df,columns = ['nom_0'])
df```
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/cb0060a5-2be3-43b6-b2d9-5001b6f3f0ac)
```
from category_encoders import BinaryEncoder
e3 = BinaryEncoder()
bn = e3.fit_transform(df[['bin_1','bin_2']])
df = pd.concat([df,bn],axis = 1)
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/76af4815-e685-4933-bc86-0e7f113496ab)
```
from sklearn.preprocessing import  StandardScaler
ss = StandardScaler()
df[['ord_2']] = ss.fit_transform(df[['ord_2']])
df
```
![image](https://github.com/mathes6112004/ODD2023-Datascience-Ex-05/assets/119477782/cccc9308-2a69-4c90-a705-eb381e1ba42e)

## Result:
Feature Encoding process and Feature Scaling process is applied to the given data frame sucessfully.
