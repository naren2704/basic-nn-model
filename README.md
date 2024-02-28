# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/naren2704/basic-nn-model/assets/118706984/88c5897f-1b21-42f3-8d5b-7c8775388311)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:NARENDRAN B
### Register Number:212222240069
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds,_ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('ex1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})

X = df[['input']].values
y = df[['output']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

AI_Brain = Sequential([
    Dense(units = 1, activation = 'relu', input_shape=[1]),
    Dense(units = 5, activation = 'relu'),
    Dense(units = 1)
])

AI_Brain.compile(optimizer= 'rmsprop', loss="mse")
AI_Brain.fit(X_train1,y_train,epochs=5000)
AI_Brain.summary()

loss_df = pd.DataFrame(AI_Brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
AI_Brain.evaluate(X_test1,y_test)
X_n1 = [[20]]
X_n1_1 = Scaler.transform(X_n1)
AI_Brain.predict(X_n1_1)


```
## Dataset Information

![image](https://github.com/naren2704/basic-nn-model/assets/118706984/1de54efc-3498-4c95-bd5c-7f13d21d4dc9)


## OUTPUT
![image](https://github.com/naren2704/basic-nn-model/assets/118706984/a9aaf22d-a8c4-4b10-9088-ea225a902f4e)



### Test Data Root Mean Squared Error
![image](https://github.com/naren2704/basic-nn-model/assets/118706984/569bf011-d3bc-4ec1-8e53-3498813a3ad4)



### New Sample Data Prediction

![image](https://github.com/naren2704/basic-nn-model/assets/118706984/6b238842-7812-4537-8cd8-69d333b57743)


## RESULT
Thus the Process of developing a neural network regression model for the created dataset is successfully executed.

Include your result here
