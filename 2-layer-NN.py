import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import warnings

from keras.models import Sequential
from keras.layers import Dense
#from keras.optimizers import Adam
from keras.optimizers import SGD

warnings.filterwarnings('ignore')
data = pd.read_excel("User_Modeling_Dataset.xlsx")

#print(data.head())

X = data.iloc[:,0:4].values
Y = data.iloc[:,5].values.reshape(-1,1)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(Y)

xtrain, xtest, ytrain, ytest = train_test_split(X , y , test_size=0.30 , random_state=0)

model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='sigmoid', name='fc1'))
model.add(Dense(10, activation='sigmoid', name='fc2'))
model.add(Dense(3, activation='softmax', name = 'output'))

#optimizer = Adam(lr=0.001)
#optimizer = SGD(lr=0.0005, nesterov=True)
optimizer = SGD(lr=0.3)
#model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer, loss='mean_squared_error', metrics = ['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

model.fit(xtrain, ytrain, verbose=2, batch_size=8, epochs=200)

results = model.evaluate(xtest, ytest)
print('Final test set loss:',results[0])
print('Final test set accuracy:',results[1])
#print('Final test set loss: {:4f}'.format(results[0]))
#print('Final test set accuracy: {:4f}'.format(results[1]))