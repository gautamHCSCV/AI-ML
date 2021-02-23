from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
data = loadtxt("pima-indians-diabetes.csv", delimiter=',')
x = data[:,:8]
y = data[:,8]
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation= 'sigmoid'))
model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
model.fit(x,y,epochs=150, batch_size=25)
loss, acc = model.evaluate(x,y)
print(acc)