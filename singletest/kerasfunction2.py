import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt


# model creation
model = Sequential()
model.add(Dense(units=100, input_dim=1, activation="sigmoid"))
model.add(Dense(units=100, activation="sigmoid"))
model.add(Dense(units=100,  activation="sigmoid"))
model.add(Dense(units=100,  activation="sigmoid"))
model.add(Dense(units=100,  activation="sigmoid"))
model.add(Dense(units=1, activation='linear'))

# model compilation
model.compile(optimizer='adam', loss='mean_squared_error')

x = np.linspace(-1,1, 1000)
y = x**3 * np.cos(x*np.pi)  #

# Model training
model.fit(x, y, epochs=40, batch_size=10)

# Utiliser le modèle pour prédire à partir de nouvelles données
predictions = model.predict(x)


plt.plot(x, y, color='blue')
plt.plot(x, predictions, color='red')
plt.show()