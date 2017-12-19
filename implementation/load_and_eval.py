# loads a trained model and evaluates it
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical

import read_data as rd

model = load_model('./output/nikulaj_model.hdf5')
model.summary()

(_, _), (x_test, y_test) = rd.read_data("./data")
y_test = to_categorical(y_test)

score = model.evaluate(x_test, y_test, batch_size=128)
print("\n", "Model accuracy: ", score[1], "\n")
