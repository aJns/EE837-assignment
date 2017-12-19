from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import to_categorical

import create_model as cm
import read_data as rd



## Creating the model

data_shape = (28, 28, 1)

model = cm.create_model(data_shape)

model.summary()

## Compiling the model

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
        )

## Training

try:
        (x_train, y_train), (x_test, y_test) = rd.read_data("/data")
        using_floydhub = True
except IOError:
        print("Couldn't open the dataset in /data, loading keras dataset")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        using_floydhub = False

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model.fit(x_train, y_train,
        epochs=20,
        batch_size=128)

## Evaluation
score = model.evaluate(x_test, y_test, batch_size=128)
print("\n", "Model accuracy: ", score[1], "\n")


## save model
if using_floydhub:
        output_dir = '/output/'
else:
        output_dir = './output/'
        if not os.path.exists(output_dir):
                os.makedirs(output_dir)

model.save(output_dir+'nikulaj_model.hdf5')


