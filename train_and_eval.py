from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import to_categorical

import create_model as cm



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

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model.fit(x_train, y_train,
        epochs=20,
        batch_size=128)

## Evaluation
score = model.evaluate(x_test, y_test, batch_size=128)
print("\n", "Model accuracy: ", score[1], "\n")


