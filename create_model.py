from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D, Conv2D, Dropout, Flatten, Reshape


def create_model(data_shape):

        model = Sequential()

        model.add( Reshape(data_shape, input_shape=data_shape[0:2]) )

        model.add( Conv2D(filters=28, kernel_size=(3,3)) )
        model.add( Activation('relu') )

        model.add( Conv2D(filters=28, kernel_size=(3,3)) )
        model.add( Activation('relu') )

        model.add( MaxPooling2D(pool_size=(2,2)) )
        model.add( Dropout(0.25) )

        model.add( Conv2D(filters=14, kernel_size=(3,3)))
        model.add( Activation('relu') )

        model.add( Conv2D(filters=14, kernel_size=(3,3)))
        model.add( Activation('relu') )

        model.add( MaxPooling2D(pool_size=(2,2)) )
        model.add( Dropout(0.25) )

        model.add( Flatten() )

        model.add( Dense(126) )
        model.add( Activation('relu') )
        model.add( Dropout(0.5) )

        model.add( Dense(10) )
        model.add( Activation('softmax') )


        return model


if __name__ == "__main__":
        model = create_model( (28, 28, 1) )
        model.summary()
