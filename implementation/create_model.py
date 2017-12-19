from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D, Conv2D, Dropout, Flatten, Reshape
from keras.utils import plot_model


def create_model(data_shape):

        model = Sequential()

        model.add( Reshape(data_shape, input_shape=data_shape[0:2]) )

####### CONV ##################################################################
        filter_count = 64

        model.add( Conv2D(filters=filter_count, kernel_size=(3,3), padding='same'))
        model.add( Activation('relu') )

        model.add( Conv2D(filters=filter_count, kernel_size=(2,2), padding='same'))
        model.add( Activation('relu') )

        model.add( Conv2D(filters=filter_count, kernel_size=(1,1), padding='same'))
        model.add( Activation('relu') )

        model.add( MaxPooling2D(pool_size=(2,2)) )
        model.add( Dropout(0.25) )

####### CONV ##################################################################
        filter_count = 128

        model.add( Conv2D(filters=filter_count, kernel_size=(3,3), padding='same'))
        model.add( Activation('relu') )

        model.add( Conv2D(filters=filter_count, kernel_size=(2,2), padding='same'))
        model.add( Activation('relu') )

        model.add( Conv2D(filters=filter_count, kernel_size=(1,1), padding='same'))
        model.add( Activation('relu') )

        model.add( MaxPooling2D(pool_size=(2,2)) )
        model.add( Dropout(0.25) )

####### CONV ##################################################################
        filter_count = 256

        model.add( Conv2D(filters=filter_count, kernel_size=(3,3), padding='same'))
        model.add( Activation('relu') )

        model.add( Conv2D(filters=filter_count, kernel_size=(2,2), padding='same'))
        model.add( Activation('relu') )

        model.add( Conv2D(filters=filter_count, kernel_size=(1,1), padding='same'))
        model.add( Activation('relu') )

        model.add( MaxPooling2D(pool_size=(2,2)) )
        model.add( Dropout(0.25) )

####### CONV ##################################################################
        filter_count = 512

        model.add( Conv2D(filters=filter_count, kernel_size=(3,3), padding='same'))
        model.add( Activation('relu') )

        model.add( Conv2D(filters=filter_count, kernel_size=(2,2), padding='same'))
        model.add( Activation('relu') )

        model.add( Conv2D(filters=filter_count, kernel_size=(1,1), padding='same'))
        model.add( Activation('relu') )

        model.add( MaxPooling2D(pool_size=(2,2)) )
        model.add( Dropout(0.25) )

####### DENSE #################################################################
        model.add( Flatten() )

        model.add( Dense(128) )
        model.add( Activation('relu') )
        model.add( Dropout(0.5) )

        model.add( Dense(10) )
        model.add( Activation('softmax') )


        return model


if __name__ == "__main__":
        model = create_model( (28, 28, 1) )
        model.summary()
        try:
                plot_model(model, to_file='./output/model.png')
        except:
                pass
