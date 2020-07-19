from keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, concatenate, Dropout, Lambda, Conv2DTranspose, Add
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.metrics import MeanIoU


class Unet(Model):
    """
    This design is adapted from the following examples:
    https://keras.io/api/models/model/
    https://github.com/sevakon/unet-keras/
    """

    def __init__(self, input_dim, n_classes=1, n_filters=16, pretrained_weights=None):
        self.input_dim = input_dim
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.dropout = 0.1
        self.stride_size = (2, 2)

        if n_classes == 1:
            self.loss = 'binary_crossentropy'
            self.last_act = 'sigmoid'
        else:
            self.loss = 'categorical_crossentropy'
            self.last_act = 'softmax'

        inputs = Input(self.input_dim)
        s = Lambda(lambda x: x / 255.)(inputs)

        conv1 = self.__double_conv2D(s, self.n_filters * 1)
        max_p1 = MaxPooling2D(self.pool_size)(conv1)

        conv2 = self.__double_conv2D(max_p1, self.n_filters * 2)
        max_p2 = MaxPooling2D(self.pool_size)(conv2)

        conv3 = self.__double_conv2D(max_p2, self.n_filters * 3)
        max_p3 = MaxPooling2D(self.pool_size)(conv3)

        conv4 = self.__double_conv2D(max_p3, self.n_filters * 4)
        max_p4 = MaxPooling2D(self.pool_size)(conv4)

        conv5 = self.__double_conv2D(max_p4, self.n_filters * 5)

        up_conv6 = self.__deconvolution(conv5, self.n_filters * 4)
        up_conv6 = concatenate([up_conv6, conv4])
        conv6 = self.__double_conv2D(up_conv6, self.n_filters * 4)

        up_conv7 = self.__deconvolution(conv6, self.n_filters * 3)
        up_conv7 = concatenate([up_conv7, conv3])
        conv7 = self.__double_conv2D(up_conv7, self.n_filters * 3)

        up_conv8 = self.__deconvolution(conv7, self.n_filters * 2)
        up_conv8 = concatenate([up_conv8, conv2])
        conv8 = self.__double_conv2D(up_conv8, self.n_filters * 2)

        up_conv9 = self.__deconvolution(conv8, self.n_filters * 1)
        up_conv9 = concatenate([up_conv9, conv1])
        conv9 = self.__double_conv2D(up_conv9, self.n_filters * 1)

        outputs = Conv2D(n_classes, (1, 1), activation=self.last_act)(conv9)

        # initialize Keras Model with defined above input and output layers
        super(Unet, self).__init__(inputs=inputs, outputs=outputs)

        # load preatrained weights
        if pretrained_weights:
            self.load_weights(pretrained_weights)

    def __double_conv2D(self, inputs, filters):
        c = Conv2D(filters, self.kernel_size, activation=relu, kernel_initializer='he_normal', padding='same')(inputs)
        c = Dropout(self.dropout)(c)  # in order to avoid over-fitting
        c = Conv2D(filters, self.kernel_size, activation=relu, kernel_initializer='he_normal', padding='same')(c)

        return c

    def __deconvolution(self, inputs, filters):
        d = Conv2DTranspose(filters, self.kernel_size, strides=self.stride_size, padding='same')(inputs)

        return d

    def build(self):
        self.compile(optimizer=Adam(), loss=self.loss, metrics=[MeanIoU(num_classes=self.n_classes)])
        self.summary()

    def save_model(self, name):
        self.save_weights(name)

    @staticmethod
    def checkpoint(name):
        return ModelCheckpoint(name, monitor='iou_score', verbose=1, mode='max', save_best_only=True,
                               save_weights_only=True)

    @staticmethod
    def early_stopping():
        return EarlyStopping(monitor='iou_score', patience=5, verbose=1, mode='min')
