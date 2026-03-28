import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Rescaling
from tensorflow.keras.models import Model

def build_colorization_model():
    # Input
    inputs = Input(shape=(256, 256, 1), name="L_channel_input")
    x = Rescaling(1./100.0)(inputs)

    # Encoder 1
    x = Conv2D(32, (3, 3), padding='same', name="enc1_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same', name="enc1_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), name="enc1_pool")(x)
    
    # Encoder 2
    x = Conv2D(64, (3, 3), padding='same', name="enc2_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name="enc2_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), name="enc2_pool")(x)

    # Encoder 3
    x = Conv2D(128, (3, 3), padding='same', name="enc3_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name="enc3_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), name="enc3_pool")(x)
    
    # Bottleneck
    x = Conv2D(256, (3, 3), padding='same', name="bottleneck_conv")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Decoder 1
    x = UpSampling2D((2, 2), name="dec1_upsample")(x)
    x = Conv2D(128, (3, 3), padding='same', name="dec1_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name="dec1_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Decoder 2
    x = UpSampling2D((2, 2), name="dec2_upsample")(x)
    x = Conv2D(64, (3, 3), padding='same', name="dec2_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name="dec2_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Decoder 3
    x = UpSampling2D((2, 2), name="dec3_upsample")(x)
    x = Conv2D(32, (3, 3), padding='same', name="dec3_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Output
    outputs = Conv2D(2, (1, 1), padding='same', activation='tanh', name="output_layer")(x)
    
    return Model(inputs, outputs, name="Colorization_Model")

if __name__ == '__main__':
    model = build_colorization_model()
    model.summary()