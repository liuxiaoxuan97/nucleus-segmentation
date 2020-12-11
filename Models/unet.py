import numpy as np
import matplotlib.pyplot as plt
import collections
import os
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.optimizers import Adam
from utils.dice_coef import dice_coef, dice_coef_numpy



opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
def unet(width=256,height=256,chan=3):
    inp = Input((width, height, chan))
    norm = Lambda(lambda x: x / 255) (inp)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (norm)
    d1 = Dropout(0.1) (c1)
    c2 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (d1)
    p1 = MaxPooling2D((2, 2)) (c2)
    c3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    d2 = Dropout(0.1) (c3)
    c4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (d2)
    p2 = MaxPooling2D((2, 2)) (c4)

    c5 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    d3 = Dropout(0.2) (c5)
    c6 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (d3)
    p3 = MaxPooling2D((2, 2)) (c6)

    c7 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    d4 = Dropout(0.2) (c7)
    c8 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (d4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c8)

    c9 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    d5 = Dropout(0.3) (c9)
    c10 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (d5)

    u1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c9)
    u2 = concatenate([u1, c8])
    u3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u2)
    d6 = Dropout(0.2) (u3)
    u4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (d6)

    u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (u4)
    u6 = concatenate([u5, c6])
    u7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    d7 = Dropout(0.2) (u7)
    u8 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (d7)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (u8)
    u10 = concatenate([u9, c4])
    u11 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u10)
    d8 = Dropout(0.1) (u11)
    u12 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (d8)

    u13 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (u12)
    u14 = concatenate([u13, c2], axis=3)
    u15 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u14)
    d9 = Dropout(0.1) (u15)
    u16 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (d9)

    out = Conv2D(1, (1, 1), activation='sigmoid') (u16)

    model = Model(inputs=[inp], outputs=[out])
    
    return model
