#!/usr/bin/env python3

"""
    DCGAN implemented using Keras-Tensorflow. There are two parts to the DCGAN model, the discriminator and the
        generator. Refer to Notes file for more detailed description of the model's layers.
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout, Activation
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.layers import Input
from src.preprocess_data import load_data
import numpy as np
import matplotlib.pyplot as plt
import os


def build_discriminator():
    """
    Function that builds the layers in the Discriminator.
    The Discriminator takes an input vector with the shape (64, 64, 3), after the the final sigmoid layer outputs
        a value between 0 - 1 representing fake or real.
    :return: model
    """
    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(64, 64, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()

    return model


def build_generator():
    model = Sequential()
    model.add(Dense(4*4*512, input_dim=100))
    model.add(Reshape((4, 4, 512)))
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same'))
    model.add(Activation('tanh'))
    model.summary()

    return model


def build_combined(generator, discriminator):
    dcgan_input = Input(shape=(100,))
    gen_input = generator(dcgan_input)
    dcgan_output = discriminator(gen_input)
    model = Model(inputs=dcgan_input, outputs=dcgan_output)
    return model


class DCGAN:
    def __init__(self):
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.discriminator = build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.generator = build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.combined = build_combined(self.generator, self.discriminator)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def summary(self):
        return self.combined.summary()

    def train(self, epochs, batch_size, save_interval):
        X_train, X_test, y_train, y_test = load_data()
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        print('Data successfully loaded.')
        half_batch = int(batch_size/2)

        for epoch in range(epochs):
            # train discriminator
            self.discriminator.trainable = True
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.uniform(-1, 1, size=[half_batch, 100])
            gen_imgs = self.generator.predict(noise)

            train_imgs = np.concatenate((imgs, gen_imgs))
            labels = np.ones([batch_size, 1])
            labels[half_batch, :] = 0

            d_loss = self.discriminator.train_on_batch(train_imgs, labels)

            # train generator
            self.discriminator.trainable = False
            noise = np.random.uniform(-1, 1, size=[batch_size, 100])
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # print progress of training
            print('{0} [D loss: {1}] [G loss: {2}]'.format(epoch, d_loss, g_loss))
            if epoch % save_interval == 0 or epoch == epochs - 1:
                self.plot_imgs(epoch)
                self.plot_train_sample(imgs)

    def plot_imgs(self, epoch):
        output_dir = os.path.abspath('../output')
        noise_input = np.random.uniform(-1, 1, size=[25, 100])
        predictions = self.generator.predict(noise_input)
        predictions = predictions * 127.5 + 127.5
        predictions = predictions.astype(np.uint8)

        plt.figure(figsize=(5, 5))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(predictions[i, :, :, :])
            plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'minimalism_{}_epoch.png'.format(epoch)))
        plt.close()

    def plot_train_sample(self, imgs):
        output_dir = os.path.abspath('../output')
        sample_imgs = imgs[0:25]
        sample_imgs = sample_imgs * 127.5 + 127.5
        sample_imgs = sample_imgs.astype(np.uint8)
        plt.figure(figsize=(5, 5))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(sample_imgs[i, :, : , :])
            plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'minimalism_training_sample'))
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.summary()
    #dcgan.train(epochs=5000, batch_size=128, save_interval=500)




