
from __future__ import print_function, division
## python libs
import os
from skimage import io, metrics
from skimage.metrics import structural_similarity as ssim
import numpy as np
## tf-Keras libs
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dropout, Concatenate
from keras.layers import Add, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.applications import vgg19
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.nn import relu, tanh


def lrelu(x, leak=0.2):
    return tf.maximum(leak*x, x)



def VGG19_Content(dataset='imagenet'):
    # Load VGG, trained on imagenet data
    vgg = vgg19.VGG19(include_top=False, weights=dataset)
    vgg.trainable = False
    content_layers = ['block5_conv2']
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    return Model(vgg.input, content_outputs)

def netG16_encoder(x):
    enc_conv1 = tcl.conv2d(x, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv1')
    enc_conv1 = tcl.batch_norm(enc_conv1)
    enc_conv1 = lrelu(enc_conv1)
    print (enc_conv1)
    enc_conv2 = tcl.conv2d(enc_conv1, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv2')
    enc_conv2 = tcl.batch_norm(enc_conv2)
    enc_conv2 = lrelu(enc_conv2)
    print (enc_conv2)
    enc_conv3 = tcl.conv2d(enc_conv2, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv3')
    enc_conv3 = tcl.batch_norm(enc_conv3)
    enc_conv3 = lrelu(enc_conv3)
    print (enc_conv3)
    enc_conv4 = tcl.conv2d(enc_conv3, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv4')
    enc_conv4 = tcl.batch_norm(enc_conv4)
    enc_conv4 = lrelu(enc_conv4)
    print (enc_conv4)
    enc_conv5 = tcl.conv2d(enc_conv4, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv5')
    enc_conv5 = tcl.batch_norm(enc_conv5)
    enc_conv5 = lrelu(enc_conv5)
    print (enc_conv5)
    enc_conv6 = tcl.conv2d(enc_conv5, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv6')
    enc_conv6 = tcl.batch_norm(enc_conv6)
    enc_conv6 = lrelu(enc_conv6)
    print (enc_conv6)
    enc_conv7 = tcl.conv2d(enc_conv6, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv7')
    enc_conv7 = tcl.batch_norm(enc_conv7)
    enc_conv7 = lrelu(enc_conv7)
    print (enc_conv7)
    enc_conv8 = tcl.conv2d(enc_conv7, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_enc_conv8')
    enc_conv8 = tcl.batch_norm(enc_conv8)
    enc_conv8 = lrelu(enc_conv8)
    print (enc_conv8); print("\n")
    layers = [enc_conv1, enc_conv2, enc_conv3, enc_conv4, enc_conv5, enc_conv6, enc_conv7, enc_conv8]
    return layers
def netG16_decoder(layers, lab=False):
    enc_conv1, enc_conv2, enc_conv3, enc_conv4, enc_conv5, enc_conv6, enc_conv7, enc_conv8 = layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6], layers[7]
    # decoder, no batch norm
    dec_conv1 = tcl.convolution2d_transpose(enc_conv8, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv1')
    dec_conv1 = relu(dec_conv1)
    dec_conv1 = tf.concat([dec_conv1, enc_conv7], axis=3)
    print (dec_conv1)
    dec_conv2 = tcl.convolution2d_transpose(dec_conv1, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv2')
    dec_conv2 = relu(dec_conv2)
    dec_conv2 = tf.concat([dec_conv2, enc_conv6], axis=3)
    print (dec_conv2)
    dec_conv3 = tcl.convolution2d_transpose(dec_conv2, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv3')
    dec_conv3 = relu(dec_conv3)
    dec_conv3 = tf.concat([dec_conv3, enc_conv5], axis=3)
    print (dec_conv3)
    dec_conv4 = tcl.convolution2d_transpose(dec_conv3, 512, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv4')
    dec_conv4 = relu(dec_conv4)
    dec_conv4 = tf.concat([dec_conv4, enc_conv4], axis=3)
    print (dec_conv4)
    dec_conv5 = tcl.convolution2d_transpose(dec_conv4, 256, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv5')
    dec_conv5 = relu(dec_conv5)
    dec_conv5 = tf.concat([dec_conv5, enc_conv3], axis=3)
    print (dec_conv5)
    dec_conv6 = tcl.convolution2d_transpose(dec_conv5, 128, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv6')
    dec_conv6 = relu(dec_conv6)
    dec_conv6 = tf.concat([dec_conv6, enc_conv2], axis=3)
    print (dec_conv6)
    dec_conv7 = tcl.convolution2d_transpose(dec_conv6, 64, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv7')
    dec_conv7 = relu(dec_conv7)
    dec_conv7 = tf.concat([dec_conv7, enc_conv1], axis=3)
    print (dec_conv7)
    c = 2 if lab else 3
    dec_conv8 = tcl.convolution2d_transpose(dec_conv7, c, 4, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_dec_conv8')
    dec_conv8 = tanh(dec_conv8)
    print (dec_conv1)
    return dec_conv8
class FUNIE_GAN_UP():
    def __init__(self, imrow=256, imcol=256, imchan=3):
        self.img_rows, self.img_cols, self.channels = imrow, imcol, imchan
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        self.vgg_content = VGG19_Content()
        self.disc_patch = (16, 16, 1)
        self.n_residual_blocks = 5
        self.gf, self.df = 32, 32
        optimizer = Adam(0.0003, 0.5)
        self.d_A = self.FUNIE_UP_discriminator()
        self.d_B = self.FUNIE_UP_discriminator()
        self.d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.g = self.FUNIE_UP_generator()
        self.g_BA = self.FUNIE_UP_generator()
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)
        self.d_A.trainable = False
        self.d_B.trainable = False
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        self.combined = Model(inputs=[img_A, img_B], outputs=[ valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                            loss_weights=[1, 1, 10, 10, 1, 1], optimizer=optimizer)


   
def ssim_ratio():
    underwater_path = 'train_images\trainA/'
    enhanced_path = 'testImages/'
    underwater_files = os.listdir(underwater_path)
    enhanced_files = os.listdir(enhanced_path)
    ssim=[]
    for i, filename in enumerate(underwater_files):  
        underwater_img = io.imread(os.path.join(underwater_path, filename), as_gray=True)
        enhanced_img = io.imread(os.path.join(enhanced_path, enhanced_files[i]), as_gray=True)
        ssim_ratio = ssim(underwater_img, enhanced_img, win_size=3)
        ssim.append(ssim_ratio)
    return ssim

def perceptual_distance(self, y_true, y_pred):
        
        y_true = (y_true+1.0)*127.5 # [-1,1] -> [0, 255]
        y_pred = (y_pred+1.0)*127.5 # [-1,1] -> [0, 255]
        rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
        r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
        g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
        b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
        return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


def total_gen_loss(self, org_content, gen_content):
        # custom perceptual loss function
        vgg_org_content = self.vgg_content(org_content)
        vgg_gen_content = self.vgg_content(gen_content)
        content_loss = K.mean(K.square(vgg_org_content - vgg_gen_content), axis=-1)
        mae_gen_loss = K.mean(K.abs(org_content-gen_content))
        perceptual_loss = self.perceptual_distance(org_content, gen_content)
        #gen_total_err = 0.7*mae_gen_loss+0.3*content_loss # v1
        # updated loss function in v2
        gen_total_err = 0.7*mae_gen_loss+0.2*content_loss+0.1*perceptual_loss
        return gen_total_err


def FUNIE_UP_generator(self):
        
        def conv2d(layer_input, filters, f_size=3, bn=True):
            u = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            u = LeakyReLU(alpha=0.2)(d)
            if bn: u = BatchNormalization(momentum=0.8)(d)
            return u

        
    
        u0 = Input(shape=self.img_shape)
        gan_output = conv2d(netG16_encoder(u0),self.gf*1,f_size=5,bn=False)
        u1 = conv2d(gan_output, self.gf*1, f_size=5, bn=False) 
        u2 = conv2d(u1, self.gf*4, f_size=4, bn=True)  
        u3 = conv2d(u2, self.gf*8, f_size=4, bn=True)  
        u4 = conv2d(u3, self.gf*8, f_size=3, bn=True) 
        u5 = conv2d(u4, self.gf*8, f_size=3, bn=True) 
       
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u5)

        return Model(u0,output_img)



def FUNIE_UP_discriminator(self):
        
        def d_layer(layer_input, filters, strides_=2, f_size=3, bn=True):
            
            d = Conv2D(filters, kernel_size=f_size, strides=strides_, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn: d = BatchNormalization(momentum=0.8)(d)
            return d
        decoder=netG16_decoder(self.g)    
        d1 = d_layer(decoder, self.df, bn=False) 
        d2 = d_layer(d1, self.df*2) 
        d3 = d_layer(d2, self.df*4) 
        d4 = d_layer(d3, self.df*8) 
        d5 = d_layer(d4, self.df*8, strides_=1) 
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d5)

        return Model(decoder, validity)

