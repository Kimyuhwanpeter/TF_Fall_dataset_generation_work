# -*- coding:utf-8 -*-
from absl import flags
from random import random
from D2_conv_model import *

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import sys

flags.DEFINE_integer("img_size", 112, "Image size (width and height)")

flags.DEFINE_integer("frames", 16, "Number of frames")

flags.DEFINE_integer("ch", 3, "Image channels")

flags.DEFINE_integer("batch_size", 1, "Training batch size")

flags.DEFINE_float("lr", 0.0002, "Learning rate")

flags.DEFINE_integer("epochs", 200, "Number of training epochs")

flags.DEFINE_string("A_img_path", "D:/[1]DB/[4]etc_experiment/Fall_dataset/[5]TCL_dataset/GAN_data/divide/Fall_stand_Drinking/train/A_train/", "Training A image path")

flags.DEFINE_string("B_img_path", "D:/[1]DB/[4]etc_experiment/Fall_dataset/[5]TCL_dataset/GAN_data/divide/Fall_stand_Drinking/train/B_train/", "Training B image path")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Restore the checkpoint files")

flags.DEFINE_string("save_checkpoint", "", "Save checkpoint path")

flags.DEFINE_string("sample_images", "", "Save sample images")

flags.DEFINE_string("graphs", "", "Save training graphs")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

g_optim = tf.keras.optimizers.Adam(FLAGS.lr)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr)

label_list = [("Drinking_some_water", 0),
              ("Falls_from_bed", 1),
              ("Falls_from_standing", 2),
              ("Getting_off_a_bed", 3),
              ("Jogging", 4),
              ("Lying_down", 5),
              ("Lying_still", 6),
              ("Moving_on_a_chair", 7),
              ("Object_picking", 8),
              ("Sitting_down", 9),
              ("Sitting_on_a_bed", 10),
              ("Standing_still", 11),
              ("Standing_up", 12),
              ("Walking", 13)]

def all_data(folder_list, path):

    list_ = folder_list
    folder_list = [path + "/" + folder for folder in folder_list]

    img, lab = [], []
    for i in range(len(folder_list)):
        folder_list_ = os.listdir(folder_list[i])
        for k in range(len(label_list)):
            #print(folder_list[i].split('/')[-1])
            if folder_list[i].split('/')[-1] == label_list[k][0]:
                num = label_list[k][1]

        print(folder_list[i])
        folder_list_ = [path + "/" + folder_list[i].split("/")[-1] + "/" + folder for folder in folder_list_ if folder != ".DS_Store"]

        label = ["{}".format(num) for j in range(len(folder_list_))]
        lab.extend(label)

        for j in range(len(folder_list_)):
            real_img = os.listdir(folder_list_[j])
            real_img = [folder_list_[j] + "/" + name for name in real_img]
            img.append(real_img)

    data = list(zip(img, lab))

    return data

def train_func(img, lab):

    input = []
    for i in range(0, FLAGS.frames):

        img_ = tf.io.read_file(img[i])
        img_ = tf.image.decode_jpeg(img_, 3)

        if random() > 0.5:
            img_ = tf.image.resize(img_, [FLAGS.img_size + 2, FLAGS.img_size + 2])
            img_ = tf.image.random_crop(img_, [FLAGS.img_size, FLAGS.img_size, 3])
        else:
            img_ = tf.image.resize(img_, [FLAGS.img_size, FLAGS.img_size])

        img_ = img_ / 127.5 - 1.

        input.append(img_)

    input_img = input

    #label = lab     # for binary
    label = tf.one_hot(lab, 14)    # for multi

    return input_img, label

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(A_img, B_img, A2B_gener, B2A_gener, A_dis, B_dis):

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        for i in range(FLAGS.batch_size):
            g_, d_ = 0, 0
            generator_variables, discriminator_variables = [], []
            for j in range(FLAGS.frames):
                fake_B = run_model(A2B_gener, A_img[:, j, :, :, :], True)
                fake_A_ = run_model(B2A_gener, fake_B, True)
                fake_A = run_model(B2A_gener, B_img[:, j, :, :, :], True)
                fake_B_ = run_model(A2B_gener, fake_A, True)

                DB_fake = run_model(A_dis, fake_A, True)
                DA_fake = run_model(B_dis, fake_B, True)
                DA_real = run_model(A_dis, A_img[:, j, :, :, :], True)
                DB_real = run_model(B_dis, B_img[:, j, :, :, :], True)

                id_fake_A = run_model(A2B_gener, B_img[:, j, :, :, :], True)
                id_fake_B = run_model(B2A_gener, A_img[:, j, :, :, :], True)

                g_loss = mae_criterion(DB_fake, tf.ones_like(DB_fake)) + mae_criterion(DA_fake, tf.ones_like(DA_fake)) \
                        + (10.0 * abs_criterion(A_img[:, j, :, :, :], fake_A_)) + (10.0 * abs_criterion(B_img[:, j, :, :, :], fake_B_)) \
                        + (10.0 * abs_criterion(B_img[:, j, :, :, :], id_fake_A) + (10.0 * abs_criterion(A_img[:, j, :, :, :], id_fake_B)))

                disc_A_loss = (mae_criterion(DA_real, tf.ones_like(DA_real)) + mae_criterion(DA_fake, tf.zeros_like(DA_fake))) / 2
                disc_B_loss = (mae_criterion(DB_real, tf.ones_like(DB_real)) + mae_criterion(DB_fake, tf.zeros_like(DB_fake))) / 2
    
                d_loss = disc_A_loss + disc_B_loss
                
                g_ += g_loss
                d_ += d_loss
                #print("=================================================") # 코랩에 돌리고있는거 확인해보고 고치기!!
                #print((A2B_gener.trainable_variables + B2A_gener.trainable_variables)) # 7 프레임 까지만 허용되고, 그 이후부터는 OOM
                generator_variables.append(A2B_gener.trainable_variables + B2A_gener.trainable_variables)
                discriminator_variables.append(A_dis.trainable_variables + B_dis.trainable_variables)

        g_ /= FLAGS.frames
        d_ /= FLAGS.frames
        print("============================================================")
        generator_variables = tf.reduce_sum(generator_variables, keepdims=True) / FLAGS.frames
        discriminator_variables = tf.reduce_sum(discriminator_variables, keepdims=True) / FLAGS.frames
        print("============================================================")

    generator_gradients = g_tape.gradient(g_loss, generator_variables)
    discriminator_gradients = d_tape.gradient(d_loss, discriminator_variables)

    g_optim.apply_gradients(zip(generator_gradients, generator_variables))
    
    d_optim.apply_gradients(zip(discriminator_gradients, discriminator_variables))

    return g_loss, d_loss

def abs_criterion(input, target):
    return tf.reduce_mean(tf.abs(input - target))

def mae_criterion(input, target):
    return tf.reduce_mean((input - target)**2)

def main():

    G_a2b_model = ResnetGenerator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    G_b2a_model = ResnetGenerator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    D_a_model = ConvDiscriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    D_b_model = ConvDiscriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    
    G_a2b_model.summary()
    D_a_model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(G_a2b_model=G_a2b_model,
                                   G_b2a_model=G_b2a_model,
                                   D_a_model=D_a_model,
                                   D_b_model=D_b_model,
                                   g_optim=g_optim,
                                   d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("======================================")
            print("* Restored the latest checkpoint!!!! *")
            print("======================================")

    if FLAGS.train:
        count = 0

        for epoch in range(FLAGS.epochs):

            #############################################################################
            A_folder_list = os.listdir(FLAGS.A_img_path)
            A_folder_list = [folder for folder in A_folder_list if folder !=".DS_Store"]
            A_folder_list = all_data(A_folder_list, FLAGS.A_img_path)
            print(len(A_folder_list))

            A_img, A_lab = zip(*A_folder_list)
            A_img, A_lab = np.array(A_img), np.array(A_lab, dtype=np.int32)
            A_gener = tf.data.Dataset.from_tensor_slices((A_img, A_lab))
            A_gener = A_gener.shuffle(len(A_img))
            A_gener = A_gener.map(train_func)
            A_gener = A_gener.batch(FLAGS.batch_size)
            A_gener = A_gener.prefetch(tf.data.experimental.AUTOTUNE)
            #############################################################################
            #############################################################################
            B_folder_list = os.listdir(FLAGS.B_img_path)
            B_folder_list = [folder for folder in B_folder_list if folder != ".DS_Store"]
            B_folder_list = all_data(B_folder_list, FLAGS.B_img_path)
            print(len(B_folder_list))

            B_img, B_lab = zip(*B_folder_list)
            B_img, B_lab = np.array(B_img), np.array(B_lab, dtype=np.int32)
            B_gener = tf.data.Dataset.from_tensor_slices((B_img, B_lab))
            B_gener = B_gener.shuffle(len(B_img))
            B_gener = B_gener.map(train_func)
            B_gener = B_gener.batch(FLAGS.batch_size)
            B_gener = B_gener.prefetch(tf.data.experimental.AUTOTUNE)
            #############################################################################

            train_idx = min(len(A_folder_list), len(B_folder_list)) // FLAGS.batch_size
            A_train_iter = iter(A_gener)
            B_train_iter = iter(B_gener)

            for step in range(train_idx):
                batch_A_img, _ = next(A_train_iter)
                batch_B_img, _ = next(B_train_iter)

                g_loss, d_loss = cal_loss(batch_A_img,
                                            batch_B_img,
                                            G_a2b_model,
                                            G_b2a_model,
                                            D_a_model,
                                            D_b_model)
                print("Epoch: {} [{}/{}] g_loss = {}, d_loss = {}".format(epoch, step + 1, train_idx, g_loss, d_loss))

                if count % 10 == 0:
                    for i in range(FLAGS.batch_size):
                        for j in range(FLAGS.frames):
                            fake_B = run_model(G_a2b_model, batch_A_img[:, j, :, :, :], False)
                            fake_A = run_model(G_b2a_model, batch_B_img[:, j, :, :, :], False)

                            fakeB = fake_B[i]
                            fakeA = fake_A[i]
                            realB = batch_B_img[i, j, :, :, :]
                            realA = batch_A_img[i, j, :, :, :]

                            B_folder = "C:/Users/Yuhwan/Pictures/sample/fake_B_{}/".format(count)
                            A_folder = "C:/Users/Yuhwan/Pictures/sample/fake_A_{}/".format(count)

                            realB_folder = "C:/Users/Yuhwan/Pictures/sample/real_B_{}/".format(count)
                            realA_folder = "C:/Users/Yuhwan/Pictures/sample/real_A_{}/".format(count)

                            if not os.path.isdir(B_folder):
                                os.makedirs(B_folder)
                            if not os.path.isdir(A_folder):
                                os.makedirs(A_folder)

                            if not os.path.isdir(realB_folder):
                                os.makedirs(realB_folder)
                            if not os.path.isdir(realA_folder):
                                os.makedirs(realA_folder)

                            plt.imsave(B_folder + "fake_B_{}.jpg".format(j), fakeB * 0.5 + 0.5)
                            plt.imsave(A_folder + "fake_A_{}.jpg".format(j), fakeA * 0.5 + 0.5)
                            plt.imsave(realB_folder + "real_B_{}.jpg".format(j), realB * 0.5 + 0.5)
                            plt.imsave(realA_folder + "reaa_A_{}.jpg".format(j), realA * 0.5 + 0.5)


                count += 1

                        


if __name__ == "__main__":
    main()