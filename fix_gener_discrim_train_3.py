# -*- coding: utf-8 -*-
from fix_gener_discrim_model_3 import *
from absl import flags

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

flags.DEFINE_string("A_tr_img_path", "D:/[1]DB/[4]etc_experiment/Fall_dataset/[5]TCL_dataset/GAN_data/divide/preprocessed/Walking_fall_stand/train/A_train", "A training image path")

flags.DEFINE_string("B_tr_img_path", "D:/[1]DB/[4]etc_experiment/Fall_dataset/[5]TCL_dataset/GAN_data/divide/preprocessed/Walking_fall_stand/train/B_train", "B training image path")

flags.DEFINE_integer("img_size", 112, "Height and width")

flags.DEFINE_integer("img_ch", 3, "Channels")

flags.DEFINE_integer("img_fr", 16, "Image frames")

flags.DEFINE_integer("epochs", 200, "Training epochs")

flags.DEFINE_integer("classes", 14, "Number of classes")

flags.DEFINE_integer("batch_size", 1, "Batch size")

flags.DEFINE_float("lr", 0.0002, "Learning rate")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_string("save_images", "C:/Users/Yuhwan/Pictures/sample", "Save sample images path")

flags.DEFINE_string("save_checkpoint", "", "")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint file path")

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
            real_img = [int(img.split(".")[0]) for img in real_img]
            real_img.sort()
            real_img = [str(img) + ".png" for img in real_img]
            real_img = [folder_list_[j] + "/" + name for name in real_img]
            img.append(real_img)

    data = list(zip(img, lab))

    return data

def train_func(img, lab):

    input = []
    for i in range(0, FLAGS.img_fr):

        img_ = tf.io.read_file(img[i])
        img_ = tf.image.decode_png(img_, FLAGS.img_ch)
        img_ = tf.image.resize(img_, [FLAGS.img_size, FLAGS.img_size])

        img_ = img_ / 127.5 - 1.

        input.append(img_)

    input_img = input

    label = tf.one_hot(lab, FLAGS.classes)

    return input_img, label

@tf.function
def run_model(model, img, training=True):
    return model(img, training=training)

def cal_loss(A2B_model, B2A_model, A_model, B_model, A_img, B_img):

    g_loss = 0
    d_loss = 0
    for i in range(FLAGS.img_fr):
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

            fake_B = run_model(A2B_model, A_img[:, i], True)
            fake_A = run_model(B2A_model, B_img[:, i], True)

            fake_A_ = run_model(B2A_model, fake_B, True)
            fake_B_ = run_model(A2B_model, fake_A, True)

            DA_real = run_model(A_model, A_img[:, i], True)
            DB_real = run_model(B_model, B_img[:, i], True)

            DA_fake = run_model(A_model, fake_A, True)
            DB_fake = run_model(B_model, fake_B, True)

            G_cycle_loss = (tf.reduce_mean(tf.abs(A_img[:, i] - fake_A_)) + tf.reduce_mean(tf.abs(B_img[:, i] - fake_B_))) * 10.0
            G_adv_loss = tf.reduce_mean((DA_fake - tf.ones_like(DA_fake))**2) + tf.reduce_mean((DB_fake - tf.ones_like(DB_fake))**2) \
                        + G_cycle_loss
            
            D_adv_loss = ( tf.reduce_mean((DA_real - tf.ones_like(DA_real))**2) + tf.reduce_mean((DA_fake - tf.zeros_like(DA_fake))**2) ) * 0.5 \
                        + ( tf.reduce_mean((DB_real - tf.ones_like(DB_real))**2) + tf.reduce_mean((DB_fake - tf.zeros_like(DB_fake))**2) ) * 0.5

        g_grads = g_tape.gradient(G_adv_loss, A2B_model.trainable_variables + B2A_model.trainable_variables)
        d_grads = d_tape.gradient(D_adv_loss, A_model.trainable_variables + B_model.trainable_variables)

        g_optim.apply_gradients(zip(g_grads, A2B_model.trainable_variables + B2A_model.trainable_variables))
        d_optim.apply_gradients(zip(d_grads, A_model.trainable_variables + B_model.trainable_variables))

        g_loss += G_adv_loss
        d_loss += D_adv_loss

    g_loss /= FLAGS.img_fr
    d_loss /= FLAGS.img_fr

    return g_loss, d_loss

def main():
    A2B_model = fix_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
    A2B_model.summary()
    B2A_model = fix_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
    B2A_model.summary()
    A_model = fix_discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
    A_model.summary()
    B_model = fix_discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
    B_model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_model=A2B_model, B2A_model=B2A_model,
                                   A_model=A_model, B_model=B_model,
                                   g_optim=g_optim, d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("==============")
            print("Restored!!!!!!")
            print("==============")
    if FLAGS.train:
        count = 0
        A_img_path = os.listdir(FLAGS.A_tr_img_path)
        A_img_path = [folder for folder in A_img_path if folder !=".DS_Store"]
        A_img_path = all_data(A_img_path, FLAGS.A_tr_img_path)

        B_img_path = os.listdir(FLAGS.B_tr_img_path)
        B_img_path = [folder for folder in B_img_path if folder !=".DS_Store"]
        B_img_path = all_data(B_img_path, FLAGS.B_tr_img_path)

        for epoch in range(FLAGS.epochs):
            A_tr_img, A_tr_lab = zip(*A_img_path)
            A_tr_img, A_tr_lab = np.array(A_tr_img), np.array(A_tr_lab, dtype=np.int32)
            B_tr_img, B_tr_lab = zip(*B_img_path)
            B_tr_img, B_tr_lab = np.array(B_tr_img), np.array(B_tr_lab, dtype=np.int32)

            A_gener = tf.data.Dataset.from_tensor_slices((A_tr_img, A_tr_lab))
            A_gener = A_gener.shuffle(len(A_tr_img))
            A_gener = A_gener.map(train_func)
            A_gener = A_gener.batch(FLAGS.batch_size)
            A_gener = A_gener.prefetch(tf.data.experimental.AUTOTUNE)

            B_gener = tf.data.Dataset.from_tensor_slices((B_tr_img, B_tr_lab))
            B_gener = B_gener.shuffle(len(B_tr_img))
            B_gener = B_gener.map(train_func)
            B_gener = B_gener.batch(FLAGS.batch_size)
            B_gener = B_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_idx = min(len(A_tr_img), len(B_tr_img)) // FLAGS.batch_size
            A_iter = iter(A_gener)
            B_iter = iter(B_gener)
            for step in range(tr_idx):
                A_images, _ = next(A_iter)
                B_images, _ = next(B_iter)

                g_loss, d_loss = cal_loss(A2B_model, B2A_model, A_model, B_model, A_images, B_images)

                print("Epoch(s): {} [{}/{}] g_loss = {}, d_loss = {}".format(epoch, step + 1, tr_idx, g_loss, d_loss))

                if count % 100 == 0:
                    for i in range(FLAGS.img_fr):
                        fake_B = run_model(A2B_model, A_images[:, i], False)
                        fake_A = run_model(B2A_model, B_images[:, i], False)

                        fakeA_folder = FLAGS.save_images + "/" + "fakeA_img_{}/".format(count)
                        fakeB_folder = FLAGS.save_images + "/" + "fakeB_img_{}/".format(count)
                        realA_folder = FLAGS.save_images + "/" + "realA_img_{}/".format(count)
                        realB_folder = FLAGS.save_images + "/" + "realB_img_{}/".format(count)

                        if not os.path.isdir(fakeA_folder):
                            os.makedirs(fakeA_folder)
                        if not os.path.isdir(fakeB_folder):
                            os.makedirs(fakeB_folder)
                        if not os.path.isdir(realA_folder):
                            os.makedirs(realA_folder)
                        if not os.path.isdir(realB_folder):
                            os.makedirs(realB_folder)

                        plt.imsave(fakeA_folder + "fakeA_img_{}.jpg".format(i), fake_A[0] * 0.5 + 0.5)
                        plt.imsave(fakeB_folder + "fakeB_img_{}.jpg".format(i), fake_B[0] * 0.5 + 0.5)
                        plt.imsave(realA_folder + "realA_img_{}.jpg".format(i), A_images[0,i] * 0.5 + 0.5)
                        plt.imsave(realB_folder + "realB_img_{}.jpg".format(i), B_images[0,i] * 0.5 + 0.5)
                if count % 500 == 0:
                    num_ = int(count // 500)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                    ckpt = tf.train.Checkpoint(A2B_model=A2B_model, B2A_model=B2A_model,
                                               A_model=A_model, B_model=B_model,
                                               g_optim=g_optim, d_optim=d_optim)
                    ckpt_dir = model_dir + "/" + "Fall_GAN_{}.ckpt".format(count)
                    ckpt.save(ckpt_dir)

                count += 1

if __name__ == "__main__":
    main()