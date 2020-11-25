# -*- coding: utf-8 -*-
from absl import flags
from fix_gener_discrim_model import *

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os

flags.DEFINE_integer("img_size", 64, "Height and Width")

flags.DEFINE_integer("img_ch", 3, "Channels")

flags.DEFINE_integer("img_fr", 16, "frames")

flags.DEFINE_integer("batch_size", 1, "Batch size")

flags.DEFINE_integer("classes", 14, "Number of classes")

flags.DEFINE_float("lr", 0.0002, "Learning rate")

flags.DEFINE_integer("epochs", 100, "Total epochs in training")

flags.DEFINE_string("tr_img_path", "D:/[1]DB/[4]etc_experiment/Fall_dataset/[5]TCL_dataset/GAN_data/divide/preprocessed/Walking_fall_stand/train/B_train", "Training image path")

flags.DEFINE_string("input_img_path", "D:/[1]DB/[4]etc_experiment/Fall_dataset/[5]TCL_dataset/GAN_data/divide/preprocessed/Walking_fall_stand/train/A_train", "Input image path")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_string("save_checkpoint", "", "Save checkpoint path")

flags.DEFINE_string("save_images", "C:/Users/Yuhwan/Pictures/sample", "Save training sample path")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint path to restore")

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

    input_noise = tf.random.uniform([FLAGS.img_fr, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch], dtype=tf.dtypes.float32, seed=1234) / 0.5  - 1.
    return input_img, input_noise, label

#@tf.function
def run_model(model, data, training=True):
    return model(data, training=training)

def cal_loss(ge_model, dis_model, batch_images, batch_labels, input_images):

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_noise = run_model(ge_model, input_images, True)
        fake_ = run_model(ge_model, batch_images, True) # CycleGAN에 있는것철 id loss를 보고 만든것

        D_real = run_model(dis_model, batch_images, True)
        D_fake = run_model(dis_model, fake_noise, True)

        g_id_loss = tf.reduce_mean((fake_noise - batch_images)**2) * 10.
        g_adv_loss = tf.reduce_mean((D_fake - tf.ones_like(D_fake))**2)
        g_loss = g_id_loss + g_adv_loss

        d_loss = (tf.reduce_mean((D_fake - tf.zeros_like(D_fake))**2) + tf.reduce_mean((D_real - tf.ones_like(D_real))**2)) / 2.

    g_grads = g_tape.gradient(g_loss, ge_model.trainable_variables)
    d_grads = d_tape.gradient(d_loss, dis_model.trainable_variables)

    g_optim.apply_gradients(zip(g_grads, ge_model.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, dis_model.trainable_variables))

    return g_loss, d_loss

def main():
    ge_model = generator()
    dis_model = discriminator()
    ge_model.summary()
    dis_model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(ge_model=ge_model, dis_model=dis_model,
                                   g_optim=g_optim, d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("==============================")
            print("Restored the checkpoint!!!!!!!")
            print("==============================")

    if FLAGS.train:
        count = 0
        img_path = os.listdir(FLAGS.tr_img_path)
        img_path = [folder for folder in img_path if folder !=".DS_Store"]
        img_path = all_data(img_path, FLAGS.tr_img_path)

        input_path = os.listdir(FLAGS.input_img_path)
        input_path = [folder for folder in input_path if folder !=".DS_Store"]
        input_path = all_data(input_path, FLAGS.input_img_path)

        for epoch in range(FLAGS.epochs):
            tr_img, tr_lab = zip(*img_path)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab, dtype=np.int32)

            input_img, input_lab = zip(*input_path)
            input_img, input_lab = np.array(input_img), np.array(input_lab, dtype=np.int32)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_lab))
            tr_gener = tr_gener.map(train_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            in_gener = tf.data.Dataset.from_tensor_slices((input_img, input_lab))
            in_gener = in_gener.shuffle(len(input_img))
            in_gener = in_gener.map(train_func)
            in_gener = in_gener.batch(FLAGS.batch_size)
            in_gener = in_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_idx = min(len(tr_lab), len(input_lab)) // FLAGS.batch_size
            tr_iter = iter(tr_gener)
            in_iter = iter(in_gener)
            for step in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                input_images, _, input_labels = next(in_iter)

                g_loss, d_loss = cal_loss(ge_model, dis_model, batch_images, batch_labels, input_images)
                print("Epoch: {} [{}/{}] G_loss = {}, D_loss = {}".format(epoch, step + 1, tr_idx, g_loss, d_loss))

                if count % 10 == 0:
                    fake_img = run_model(ge_model, input_images, False)
                    for i in range(FLAGS.img_fr):
                        fakeImg = fake_img[0][i]
                        gr_imgs = batch_images[0][i]

                        fake_folder = FLAGS.save_images + "/" + "fake_img_{}/".format(count)
                        gt_folder = FLAGS.save_images + "/" + "gt_img_{}/".format(count)

                        if not os.path.isdir(fake_folder):
                            os.makedirs(fake_folder)
                        if not os.path.isdir(gt_folder):
                            os.makedirs(gt_folder)

                        plt.imsave(fake_folder + "fake_img_{}.jpg".format(i), fakeImg * 0.5 + 0.5)
                        plt.imsave(gt_folder + "gt_img_{}.jpg".format(i), gr_imgs * 0.5 + 0.5)

                count += 1




if __name__ == "__main__":
    main()