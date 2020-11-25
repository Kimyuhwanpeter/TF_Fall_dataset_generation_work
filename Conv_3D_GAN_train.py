# -*- coding:utf-8 -*-
from Conv_3D_GAN_model import *
from random import shuffle, random

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"load_size": 142,

                           "input_size": 112,

                           "frames": 16,

                           "channels": 3,

                           "batch_size": 2,

                           "epochs": 200,

                           "lr": 0.0002,

                           "img_path": "D:/[1]DB/[4]etc_experiment/Fall_dataset/[5]TCL_dataset/GAN_data/divide/preprocessed/Walking_fall_stand/train/A_train",

                           "gt_img_path": "D:/[1]DB/[4]etc_experiment/Fall_dataset/[5]TCL_dataset/GAN_data/divide/preprocessed/Walking_fall_stand/train/B_train",

                           "train": True,

                           "pre_checkpoint": False,

                           "pre_checkpoint_path": "",

                           "save_checkpoint": "",

                           "save_images": "",

                           "save_graphs": ""})

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
    for i in range(0, FLAGS.frames):

        img_ = tf.io.read_file(img[i])
        img_ = tf.image.decode_png(img_, FLAGS.channels)
        img_ = tf.image.resize(img_, [FLAGS.load_size, FLAGS.load_size])
        img_ = tf.image.random_crop(img_, [FLAGS.input_size, FLAGS.input_size, FLAGS.channels])

        img_ = img_ / 127.5 - 1.

        input.append(img_)

    input_img = input

    #label = lab     # for binary
    label = tf.one_hot(lab, 14)    # for multi

    return input_img, label

def abs_criterion(input, target):
    return tf.reduce_mean(tf.abs(input - target))

def mae_criterion(input, target):
    return tf.reduce_mean((input - target)**2)

@tf.function
def cal_loss(tr_images, gt_images, first_gener, discrim, final_gener):
    with tf.GradientTape(persistent=True) as tape:
        fake_out_encoder = first_gener(tr_images, True)
        fake_out_discri = discrim(fake_out_encoder, True)
        fake_img = final_gener(fake_out_discri, True)

        real_out_encoder = first_gener(gt_images, True)
        real_out_discri = discrim(real_out_encoder, True)

        g_adv_loss = mae_criterion(fake_out_discri, tf.ones_like(fake_out_discri))
        d_adv_loss = (mae_criterion(fake_out_discri, tf.zeros_like(fake_out_discri)) \
                        + mae_criterion(real_out_discri, tf.ones_like(real_out_discri))) * 0.5

        g_id_loss = mae_criterion(fake_img, gt_images)

        g_loss = g_adv_loss + g_id_loss

    g_grads_1 = tape.gradient(g_id_loss, first_gener.trainable_variables)
    g_grads_2 = tape.gradient(g_id_loss, discrim.trainable_variables)
    g_grads_3 = tape.gradient(g_id_loss, final_gener.trainable_variables)

    d_grads_1 = tape.gradient(d_adv_loss, first_gener.trainable_variables)
    d_grads_2 = tape.gradient(d_adv_loss, discrim.trainable_variables)

    g_optim.apply_gradients(zip(g_grads_1, first_gener.trainable_variables))
    g_optim.apply_gradients(zip(g_grads_2, discrim.trainable_variables))
    g_optim.apply_gradients(zip(g_grads_3, final_gener.trainable_variables))

    d_optim.apply_gradients(zip(d_grads_1, first_gener.trainable_variables))
    d_optim.apply_gradients(zip(d_grads_2, discrim.trainable_variables))

    return g_loss, d_adv_loss

def main():
    first_gener = generator_3D(input_shape=(FLAGS.frames, FLAGS.input_size, FLAGS.input_size, FLAGS.channels))
    discrim = discriminator(input_shape=(FLAGS.frames, 14, 14, 256))
    final_gener = gener_2_dis_2_gener(input_shape=(FLAGS.frames, 56, 56, 1))
    # 모델에 대해선믄 조금만 더 생각해보자!

    first_gener.summary()
    discrim.summary()
    final_gener.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(first_gener=first_gener,
                                   final_gener=final_gener,
                                   discrim=discrim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("* Restored the latest checkpoint")

    if FLAGS.train:
        count = 0

        for epoch in range(FLAGS.epochs):
            img_path = os.listdir(FLAGS.img_path)
            img_path = [folder for folder in img_path if folder !=".DS_Store"]
            img_path = all_data(img_path, FLAGS.img_path)
            print("================================")
            print("Number of train dataset: {}".format(len(img_path)))
            tr_img, tr_lab = zip(*img_path)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab, dtype=np.int32)

            gt_img_path = os.listdir(FLAGS.gt_img_path)
            gt_img_path = [folder for folder in gt_img_path if folder != ".DS_Store"]
            gt_img_path = all_data(gt_img_path, FLAGS.gt_img_path)
            print("Number of target dataset: {}".format(len(gt_img_path)))
            print("================================")
            gt_img, gt_lab = zip(*gt_img_path)
            gt_img, gt_lab = np.array(gt_img), np.array(gt_lab, dtype=np.int32)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(train_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            gt_gener = tf.data.Dataset.from_tensor_slices((gt_img, gt_lab))
            gt_gener = gt_gener.shuffle(len(gt_img))
            gt_gener = gt_gener.map(train_func)
            gt_gener = gt_gener.batch(FLAGS.batch_size)
            gt_gener = gt_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_idx = min(len(tr_img), len(gt_img)) // FLAGS.batch_size
            tr_iter = iter(tr_gener)
            gt_iter = iter(gt_gener)
            for step in range(tr_idx):

                tr_images, tr_labels = next(tr_iter)
                gt_images, gt_labels = next(gt_iter)

                g_loss, d_loss = cal_loss(tr_images, gt_images,first_gener, discrim, final_gener)

                print(g_loss, d_loss)

                if count % 500 == 0:
                    fake_out_encoder = first_gener(tr_images, False)
                    fake_out_discri = discrim(fake_out_encoder, False)
                    fake_img = final_gener(fake_out_discri, False)

                    num = int(count // 500)                                        
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num)
                    if not os.path.isdir(model_dir):
                        os.makedirs(model_dir)
                        print("Make {} files to save checkpoint files".format(num))
                    ckpt = tf.train.Checkpoint(first_gener=first_gener,
                                                final_gener=final_gener,
                                                discrim=discrim)
                    ckpt_dir = model_dir + "/" + "New_fall_GAN_{}.ckpt".format(count)
                    ckpt.save(ckpt_dir)
                        

                    for i in range(FLAGS.frames):
                        fakeImg = fake_img[0][i]
                        tr_imgs = tr_images[0][i]
                        gr_imgs = gt_images[0][i]

                        fake_folder = FLAGS.save_images + "/" + "fake_img_{}/".format(count)
                        train_folder = FLAGS.save_images + "/" + "train_img_{}/".format(count)
                        gt_folder = FLAGS.save_images + "/" + "gt_img_{}/".format(count)

                        if not os.path.isdir(fake_folder):
                            os.makedirs(fake_folder)
                        if not os.path.isdir(train_folder):
                            os.makedirs(train_folder)
                        if not os.path.isdir(gt_folder):
                            os.makedirs(gt_folder)

                        plt.imsave(fake_folder + "fake_img_{}.jpg".format(i), fakeImg * 0.5 + 0.5)
                        plt.imsave(train_folder + "train_img_{}.jpg".format(i), tr_imgs * 0.5 + 0.5)
                        plt.imsave(gt_folder + "gt_img_{}.jpg".format(i), gr_imgs * 0.5 + 0.5)

                count += 1


if __name__ == "__main__":
    main()