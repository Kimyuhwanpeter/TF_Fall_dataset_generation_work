# -*- coding: utf-8 -*-
from absl import flags
from GAN_based_fall_detection_model import *
from GAN_based_fall_detection_model2 import *

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import sys

flags.DEFINE_string("tr_img_path", "D:/[1]DB/[4]etc_experiment/Fall_dataset/[5]TCL_dataset/GAN_data/divide/preprocessed/Walking_fall_stand/train/B_train", "Training image path")

flags.DEFINE_integer("input_size", 64, "Input height and width")

flags.DEFINE_integer("img_size", 112, "Real image height and width")

flags.DEFINE_integer("input_ch", 3, "Input channels")

flags.DEFINE_integer("frames", 16, "Real image's frames")

flags.DEFINE_integer("batch_size", 2, "Batch size")

flags.DEFINE_integer("epochs", 200, "Total epochs")

flags.DEFINE_float("lr", 0.0001, "Learnaing rate")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint files path")

flags.DEFINE_string("save_images", "C:/Users/Yuhwan/Pictures/sample", "")

flags.DEFINE_string("save_checkpoint", "", "")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

auto_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5, beta_2=0.5)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5, beta_2=0.5)

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
        img_ = tf.image.decode_png(img_, FLAGS.input_ch)
        img_ = tf.image.resize(img_, [FLAGS.img_size, FLAGS.img_size])

        img_ = img_ / 127.5 - 1.

        input.append(img_)

    input_img = input

    label = tf.one_hot(lab, 14)

    input_noise = tf.random.uniform([FLAGS.frames, FLAGS.img_size, FLAGS.img_size, FLAGS.input_ch], dtype=tf.dtypes.float32, seed=1234) / 0.5  - 1.
    return input_img, input_noise, label

@tf.function
def run_model(model, input, training=True):
    return model(input, training=training)

def mae_criterion(input, target):
    return tf.reduce_mean((input - target)**2)

def cal_loss(images, noise, autoencoder, discrim):
    with tf.GradientTape(persistent=True) as tape:
        noise_fake_img = run_model(autoencoder, noise, True)    # 이 noise 를 넣은게 약간의 흠인가? --> 그런것같다. noise말고 다른걸 넣어볼까?
        image_fake_img = run_model(autoencoder, images, True)
        D_real = run_model(discrim, images, True)
        D_fake = run_model(discrim, noise_fake_img, True)

        D_loss = ( mae_criterion(D_real, tf.ones_like(D_real)) + mae_criterion(D_fake, tf.zeros_like(D_fake)) )  / 2.

        G_adv_loss = mae_criterion(D_fake, tf.ones_like(D_fake))
        G_id_loss = tf.reduce_mean((images - image_fake_img)**2)
        G_loss = G_adv_loss + (G_id_loss * 10.)

    g_grads = tape.gradient(G_loss, autoencoder.trainable_variables)
    d_grads = tape.gradient(D_loss, discrim.trainable_variables)

    auto_optim.apply_gradients(zip(g_grads, autoencoder.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, discrim.trainable_variables))
    return G_loss, D_loss

def main():
    autoencoder = AUTOENCODER_v2(input_shape=(FLAGS.frames, FLAGS.img_size, FLAGS.img_size, FLAGS.input_ch))
    discrim = discriminator(input_shape=(FLAGS.frames, FLAGS.img_size, FLAGS.img_size, FLAGS.input_ch))
    autoencoder.summary()
    discrim.summary()
    # discriminator는 지금 2d conv로 이루어져있으며

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(autoencoder=autoencoder,
                                   discrim=discrim,
                                   auto_optim=auto_optim,
                                   d_optim=d_optim)
        ckpt_manger = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manger.latest_checkpoint:
            ckpt.restore(ckpt_manger.latest_checkpoint)
            print("========================================")
            print("Restored the latest checkpoint files!!!!")
            print("========================================")

    if FLAGS.train:
        count = 0
        img_path = os.listdir(FLAGS.tr_img_path)
        img_path = [folder for folder in img_path if folder !=".DS_Store"]
        img_path = all_data(img_path, FLAGS.tr_img_path)

        tr_img, tr_lab = zip(*img_path)
        tr_img, tr_lab = np.array(tr_img), np.array(tr_lab, dtype=np.int32)
        
        for epoch in range(FLAGS.epochs):
            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(train_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(tr_img) // FLAGS.batch_size

            for step in range(tr_idx):
                batch_images, batch_noise, _ = next(tr_iter)

                # Define loss func  (discrim가 2D conv로 이루어져있기 때문에 배치 내에 있든 이미지들에 대해 각각 입력해주어야한다)
                g_loss, d_loss = cal_loss(batch_images, batch_noise, autoencoder, discrim)
                
                print("Epoch: {} [{}/{}] G_loss = {}, D_loss = {}".format(epoch, step + 1, tr_idx, g_loss, d_loss))

                if count % 50 == 0:
                    fake_img = run_model(autoencoder, batch_noise, False)
                    for i in range(FLAGS.frames):
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
                    num_ = int(count / 500)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                    ckpt = tf.train.Checkpoint(autoencoder=autoencoder,
                                               discrim=discrim,
                                               auto_optim=auto_optim,
                                               d_optim=d_optim)
                    ckpt_dir = model_dir + "/New_GAN_{}.ckpt".format(count)

                    ckpt.save(ckpt_dir)

                count += 1

if __name__ == "__main__":
    main()