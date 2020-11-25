# -*- coding:utf-8 -*-
from random import shuffle, random
from absl import flags
from fix_gener_discrim_model_2 import *

import matplotlib.pyplot as plt
#import tensorflow as tf
import numpy as np
import sys
import os

flags.DEFINE_string("tr_img_path", "D:/[1]DB/[4]etc_experiment/Fall_dataset/[5]TCL_dataset/GAN_data/divide/preprocessed/Walking_fall_stand/train/B_train", "Training image path")

flags.DEFINE_integer("img_size", 256, "Image --> height and width")

flags.DEFINE_integer("img_ch", 3, "Image --> channels")

flags.DEFINE_integer("img_fr", 16, "Image --> frames")

flags.DEFINE_integer("classes", 14, "Number of classes")

flags.DEFINE_float("lr", 0.0002, "Learning rate")

flags.DEFINE_integer("batch_size", 1, "Training batch size")

flags.DEFINE_integer("epochs", 100, "Total epochs")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint files")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_string("save_checkpoint", "", "Save checkpoint path")

flags.DEFINE_string("save_images", "C:/Users/Yuhwan/Pictures/sample", "Save sample images path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

g_mapping_optim = tf.keras.optimizers.Adam(FLAGS.lr)
g_sythesis_optim = tf.keras.optimizers.Adam(FLAGS.lr)
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

    input_noise = tf.random.uniform([FLAGS.img_fr, 256], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=1234)
    return input_img, input_noise, label

@tf.function
def gradient_penalty(fake_img, real_img, d_model):
    alpha = tf.random.uniform(shape=[FLAGS.batch_size, 1, 1, 1], minval=0., maxval=1.)
    differences = (fake_img - real_img)
    interpolates = real_img + (alpha * differences)
    gradients = tf.gradients(d_model(interpolates, True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    return gradient_penalty

def cal_loss(mapping_G_,synthesis_G_,StyleGAN_for_D_,batch_images,batch_noise):

    g_loss = 0
    d_loss = 0
    for i in range(FLAGS.img_fr):
        with tf.GradientTape(persistent=True) as g_tape, tf.GradientTape() as d_tape:

            g_mapping_output = mapping_G_(batch_noise[:, i], True)
            g_fake_output = synthesis_G_(g_mapping_output, True)

            d_real = StyleGAN_for_D_(batch_images[:, i], True)
            d_fake = StyleGAN_for_D_(g_fake_output, True)

            #d_adv_loss = tf.math.abs(tf.reduce_mean(d_fake) - tf.reduce_mean(d_real))
            #d_adv_loss = d_adv_loss + 10. * gradient_penalty(g_fake_output, batch_images[:, i], StyleGAN_for_D_)
            #g_adv_loss = -tf.reduce_mean(d_fake)

            d_adv_loss = (tf.reduce_mean((d_real - tf.ones_like(d_real))**2) \
                        + tf.reduce_mean((d_fake - tf.zeros_like(d_fake))**2)) / 2.

            g_adv_loss = tf.reduce_mean((d_fake - tf.ones_like(d_fake))**2) \
                        + tf.reduce_mean((g_fake_output - batch_images[:, i])**2) * 10.
        
        g_map_grads = g_tape.gradient(g_adv_loss, mapping_G_.trainable_variables)
        g_syn_grads = g_tape.gradient(g_adv_loss, synthesis_G_.trainable_variables)
        d_grads = d_tape.gradient(d_adv_loss, StyleGAN_for_D_.trainable_variables)

        g_mapping_optim.apply_gradients(zip(g_map_grads, mapping_G_.trainable_variables))
        g_sythesis_optim.apply_gradients(zip(g_syn_grads, synthesis_G_.trainable_variables))
        d_optim.apply_gradients(zip(d_grads, StyleGAN_for_D_.trainable_variables))

        g_loss += g_adv_loss
        d_loss += d_adv_loss

    g_loss /= FLAGS.img_fr
    d_loss /= FLAGS.img_fr

    return g_loss, d_loss

def main():

    mapping_G_ = mapping_G()
    mapping_G_.summary()
    synthesis_G_ = synthesis_G()
    synthesis_G_.summary()
    StyleGAN_for_D_ = StyleGAN_for_D()
    StyleGAN_for_D_.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(mapping_G_=mapping_G_,
                                   synthesis_G_=synthesis_G_,
                                   StyleGAN_for_D_=StyleGAN_for_D_,
                                   g_mapping_optim=g_mapping_optim,
                                   g_sythesis_optim=g_sythesis_optim,
                                   d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("==========================")
            print("Resotre!!!!!")
            print("==========================")

    if FLAGS.train:
        count = 0;
        img_path = os.listdir(FLAGS.tr_img_path)
        img_path = [folder for folder in img_path if folder !=".DS_Store"]
        img_path = all_data(img_path, FLAGS.tr_img_path)

        for epoch in range(FLAGS.epochs):
            tr_img, tr_lab = zip(*img_path)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab, dtype=np.int32)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(train_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_idx = len(tr_img) // FLAGS.batch_size
            tr_iter = iter(tr_gener)

            for step in range(tr_idx):
                batch_images, batch_noise, _ = next(tr_iter)

                g_loss, d_loss = cal_loss(mapping_G_,
                                         synthesis_G_,
                                         StyleGAN_for_D_,
                                         batch_images,
                                         batch_noise)
                print("Epoch: {} [{}/{}] G_loss = {}, D_loss = {}".format(epoch, step + 1, tr_idx, g_loss, d_loss))

                if count % 10 == 0:
                    for i in range(FLAGS.img_fr):
                        g_mapping_output = mapping_G_(batch_noise[:, i], True)
                        g_fake_output = synthesis_G_(g_mapping_output, True)

                        fake_folder = FLAGS.save_images + "/" + "fake_img_{}/".format(count)
                        gt_folder = FLAGS.save_images + "/" + "gt_img_{}/".format(count)

                        if not os.path.isdir(fake_folder):
                            os.makedirs(fake_folder)
                        if not os.path.isdir(gt_folder):
                            os.makedirs(gt_folder)

                        plt.imsave(gt_folder + "gt_img_{}.jpg".format(i), batch_images[0,i] * 0.5 + 0.5)
                        plt.imsave(fake_folder + "fake_img_{}.jpg".format(i), g_fake_output[0] * 0.5 + 0.5)

                if count % 500 ==0:
                    num_ = int(count // 500)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)

                    ckpt = tf.train.Checkpoint(mapping_G_=mapping_G_,
                                               synthesis_G_=synthesis_G_,
                                               StyleGAN_for_D_=StyleGAN_for_D_,
                                               g_mapping_optim=g_mapping_optim,
                                               g_sythesis_optim=g_sythesis_optim,
                                               d_optim=d_optim)
                    ckpt_dir = model_dir + "/" + "styleGAN_for_FALL_{}.ckpt".format(count)
                    ckpt.save(ckpt_dir)

                count += 1

if __name__ == "__main__":
    main()