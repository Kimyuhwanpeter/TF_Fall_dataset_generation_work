# -*- coduing: utf-8 -*-
from absl import flags

import tensorflow as tf
import numpy as np
import os
import sys

flags.DEFINE_string("img_path", "D:/[1]DB/[4]etc_experiment/Fall_dataset/[5]TCL_dataset/train", "Data image path")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

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

def main():
    data_list = os.listdir(FLAGS.img_path)

    A_train_buf = []
    for data in data_list:
        if data == label_list[1][0]:

            path = FLAGS.img_path + "/" + data
            
            A_train_buf.append()
    
    

if __name__ == "__main__":
    main()