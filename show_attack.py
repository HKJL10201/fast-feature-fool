"""
The code takes in the image list and generated perturbation and calculates the
fooling rate and classification accuracy on the ILSVRC validation set (50K images)
"""

from nets.vgg_f import vggf
from nets.caffenet import caffenet
from nets.vgg_16 import vgg16
from nets.vgg_19 import vgg19
from nets.googlenet import googlenet
from misc.utils import img_preprocess, upsample, downsample
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def build_class_dic(filename):
    dic = [""]
    with open(filename, "r") as f:
        for line in f:
            words = line.split()
            dic.append(words[2])
    return dic


def save_img(img, name):
    im = np.transpose(img[0], (0, 1, 2))
    im = Image.fromarray(np.uint8(im))
    im.save("imgs/attack/%s.jpg" % name)


def choose_net(network, adv_image):
    MAP = {
        "vggf": vggf,
        "caffenet": caffenet,
        "vgg16": vgg16,
        "vgg19": vgg19,
        "googlenet": googlenet,
    }
    if network == "caffenet":
        size = 227
    else:
        size = 224
    # loading the perturbation
    pert_load = np.load(adv_image, allow_pickle=True, encoding="latin1")
    # preprocessing if necessary
    if pert_load.shape[1] == 224 and size == 227:
        pert_load = upsample(np.squeeze(pert_load))
    elif pert_load.shape[1] == 227 and size == 224:
        pert_load = downsample(np.squeeze(pert_load))
    elif pert_load.shape[1] not in [224, 227]:
        raise Exception("Invalid size of input perturbation")
    adv_image = tf.constant(pert_load, dtype="float32")
    # placeholder to pass image
    input_image = tf.placeholder(
        shape=[None, size, size, 3], dtype="float32", name="input_image"
    )
    input_batch = tf.concat([input_image, tf.add(input_image, adv_image)], 0)

    return MAP[network](input_batch), input_image, tf.add(input_image, adv_image)


def classify(net, in_im, adv_smp, net_name, im_list, gt_labels, index):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    imgs = open(im_list).readlines()
    gt_labels = open(gt_labels).readlines()
    labels = build_class_dic("map_clsloc.txt")

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in index:
            name = imgs[i]
            if net_name == "caffenet":
                im = img_preprocess(name.strip(), size=227)
            else:
                im = img_preprocess(name.strip())
            softmax_scores = sess.run(net["prob"], feed_dict={in_im: im})
            inp_img = sess.run(in_im, feed_dict={in_im: im})
            adv_img = sess.run(adv_smp, feed_dict={in_im: im})
            ori_predict = np.argmax(softmax_scores[0])
            adv_predict = np.argmax(softmax_scores[1])
            true_label = int(gt_labels[i].strip())
            print(labels[true_label], labels[ori_predict], labels[adv_predict])
            save_img(inp_img, "-".join([labels[true_label], labels[ori_predict]]))
            save_img(adv_img, "--".join([labels[true_label], labels[adv_predict]]))


def main():
    network = "googlenet"
    adv_im = "perturbations/perturbation_googlenet.npy"
    img_list = "img_list.txt"
    gt_labels = "gt_labels.txt"
    index = [6]

    net, inp_im, adv_smp = choose_net(network, adv_im)
    classify(net, inp_im, adv_smp, network, img_list, gt_labels, index)


if __name__ == "__main__":
    main()
