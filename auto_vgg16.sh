#!/bin/bash
python evaluate.py --network vggf --adv_im perturbations/perturbation_vgg16.npy --img_list img_list.txt --gt_labels gt_labels.txt
python evaluate.py --network caffenet --adv_im perturbations/perturbation_vgg16.npy --img_list img_list.txt --gt_labels gt_labels.txt
python evaluate.py --network googlenet --adv_im perturbations/perturbation_vgg16.npy --img_list img_list.txt --gt_labels gt_labels.txt
python evaluate.py --network vgg16 --adv_im perturbations/perturbation_vgg16.npy --img_list img_list.txt --gt_labels gt_labels.txt
python evaluate.py --network vgg19 --adv_im perturbations/perturbation_vgg16.npy --img_list img_list.txt --gt_labels gt_labels.txt
