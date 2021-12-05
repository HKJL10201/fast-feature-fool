#!/bin/bash
python evaluate.py --network vggf --adv_im perturbations/perturbation_vggf.npy --img_list img_list.txt --gt_labels gt_labels.txt