#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate bdd100k training dataset masks

Change by OpenAI

Dataset: https://bdd-data.berkeley.edu

"""

import argparse
import glob
import json
import os
import os.path as ops
import shutil

import cv2
import numpy as np


def init_args():
    """
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir", type=str, help="The origin path of unzipped bdd100k dataset"
    )
    parser.add_argument("--val", type=bool, help="Tag for validation set", default=True)
    parser.add_argument("--test", type=bool, help="Tag for test set", default=False)

    return parser.parse_args()


def process_json_file(json_file_path, src_dir, ori_dst_dir, binary_dst_dir):
    """
    :param json_file_path:
    :param src_dir: origin clip file path
    :param ori_dst_dir:
    :param binary_dst_dir:
    :return:
    """
    assert ops.exists(json_file_path), "{:s} not exist".format(json_file_path)

    image_nums = len(os.listdir(ori_dst_dir))

    with open(json_file_path, "r") as file:
        data = json.load(file)
        for line_index, info_dict in enumerate(data):
            image_name = info_dict["name"]
            image_path = ops.join(src_dir, image_name)
            assert ops.exists(image_path), "{:s} not exist".format(image_path)

            image_name_new = "{:s}.png".format(
                "{:d}".format(line_index + image_nums).zfill(4)
            )

            src_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            dst_binary_image = np.zeros(
                [src_image.shape[0], src_image.shape[1]], np.uint8
            )

            for label in info_dict["labels"]:
                for poly in label["poly2d"]:
                    lane_pts = np.array(poly["vertices"], np.int32)
                    lane_pts = lane_pts.reshape((-1, 1, 2))
                    cv2.polylines(
                        dst_binary_image,
                        [lane_pts],
                        isClosed=False,
                        color=255,
                        thickness=5,
                    )

            dst_binary_image_path = ops.join(binary_dst_dir, image_name_new)
            dst_rgb_image_path = ops.join(ori_dst_dir, image_name_new)

            cv2.imwrite(dst_binary_image_path, dst_binary_image)
            cv2.imwrite(dst_rgb_image_path, src_image)

            print("Process {:s} success".format(image_name))


def gen_train_sample(src_dir, b_gt_image_dir, image_dir):
    """
    generate sample index file
    :param src_dir:
    :param b_gt_image_dir:
    :param image_dir:
    :return:
    """

    with open("{:s}/training/train.txt".format(src_dir), "w") as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith(".png"):
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            assert ops.exists(image_path), "{:s} not exist".format(image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None:
                print("Image pair: {:s} corrupted".format(image_name))
                continue
            else:
                info = "{:s} {:s}".format(image_path, binary_gt_image_path)
                file.write(info + "\n")
    return


def gen_train_val_sample(src_dir, b_gt_image_dir, image_dir):
    """
    generate sample index file
    :param src_dir:
    :param b_gt_image_dir:
    :param image_dir:
    :return:
    """
    with open("{:s}/training/train.txt".format(src_dir), "w") as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith(".png"):
                continue
            if int(image_name.split(".")[0]) % 13 == 0:
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            assert ops.exists(image_path), "{:s} not exist".format(image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None:
                print("Image pair: {:s} corrupted".format(image_name))
                continue
            else:
                info = "{:s} {:s}".format(image_path, binary_gt_image_path)
                file.write(info + "\n")

    with open("{:s}/training/val.txt".format(src_dir), "w") as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith(".png"):
                continue
            if int(image_name.split(".")[0]) % 13 != 0:
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            assert ops.exists(image_path), "{:s} not exist".format(image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None:
                print("Image pair: {:s} corrupted".format(image_name))
                continue
            else:
                info = "{:s} {:s}".format(image_path, binary_gt_image_path)
                file.write(info + "\n")
    return


def gen_test_sample(src_dir, b_gt_image_dir, image_dir):
    """
    generate sample index file
    :param src_dir:
    :param b_gt_image_dir:
    :param image_dir:
    :return:
    """

    with open("{:s}/testing/test.txt".format(src_dir), "w") as file:

        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith(".png"):
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            assert ops.exists(image_path), "{:s} not exist".format(image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None:
                print("Image pair: {:s} corrupted".format(image_name))
                continue
            else:
                info = "{:s} {:s}".format(image_path, binary_gt_image_path)
                file.write(info + "\n")
    return


def process_bdd100k_dataset(src_dir, val_tag, test_tag):
    """
    :param src_dir:
    :return:
    """
    training_folder_path = ops.join(src_dir, "training")
    testing_folder_path = ops.join(src_dir, "testing")

    os.makedirs(training_folder_path, exist_ok=True)
    os.makedirs(testing_folder_path, exist_ok=True)

    for json_label_path in glob.glob("{:s}/lane_labels.json".format(src_dir)):
        json_label_name = ops.split(json_label_path)[1]

        shutil.copyfile(
            json_label_path, ops.join(training_folder_path, json_label_name)
        )

    gt_image_dir = ops.join(training_folder_path, "gt_image")
    gt_binary_dir = ops.join(training_folder_path, "gt_binary_image")

    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(gt_binary_dir, exist_ok=True)

    for json_label_path in glob.glob("{:s}/*.json".format(training_folder_path)):
        process_json_file(json_label_path, src_dir, gt_image_dir, gt_binary_dir)

    if val_tag == False:
        gen_train_sample(src_dir, gt_binary_dir, gt_image_dir)
    else:
        gen_train_val_sample(src_dir, gt_binary_dir, gt_image_dir)

    if test_tag == True:
        gt_image_dir_test = ops.join(testing_folder_path, "gt_image")
        gt_binary_dir_test = ops.join(testing_folder_path, "gt_binary_image")

        os.makedirs(gt_image_dir_test, exist_ok=True)
        os.makedirs(gt_binary_dir_test, exist_ok=True)

        for json_label_path in glob.glob("{:s}/*.json".format(testing_folder_path)):
            process_json_file(
                json_label_path, src_dir, gt_image_dir_test, gt_binary_dir_test
            )

        gen_test_sample(src_dir, gt_binary_dir_test, gt_image_dir_test)

    return


if __name__ == "__main__":
    args = init_args()

    process_bdd100k_dataset(args.src_dir, args.val, args.test)
