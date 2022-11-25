# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
#import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class YelpReviewFull(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "yelp_review_full"

        self.task_type = "regression"


    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list

        train_lines = map_hf_dataset_to_list(dataset, "train.csv")
        test_lines = map_hf_dataset_to_list(dataset, "test.csv")

        return train_lines, test_lines

    # def map_hf_dataset_to_list(self, hf_dataset, split_name):
    #     lines = []
    #     for datapoint in hf_dataset[split_name]:
    #         lines.append((datapoint["text"].replace("\\n", " "), str(datapoint["label"] + 1)))
    #     return lines
    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        abs_path = os.path.join(hf_dataset, split_name)
        with open(abs_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            listReader =list(reader)
            for datapoint in listReader:
                lines.append((datapoint[1].replace("\\n", " "), str(datapoint[0])))
        return lines

    def load_dataset(self):
        #return datasets.load_dataset('yelp_review_full')
        return "/nas/wab/MetaICL/MetaICL_fail_data/yelp_review_full_csv/"
def main():
    dataset = YelpReviewFull()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
