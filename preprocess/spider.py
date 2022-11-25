# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
#import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class Spider(FewshotGymTextToTextDataset):
    
    def __init__(self):
        self.hf_identifier = "spider"
        self.task_type = "text to text"
        self.license = "unknown"

    # def map_hf_dataset_to_list(self, hf_dataset, split_name):
    #     lines = []
    #     for datapoint in hf_dataset[split_name]:
    #         lines.append((datapoint["question"], datapoint["query"]))
    #     return lines
    def get_train_test_lines(self, dataset):
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list
        #train_lines = map_hf_dataset_to_list(dataset, "train")
        train_lines = map_hf_dataset_to_list(dataset, "train_merge.json")
        #test_lines = map_hf_dataset_to_list(dataset, "validation")
        test_lines = map_hf_dataset_to_list(dataset, "dev.json")
        return train_lines, test_lines
    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        abs_path = os.path.join(hf_dataset, split_name)
        with open(abs_path, 'r', encoding='utf-8') as f:
            str = f.read()
            data = json.loads(str)
            for datapoint in data:
                lines.append((datapoint["question"], datapoint["query"]))
        return lines

    def load_dataset(self):
        #return datasets.load_dataset("spider")
        return '/nas/wab/MetaICL/MetaICL_fail_data/spider'

def main():
    dataset = Spider()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()