# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datasets
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
    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        abs_path = os.path.join(hf_dataset, split_name)
        with open(abs_path, 'r', encoding='utf-8') as f:
            for datapoint in hf_dataset[split_name]:
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