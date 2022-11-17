# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset

class FinancialPhrasebank(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "financial_phrasebank"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0:"negative",
            1:"neutral",
            2:"positive",
        }

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test
        map_hf_dataset_to_list = self.get_map_hf_dataset_to_list()
        if map_hf_dataset_to_list is None:
            map_hf_dataset_to_list = self.map_hf_dataset_to_list
        lines = map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)

        n = len(lines)

        train_lines = lines[:int(0.8*n)]
        test_lines = lines[int(0.8*n):]

        return train_lines, test_lines

    # def map_hf_dataset_to_list(self, hf_dataset, split_name):
    #     lines = []
    #     for datapoint in hf_dataset[split_name]:
    #         # line[0]: input; line[1]: output
    #         lines.append((datapoint["sentence"], self.label[datapoint["label"]]))
    #     return lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        abs_path = os.path.join(dhf_dataset, 'allAgree_utf8.txt')
        with open(abs_path, 'r', encoding='utf-8') as f:
            for line in f:
                label = line.split('@')[-1]
                sent = line[:-len(label)]
                lines.append((sent, label))
        return lines

    def load_dataset(self):
        return '/Users/lishengping/codes/others/MetaICL/data/fair_download_data/FinancialPhraseBank-v1.0'
        # return datasets.load_dataset('financial_phrasebank', 'sentences_allagree')

def main():
    dataset = FinancialPhrasebank()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed, path="../data/")

if __name__ == "__main__":
    main()