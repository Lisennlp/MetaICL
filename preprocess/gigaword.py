# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
#import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class Gigaword(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "gigaword"
        self.task_type = "text to text"
        self.license = "unknown"

    # def map_hf_dataset_to_list(self, hf_dataset, split_name):
    #     lines = []
    #     for datapoint in hf_dataset[split_name]:
    #         input_text = datapoint["document"]
    #         output_text = datapoint["summary"]
    #         lines.append(("summarize: " + input_text, output_text))
    #     return lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        if split_name == 'train':
            # tgt_path = '/Users/lishengping/data/MetaICL_fail_data/ggw_data/train.tgt'
            # src_path = '/Users/lishengping/data/MetaICL_fail_data/ggw_data/train.src'
            tgt_path = '/nas/wab/MetaICL/MetaICL_fail_data/ggw_data/train.tgt'
            src_path = '/nas/wab/MetaICL/MetaICL_fail_data/ggw_data/train.src'
        else:
            # src_path = '/Users/lishengping/data/MetaICL_fail_data/ggw_data/test.src'
            # tgt_path = '/Users/lishengping/data/MetaICL_fail_data/ggw_data/test.tgt'
            src_path = '/nas/wab/MetaICL/MetaICL_fail_data/ggw_data/test.src'
            tgt_path = '/nas/wab/MetaICL/MetaICL_fail_data/ggw_data/test.tgt'
        with open(src_path, 'r') as src_f, open(tgt_path, 'r') as tgt_f:
            src_lines = src_f.readlines()
            tgt_lines = tgt_f.readlines()
            assert len(src_lines) == len(tgt_lines)
            for (src_line, tgt_line) in zip(src_lines, tgt_lines):
                lines.append(("summarize: " + src_line.strip(), tgt_line.strip()))
        return lines

    def load_dataset(self):
        return ''
        # return datasets.load_dataset('gigaword')

def main():
    dataset = Gigaword()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()
