# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
#import datasets
import numpy as np
import json

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class AI2_ARC(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "ai2_arc"
        self.task_type = "text to text"
        self.license = "unknown"

    # def get_choices_and_answer_string(self, datapoint):
    #     answer_index = datapoint["answerKey"]
    #     choices_string = ""
    #     for i in range(len(datapoint["choices"]["label"])):
    #         if datapoint["choices"]["label"][i] == answer_index:
    #             answer_string = datapoint["choices"]["text"][i]
    #         choices_string += " (" + datapoint["choices"]["label"][i] + ") " + datapoint["choices"]["text"][i]
    #     return choices_string, answer_string
    def get_choices_and_answer_string(self, datapoint):
        answer_index = datapoint["answerKey"]
        choices_string = ""
        for i in range(len(datapoint["question"]["choices"])):
            if datapoint["question"]["choices"][i]["label"] == answer_index:
                answer_string = datapoint["question"]["choices"][i]["text"]
            choices_string += " (" + datapoint["question"]["choices"][i]["label"] + ") " + datapoint["question"]["choices"][i]["text"]
        return choices_string, answer_string

    # def map_hf_dataset_to_list(self, hf_dataset, split_name):
    #     lines = []
    #     for datapoint in hf_dataset[split_name]:
    #         choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
    #         lines.append((datapoint["question"] + choices_string, answer_string))
    #     return lines
    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        abs_path = os.path.join(hf_dataset, split_name)
        with open(abs_path, 'r', encoding='utf-8') as f:
            for datapoint in f:
                datapoint = json.loads(datapoint)
                choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
                lines.append((datapoint["question"]['stem'] + choices_string, answer_string))
            return lines

    def load_dataset(self):
        #return datasets.load_dataset("ai2_arc", "ARC-Challenge")
        return '/nas/wab/MetaICL/MetaICL_fail_data/ARC-V1-Feb2018-2/ARC-Challenge'
def main():
    dataset = AI2_ARC()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()