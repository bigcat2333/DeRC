"""
Script to train BERT on MNLI with our loss function

Modified from the old "run_classifier" script from
https://github.com/huggingface/pytorch-transformer
"""

import argparse
import json
import logging
import os
import random
from collections import namedtuple
from os.path import join, exists
from typing import List, Dict, Iterable

# temporary hack for the pythonroot issue
import sys

import numpy as np
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, \
    Sampler
from tqdm import trange, tqdm

import config
import utils

import clf_distill_loss_functions
from bert_distill import BertDistill
from clf_distill_loss_functions import *

from predictions_analysis import visualize_predictions
from utils import Processor, process_par

# Its a hack, but I didn't want a tensorflow dependency be required to run this code, so
# for now we just copy-paste MNLI loading stuff from debias/datasets/mnli

HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"

NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL_MAP = {k: i for i, k in enumerate(NLI_LABELS)}
REV_NLI_LABEL_MAP = {i: k for i, k in enumerate(NLI_LABELS)}
NLI_LABEL_MAP["hidden"] = NLI_LABEL_MAP["entailment"]

FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
FEVER_LABEL_MAP = {k: i for i, k in enumerate(FEVER_LABELS)}
REV_FEVER_LABEL_MAP = {i: k for i, k in enumerate(FEVER_LABELS)}

fields = ["id", "premise", "hypothesis", "label", "anti_bias"]
TextPairExample = namedtuple("TextPairExample", fields, defaults=(None,) * len(fields))


def load_easy_hard(prefix="", no_mismatched=False):
    all_datasets = []

    all_datasets.append(("mnli_dev_matched_{}easy".format(prefix),
                         load_mnli(False, custom_path="dev_matched_{}easy.tsv".format(prefix))))
    all_datasets.append(("mnli_dev_matched_{}hard".format(prefix),
                         load_mnli(False, custom_path="dev_matched_{}hard.tsv".format(prefix))))
    if not no_mismatched:
        all_datasets.append(("mnli_dev_mismatched_{}easy".format(prefix),
                             load_mnli(False, custom_path="dev_mismatched_{}easy.tsv".format(prefix))))
        all_datasets.append(("mnli_dev_mismatched_{}hard".format(prefix),
                             load_mnli(False, custom_path="dev_mismatched_{}hard.tsv".format(prefix))))

    return all_datasets


def load_hans_subsets():
    src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
    if not exists(src):
        logging.info("Downloading source to %s..." % config.HANS_SOURCE)
        utils.download_to_file(HANS_URL, src)

    hans_datasets = []
    labels = ["entailment", "non-entailment"]
    subsets = set()
    with open(src, "r") as f:
        for line in f.readlines()[1:]:
            line = line.split("\t")
            subsets.add(line[-3])
    subsets = [x for x in subsets]

    for label in labels:
        for subset in subsets:
            name = "hans_{}_{}".format(label, subset)
            examples = load_hans(filter_label=label, filter_subset=subset)
            hans_datasets.append((name, examples))

    return hans_datasets


def load_hans(n_samples=None, filter_label=None, filter_subset=None) -> List[
    TextPairExample]:
    out = []

    if filter_label is not None and filter_subset is not None:
        logging.info("Loading hans subset: {}-{}...".format(filter_label, filter_subset))
    else:
        logging.info("Loading hans all...")

    src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
    if not exists(src):
        logging.info("Downloading source to %s..." % config.HANS_SOURCE)
        utils.download_to_file(HANS_URL, src)

    with open(src, "r") as f:
        f.readline()
        lines = f.readlines()

    if n_samples is not None:
        lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples,
                                                                replace=False)

    for line in lines:
        parts = line.split("\t")
        label = parts[0]

        if filter_label is not None and filter_subset is not None:
            if label != filter_label or parts[-3] != filter_subset:
                continue

        if label == "non-entailment":
            label = 0
        elif label == "entailment":
            label = 1
        else:
            raise RuntimeError()
        s1, s2, pair_id = parts[5:8]
        out.append(TextPairExample(pair_id, s1, s2, label))
    return out


def ensure_mnli_is_downloaded():
    mnli_source = config.GLUE_SOURCE
    if exists(mnli_source) and len(os.listdir(mnli_source)) > 0:
        return
    else:
        raise Exception("Download MNLI from Glue and put files under glue_multinli")

def load_QQP(is_train, print_antibias=False) -> List[TextPairExample]:
    all_data = []
    if is_train:
        if print_antibias:
            filename = '../dataset/QQP/train_anti_bias.tsv'
        else:
            filename = '../dataset/QQP/train.tsv'
    else:
        filename = '../dataset/QQP/dev.tsv'
    with open(filename, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            line = line.strip().split('\t')
            if line[0] == 'id':
                continue
            if print_antibias:
                all_data.append(TextPairExample(line[0], line[3], line[4], int(line[5]), int(line[6])))
            else:
                all_data.append(TextPairExample(line[0], line[3], line[4], int(line[5])))
    return all_data

def load_paws(non_duplicate, sample=None, custom_path='../dataset/QQP') -> List[TextPairExample]:
    all_data = []
    if non_duplicate:
        filename = join(custom_path, 'paws_non_duplicate.tsv')
    else:
        filename = join(custom_path, 'paws_test1.tsv')
    with open(filename, 'r') as f:
        for line in f.readlines():
            # line = line.strip().split('\t')
            # if line[0] == 'id':
            #     continue
            # all_data.append(TextPairExample(line[0], line[3], line[4], int(line[5])))
            line = eval(line)
            all_data.append(TextPairExample(line['sent_id'], line['text_q'], line['text_t'], int(line['sent_label'])))
    return all_data

def load_qqp_bias(bias_or_antibias, sample=None, custom_path='../dataset/QQP') -> List[TextPairExample]:
    all_data = []
    if bias_or_antibias == 'bias':
        filename = join(custom_path, 'remain_sim_en.tsv')
    else:
        filename = join(custom_path, 'union_sim_en.tsv')
    with open(filename, 'r') as f:
        for line in f.readlines():
            # line = line.strip().split('\t')
            # if line[0] == 'id':
            #     continue
            # all_data.append(TextPairExample(line[0], line[3], line[4], int(line[5])))
            line = eval(line)
            all_data.append(TextPairExample(line['sent_id'], line['text_q'], line['text_t'], int(line['sent_label'])))
    return all_data

def load_fever(is_train, custom_path='../dataset/fever') -> List[TextPairExample]:
    all_data = []
    if is_train:
        filename = join(custom_path, 'fever.train.jsonl')
    else:
        filename = join(custom_path, 'fever.dev.jsonl')
    with open(filename, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            # line = line.strip().split('\t')
            # if line[0] == 'id':
            #     continue
            line = eval(line)
            all_data.append(TextPairExample(line['id'], line['claim'], line['evidence'], FEVER_LABEL_MAP[line['gold_label'].rstrip()]))
    return all_data

def load_fever_symmetric(is_train, custom_path='../dataset/fever') -> List[TextPairExample]:
    all_data = []
    # filename = join(custom_path, 'fever_symmetric_generated.jsonl.txt')
    filename = join(custom_path, 'fever_symmetric_test.jsonl.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()
        # random.shuffle(lines)
        for line in lines:
            # line = line.strip().split('\t')
            # if line[0] == 'id':
            #     continue
            line = eval(line)
            # all_data.append(TextPairExample(line['id'], line['claim'], line['evidence_sentence'], FEVER_LABEL_MAP[line['label'].rstrip()]))
            all_data.append(TextPairExample(line['id'], line['claim'], line['evidence'], FEVER_LABEL_MAP[line['label'].rstrip()]))
    return all_data


def load_mnli(is_train, sample=None, custom_path=None, print_antibias=None, bias_or_antibias=None) -> List[TextPairExample]:
    ensure_mnli_is_downloaded()
    if is_train:
        if print_antibias:
            filename = join(config.GLUE_SOURCE, "train_anti_bias_mnli.tsv")
        else:
            filename = join(config.GLUE_SOURCE, "train.tsv")
    else:
        if custom_path is None:
            filename = join(config.GLUE_SOURCE, "dev_matched.tsv")
        elif custom_path != 'dev_anti_bias_mnli.tsv':
            filename = join(config.GLUE_SOURCE, custom_path)
        else:
            filename = join(config.GLUE_SOURCE, "dev_anti_bias_mnli.tsv")


    logging.info("Loading mnli " + ("train" if is_train else "dev"))
    with open(filename) as f:
        f.readline()
        lines = f.readlines()
        if bias_or_antibias is not None and bias_or_antibias == 'bias':
            lines = [line for line in lines if int(line.split('\t')[-1]) == 0]
        elif bias_or_antibias is not None and bias_or_antibias == 'anti_bias':
            lines = [line for line in lines if int(line.split('\t')[-1]) == 1]

    if sample:
        lines = np.random.RandomState(26096781 + sample).choice(lines, sample,
                                                                replace=False)

    out = []
    for line in lines:
        line = line.split("\t")
        if not print_antibias:
            if bias_or_antibias is None:
                out.append(
                    TextPairExample(line[0], line[8], line[9], NLI_LABEL_MAP[line[-1].rstrip()]))
            else:
                out.append(
                    TextPairExample(line[0], line[8], line[9], NLI_LABEL_MAP[line[-2].rstrip()]))
        else:
            if bias_or_antibias is None:
                out.append(
                    TextPairExample(line[0], line[8], line[9], NLI_LABEL_MAP[line[-2].rstrip()], int(line[-1])))
            else:
                out.append(
                    TextPairExample(line[0], line[8], line[9], NLI_LABEL_MAP[line[-3].rstrip()], int(line[-2])))
    return out



def load_all_test_jsonl():
    test_datasets = []
    test_datasets.append(("mnli_test_m", load_jsonl("multinli_0.9_test_matched_unlabeled.jsonl",
                                                    config.MNLI_TEST_SOURCE)))
    test_datasets.append(("mnli_test_mm", load_jsonl("multinli_0.9_test_mismatched_unlabeled.jsonl",
                                                     config.MNLI_TEST_SOURCE)))
    test_datasets.append(("mnli_test_hard_m", load_jsonl("multinli_0.9_test_matched_unlabeled_hard.jsonl",
                                                         config.MNLI_HARD_SOURCE)))
    test_datasets.append(("mnli_test_hard_mm", load_jsonl("multinli_0.9_test_mismatched_unlabeled_hard.jsonl",
                                                          config.MNLI_HARD_SOURCE)))
    return test_datasets


def load_jsonl(file_path, data_dir, sample=None):
    out = []
    full_path = join(data_dir, file_path)
    logging.info("Loading jsonl from {}...".format(full_path))
    with open(full_path, 'r') as jsonl_file:
        for i, line in enumerate(jsonl_file):
            example = json.loads(line)

            label = example["gold_label"]
            if label == '-':
                continue

            if not "pairID" in example.keys():
                id = i
            else:
                id = example["pairID"]
            text_a = example["sentence1"]
            text_b = example["sentence2"]

            out.append(TextPairExample(id, text_a, text_b, NLI_LABEL_MAP[label]))

    if sample:
        random.shuffle(out)
        out = out[:sample]

    return out


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, input_ids, segment_ids, label_id, bias, anti_bias):
        self.example_id = example_id
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.bias = bias
        self.anti_bias = anti_bias


class ExampleConverter(Processor):
    def __init__(self, max_seq_length, tokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def process(self, data: Iterable):
        features = []
        tokenizer = self.tokenizer
        max_seq_length = self.max_seq_length

        for example in data:
            tokens_a = tokenizer.tokenize(example.premise)

            tokens_b = None
            if example.hypothesis:
                tokens_b = tokenizer.tokenize(example.hypothesis)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            features.append(
                InputFeatures(
                    example_id=example.id,
                    input_ids=np.array(input_ids),
                    segment_ids=np.array(segment_ids),
                    label_id=example.label,
                    anti_bias=example.anti_bias,
                    bias=None
                ) if example.anti_bias is not None else\
                InputFeatures(
                    example_id=example.id,
                    input_ids=np.array(input_ids),
                    segment_ids=np.array(segment_ids),
                    label_id=example.label,
                    anti_bias=None,
                    bias=None
                ))
        return features


class InputFeatureDataset(Dataset):

    def __init__(self, examples: List[InputFeatures]):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def collate_input_features(batch: List[InputFeatures]):
    max_seq_len = max(len(x.input_ids) for x in batch)
    sz = len(batch)

    input_ids = np.zeros((sz, max_seq_len), np.int64)
    segment_ids = np.zeros((sz, max_seq_len), np.int64)
    mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)
    for i, ex in enumerate(batch):
        input_ids[i, :len(ex.input_ids)] = ex.input_ids
        segment_ids[i, :len(ex.segment_ids)] = ex.segment_ids
        mask[i, :len(ex.input_ids)] = 1

    input_ids = torch.from_numpy(input_ids)
    segment_ids = torch.from_numpy(segment_ids)
    label_ids = torch.from_numpy(np.array([x.label_id for x in batch], np.int64))

    # include example ids for test submission
    try:
        example_ids = torch.tensor([int(x.example_id) for x in batch])
    except:
        example_ids = torch.zeros(len(batch)).long()

    if batch[0].bias is None:
        if batch[0].anti_bias is not None:
            anti_bias = torch.tensor([x.anti_bias for x in batch])
            return example_ids, input_ids, mask, segment_ids, label_ids, anti_bias
        else:
            return example_ids, input_ids, mask, segment_ids, label_ids

    teacher_probs = torch.tensor([x.teacher_probs for x in batch])
    bias = torch.tensor([x.bias for x in batch])
    return example_ids, input_ids, mask, segment_ids, label_ids, bias, teacher_probs


class SortedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, seed):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.seed = seed
        if batch_size == 1:
            raise NotImplementedError()
        self._epoch = 0

    def __iter__(self):
        rng = np.random.RandomState(self._epoch + 601767 + self.seed)
        n_batches = len(self)
        batch_lens = np.full(n_batches, self.batch_size, np.int32)

        # Randomly select batches to reduce by size 1
        extra = n_batches * self.batch_size - len(self.data_source)
        batch_lens[rng.choice(len(batch_lens), extra, False)] -= 1

        batch_ends = np.cumsum(batch_lens)
        batch_starts = np.pad(batch_ends[:-1], [1, 0], "constant")

        if batch_ends[-1] != len(self.data_source):
            print(batch_ends)
            raise RuntimeError()

        bounds = np.stack([batch_starts, batch_ends], 1)
        rng.shuffle(bounds)

        for s, e in bounds:
            yield np.arange(s, e)

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def build_train_dataloader(data: List[InputFeatures], batch_size, seed, sorted):
    if sorted:
        data.sort(key=lambda x: len(x.input_ids))
        ds = InputFeatureDataset(data)
        sampler = SortedBatchSampler(ds, batch_size, seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_input_features)
    else:
        ds = InputFeatureDataset(data)
        return DataLoader(ds, batch_size=batch_size, sampler=RandomSampler(ds),
                          collate_fn=collate_input_features)


def build_eval_dataloader(data: List[InputFeatures], batch_size):
    ds = InputFeatureDataset(data)
    return DataLoader(ds, batch_size=batch_size, sampler=SequentialSampler(ds),
                      collate_fn=collate_input_features)


def convert_examples_to_features(
        examples: List[TextPairExample], max_seq_length, tokenizer, n_process=1):
    converter = ExampleConverter(max_seq_length, tokenizer)
    # print(examples[0])
    return process_par(examples, converter, n_process, chunk_size=2000, desc="featurize")


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def f1(preds, labels, non_duplicate=False):
    true, find, right = 0, 0, 0
    right_label = 0 if non_duplicate else 1
    for pred, label in zip(preds, labels):
        if pred == right_label and label == right_label:
            right += 1
            true += 1
            find += 1
        elif pred == right_label and label != right_label:
            true += 1
        elif pred != right_label and label == right_label:
            find += 1
    precision = right / find if find else 0
    recall = right / true if true else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return f1


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_on_train",
                        action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test and create submission.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="Seed for randomized elements in the training")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--print_antibias",
                        action='store_true',
                        help="Whether print losses of anti-bias samples")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    ## Our arguements
    parser.add_argument("--mode", choices=["none"])
    parser.add_argument("--penalty", type=float, default=0.03,
                        help="Penalty weight for the learn_mixin model")
    parser.add_argument("--focal_loss_gamma", type=float, default=1.0)
    parser.add_argument("--n_processes", type=int, default=4,
                        help="Processes to use for pre-processing")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_num", type=int, default=2000)
    parser.add_argument("--sorted", action="store_true",
                        help='Sort the data so most batches have the same input length,'
                             ' makes things about 2x faster. Our experiments did not actually'
                             ' use this in the end (not sure if it makes a difference) so '
                             'its off by default.')
    parser.add_argument("--dataset", choices=["mnli", "QQP", "fever"], required=True)
    parser.add_argument("--theta", type=float, default=0.1, help="for theta smoothed distillation loss")
    parser.add_argument("--add_bias_on_eval", action="store_true")

    args = parser.parse_args()

    utils.add_stdout_logger()
    
    lower_loss_fn_type=None
    if args.mode == "none":
        loss_fn = clf_distill_loss_functions.Plain()
    elif args.mode == "distill":
        loss_fn = clf_distill_loss_functions.DistillLoss()
    elif args.mode == "smoothed_distill":
        loss_fn = clf_distill_loss_functions.SmoothedDistillLoss()
    elif args.mode == "smoothed_distill_annealed":
        loss_fn = clf_distill_loss_functions.SmoothedDistillLossAnnealed()
    elif args.mode == "theta_smoothed_distill":
        loss_fn = clf_distill_loss_functions.ThetaSmoothedDistillLoss(args.theta)
    elif args.mode == "label_smoothing":
        loss_fn = clf_distill_loss_functions.LabelSmoothing(3)
    elif args.mode == "reweight_baseline":
        loss_fn = clf_distill_loss_functions.ReweightBaseline()
    elif args.mode == "permute_smoothed_distill":
        loss_fn = clf_distill_loss_functions.PermuteSmoothedDistillLoss()
    elif args.mode == "smoothed_reweight_baseline":
        loss_fn = clf_distill_loss_functions.SmoothedReweightLoss()
    elif args.mode == "bias_product_baseline":
        loss_fn = clf_distill_loss_functions.BiasProductBaseline()
    elif args.mode == "learned_mixin_baseline":
        loss_fn = clf_distill_loss_functions.LearnedMixinBaseline(args.penalty)
    elif args.mode == "reweight_by_teacher":
        loss_fn = clf_distill_loss_functions.ReweightByTeacher()
    elif args.mode == "reweight_by_teacher_annealed":
        loss_fn = clf_distill_loss_functions.ReweightByTeacherAnnealed()
    elif args.mode == "ours_and_reweight_by_teacher_annealed":
        loss_fn = clf_distill_loss_functions.ReweightByTeacherAnnealed()
        lower_loss_fn_type = 'plain'
    elif args.mode == "bias_product_by_teacher":
        loss_fn = clf_distill_loss_functions.BiasProductByTeacher()
    elif args.mode == "bias_product_by_teacher_annealed":
        loss_fn = clf_distill_loss_functions.BiasProductByTeacherAnnealed()
    elif args.mode == "focal_loss":
        loss_fn = clf_distill_loss_functions.FocalLoss(gamma=args.focal_loss_gamma)
    else:
        raise RuntimeError()

    output_dir = args.output_dir

    if args.do_train:
        if exists(output_dir):
            if len(os.listdir(output_dir)) > 0:
                logging.warning("Output dir exists and is non-empty")
        else:
            os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logging.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(output_dir) and os.listdir(output_dir) and args.do_train:
        logging.warning(
            "Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Its way ot easy to forget if this is being set by a command line flag
    if "-uncased" in args.bert_model:
        do_lower_case = True
    elif "-cased" in args.bert_model:
        do_lower_case = False
    else:
        raise NotImplementedError(args.bert_model)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=do_lower_case)

    num_train_optimization_steps = None
    train_examples = None
    if args.do_train:
        if args.dataset == 'mnli':
            train_examples = load_mnli(True, args.debug_num if args.debug else None, print_antibias=args.print_antibias)
        elif args.dataset == 'QQP':
            train_examples = load_QQP(True, args.print_antibias)
        elif args.dataset == 'fever':
            train_examples = load_fever(True)
        num_train_optimization_steps = int(
            len(
                train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        loss_fn.num_train_optimization_steps = int(num_train_optimization_steps)
        loss_fn.num_epochs = int(args.num_train_epochs)

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(
        str(PYTORCH_PRETRAINED_BERT_CACHE),
        'distributed_{}'.format(args.local_rank))

    model = BertDistill.from_pretrained(
        args.bert_model, cache_dir=cache_dir, num_labels=3, loss_fn=loss_fn)
    if lower_loss_fn_type and lower_loss_fn_type == 'plain':
        model.lower_loss_fn_type = lower_loss_fn_type
        model.lower_loss_fn = clf_distill_loss_functions.Plain()
    elif lower_loss_fn_type and lower_loss_fn_type != 'plain':
        print('lower_loss_fn_type not supported')
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    ignore = ['classifier.weight', 'classifier.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay + ignore)],
    #      'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0}
    # ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        train_features: List[InputFeatures] = convert_examples_to_features(
            train_examples, args.max_seq_length, tokenizer, args.n_processes)

        bias_map = None

        example_map = {}
        for ex in train_examples:
            example_map[ex.id] = ex

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)

        train_dataloader = build_train_dataloader(train_features, args.train_batch_size,
                                                  args.seed, args.sorted)

        model.train()
        loss_ema = 0
        total_steps = 0
        decay = 0.99

        for _ in trange(int(args.num_train_epochs), desc="Epoch", ncols=100):
            tr_loss = 0
            tr_loss_lower_anti_bias_loss = 0
            tr_loss_lower_bias_loss = 0
            tr_loss_upper_anti_bias_loss = 0
            tr_loss_upper_bias_loss = 0
            tr_num_anti_bias = 0
            tr_num_bias = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            pbar = tqdm(train_dataloader, desc="loss", ncols=100)
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(device) for t in batch)

                bias = None
                teacher_probs = None
                if not args.print_antibias:
                    example_ids, input_ids, input_mask, segment_ids, label_ids = batch
                    anti_bias_label = None
                else:
                    example_ids, input_ids, input_mask, segment_ids, label_ids, anti_bias_label = batch

                if args.print_antibias:
                    logits, loss, loss_lower_anti_bias_loss, loss_lower_bias_loss, loss_upper_anti_bias_loss, loss_upper_bias_loss = model(input_ids, segment_ids, input_mask, label_ids, bias,
                                     teacher_probs, global_step, num_train_optimization_steps, anti_bias_label)
                    num_anti_bias = (anti_bias_label == 1).int().sum().item()
                    num_bias = input_ids.size(0) - num_anti_bias
                    tr_loss_lower_anti_bias_loss += loss_lower_anti_bias_loss * num_anti_bias
                    tr_loss_lower_bias_loss += loss_lower_bias_loss * num_bias
                    tr_loss_upper_anti_bias_loss += loss_upper_anti_bias_loss * num_anti_bias
                    tr_loss_upper_bias_loss += loss_upper_bias_loss * num_bias
                    tr_num_anti_bias += num_anti_bias
                    tr_num_bias += num_bias
                else:
                    logits, loss = model(input_ids, segment_ids, input_mask, label_ids, bias,
                                        teacher_probs, global_step, num_train_optimization_steps, anti_bias_label)

                total_steps += 1
                loss_ema = loss_ema * decay + loss.cpu().detach().numpy() * (1 - decay)
                descript = "loss=%.4f" % (loss_ema / (1 - decay ** total_steps))
                pbar.set_description(descript, refresh=False)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(
                            global_step / num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if args.print_antibias:
                    if (global_step + 1) % 20 == 0:
                        logging.info("step: %d, loss: %.4f, loss_lower_anti_bias_loss: %.4f, loss_lower_bias_loss: %.4f, loss_upper_anti_bias_loss: %.4f, loss_upper_bias_loss: %.4f, total_num_anti_bias: %d, total_num_bias: %d",
                                     step, tr_loss/20, tr_loss_lower_anti_bias_loss/tr_num_anti_bias, tr_loss_lower_bias_loss/tr_num_bias, tr_loss_upper_anti_bias_loss/tr_num_anti_bias, tr_loss_upper_bias_loss/tr_num_bias, tr_num_anti_bias, tr_num_bias)
                        tr_loss = 0
                        tr_loss_lower_anti_bias_loss = 0
                        tr_loss_lower_bias_loss = 0
                        tr_loss_upper_anti_bias_loss = 0
                        tr_loss_upper_bias_loss = 0
                        tr_num_anti_bias = 0
                        tr_num_bias = 0
                        

                if (global_step + 1) % 500 == 0 and global_step > 10000:
                    logging.info('saving model_' + str(global_step))
                    model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
                    checkpoint_output_dir = '/path/to/save/checkpoints/DeRC_layer3_QQP'
                    model_dir_name = 'model_' + str(global_step)
                    model_dir = os.path.join(checkpoint_output_dir, model_dir_name)
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_config_file = os.path.join(model_dir, CONFIG_NAME)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())
                    

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Record the args as well
        arg_dict = {}
        for arg in vars(args):
            arg_dict[arg] = getattr(args, arg)
        with open(join(output_dir, "args.json"), 'w') as out_fh:
            json.dump(arg_dict, out_fh)

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        if lower_loss_fn_type:
            model = BertDistill(config, num_labels=3, loss_fn=loss_fn, lower_loss_fn_type=lower_loss_fn_type)
        else:
            model = BertDistill(config, num_labels=3, loss_fn=loss_fn)
        model.load_state_dict(torch.load(output_model_file))
    else:
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(output_config_file)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        model = BertDistill(config, num_labels=3, loss_fn=loss_fn)
        model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if not args.do_eval and not args.do_test:
        return
    if not (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        return

    model.eval()

    if args.do_eval:
        if args.dataset == 'mnli':
            eval_datasets = [("mnli_dev_m", load_mnli(False)),
                            ("mnli_dev_mm", load_mnli(False, custom_path="dev_mismatched.tsv"))]
            # eval_datasets = [("mnli_dev_m", load_mnli(False))]
            # eval_datasets += load_easy_hard(prefix="overlap_", no_mismatched=True)
            # eval_datasets += load_easy_hard()
            eval_datasets += [("hans", load_hans())]
            eval_datasets += [("dev_bias", load_mnli(False, custom_path="dev_anti_bias_mnli.tsv", bias_or_antibias='bias'))]
            eval_datasets += [("dev_anti_bias", load_mnli(False, custom_path="dev_anti_bias_mnli.tsv", bias_or_antibias='anti_bias'))]

            # eval_datasets += load_hans_subsets()
        elif args.dataset == 'QQP':
            eval_datasets = [("qqp_dev", load_QQP(False))]
            eval_datasets += [("paws", load_paws(False))]
            # eval_datasets = [("paws", load_paws(False))]
            eval_datasets += [("paws_non_duplicate", load_paws(True))]
            # eval_datasets = [("qqp_bias", load_qqp_bias('bias'))]
            # eval_datasets += [("qqp_anti_bias", load_qqp_bias('anti_bias'))]
        elif args.dataset == 'fever':
            eval_datasets = [("fever_dev", load_fever(False))]
            eval_datasets += [("fever_symmetric", load_fever_symmetric(True))]

        # stress test
        # eval_datasets += [("negation_m", load_jsonl("multinli_0.9_negation_matched.jsonl",
        #                                             "../dataset/StressTests/Negation"))]
        # eval_datasets += [("negation_mm", load_jsonl("multinli_0.9_negation_mismatched.jsonl",
        #                                              "../dataset/StressTests/Negation"))]
        # eval_datasets += [("overlap_m", load_jsonl("multinli_0.9_taut2_matched.jsonl",
        #                                             "../dataset/StressTests/Word_Overlap"))]
        # eval_datasets += [("overlap_mm", load_jsonl("multinli_0.9_taut2_mismatched.jsonl",
        #                                              "../dataset/StressTests/Word_Overlap"))]
        # eval_datasets += [("length_m", load_jsonl("multinli_0.9_length_mismatch_matched.jsonl",
        #                                             "../dataset/StressTests/Length_Mismatch"))]
        # eval_datasets += [("length_mm", load_jsonl("multinli_0.9_length_mismatch_mismatched.jsonl",
        #                                              "../dataset/StressTests/Length_Mismatch"))]

        # eval_datasets = [("rte", load_jsonl("eval_rte.jsonl",
        #                                     "../dataset/mnli_eval_suite"))]
        # eval_datasets += [("rte_glue", load_jsonl("eval_glue_rte.jsonl",
        #                                          "../dataset/mnli_eval_suite"))]
        # eval_datasets += [("sick", load_jsonl("eval_sick.jsonl",
        #                                       "../dataset/mnli_eval_suite"))]
        # eval_datasets += [("diagnostic", load_jsonl("diagnostic-full.jsonl",
        #                                             "../dataset/mnli_eval_suite"))]
        # eval_datasets += [("scitail", load_jsonl("scitail_1.0_test.txt",
        #                                           "../dataset/scitail/snli_format"))]

        # todo delete
        if args.do_eval_on_train:
            if args.dataset == 'mnli':
                eval_datasets = [("mnli_train", load_mnli(True))]
            elif args.dataset == 'QQP':
                eval_datasets = [("qqp_train", load_qqp(True))]
    else:
        eval_datasets = []

    if args.do_test:
        test_datasets = load_all_test_jsonl()
        eval_datasets += test_datasets
        subm_paths = ["../submission/{}.csv".format(x[0]) for x in test_datasets]


    # high layer
    for ix, (name, eval_examples) in enumerate(eval_datasets):
        logging.info("***** Running evaluation on %s high layer *****" % name)
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", args.eval_batch_size)
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, tokenizer)
        eval_features.sort(key=lambda x: len(x.input_ids))
        all_label_ids = np.array([x.label_id for x in eval_features])
        eval_dataloader = build_eval_dataloader(eval_features, args.eval_batch_size)

        eval_loss = 0
        nb_eval_steps = 0
        probs = []
        test_subm_ids = []

        for example_ids, input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                               desc="Evaluating",
                                                                               ncols=100):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits, _ = model(input_ids, segment_ids, input_mask)

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            probs.append(torch.nn.functional.softmax(logits, 1).detach().cpu().numpy())
            test_subm_ids.append(example_ids.cpu().numpy())

        probs = np.concatenate(probs, 0)
        test_subm_ids = np.concatenate(test_subm_ids, 0)
        eval_loss = eval_loss / nb_eval_steps

        if "hans" in name:
            # take max of non-entailment rather than taking their sum
            probs[:, 0] = probs[:, [0, 2]].max(axis=1)
            # probs[:, 0] = probs[:, 0] + probs[:, 2]
            probs = probs[:, :2]

        preds = np.argmax(probs, axis=1)

        result = {"acc": simple_accuracy(preds, all_label_ids)}
        if 'non' in name:
            result['f1'] = f1(preds, all_label_ids, True)
        else:
            result['f1'] = f1(preds, all_label_ids)
        result["loss"] = eval_loss

        conf_plot_file = os.path.join(output_dir, "eval_%s_confidence.png" % name)
        ECE, bins_acc, bins_conf, bins_num = visualize_predictions(probs, all_label_ids, conf_plot_file=conf_plot_file)
        result["ECE"] = ECE
        result["bins_acc"] = bins_acc
        result["bins_conf"] = bins_conf
        result["bins_num"] = bins_num

        output_eval_file = os.path.join(output_dir, "eval_%s_results.txt" % name)
        output_all_eval_file = os.path.join(output_dir, "eval_all_results.txt")
        with open(output_eval_file, "w") as writer, open(output_all_eval_file, "a") as all_writer:
            logging.info("***** Eval results *****")
            all_writer.write("eval results on %s:\n" % name)
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
                all_writer.write("%s = %s\n" % (key, str(result[key])))

        output_answer_file = os.path.join(output_dir, "eval_%s_answers.json" % name)
        answers = {ex.example_id: [float(x) for x in p] for ex, p in
                   zip(eval_features, probs)}
        with open(output_answer_file, "w") as f:
            json.dump(answers, f)

        # prepare submission file
        if args.do_test and ix >= len(eval_datasets) - len(test_datasets):
            with open(subm_paths.pop(0), "w") as subm_f:
                subm_f.write("pairID,gold_label\n")
                for sub_id, pred_label_id in zip(test_subm_ids, preds):
                    if args.dataset == 'mnli':
                        subm_f.write("{},{}\n".format(str(sub_id), REV_NLI_LABEL_MAP[pred_label_id]))
                    elif args.dataset == 'QQP':
                        subm_f.write("{},{}\n".format(str(sub_id), pred_label_id))
                    elif args.dataset == 'fever':
                        subm_f.write("{},{}\n".format(str(sub_id), REV_FEVER_LABEL_MAP[pred_label_id]))
    
    # low layer
    for ix, (name, eval_examples) in enumerate(eval_datasets):
        logging.info("***** Running evaluation on %s low layer *****" % name)
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", args.eval_batch_size)
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length, tokenizer)
        eval_features.sort(key=lambda x: len(x.input_ids))
        all_label_ids = np.array([x.label_id for x in eval_features])
        eval_dataloader = build_eval_dataloader(eval_features, args.eval_batch_size)

        eval_loss = 0
        nb_eval_steps = 0
        probs = []
        test_subm_ids = []

        for example_ids, input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                                               desc="Evaluating",
                                                                               ncols=100):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                _, logits = model(input_ids, segment_ids, input_mask)

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            probs.append(torch.nn.functional.softmax(logits, 1).detach().cpu().numpy())
            test_subm_ids.append(example_ids.cpu().numpy())

        probs = np.concatenate(probs, 0)
        test_subm_ids = np.concatenate(test_subm_ids, 0)
        eval_loss = eval_loss / nb_eval_steps

        if "hans" in name:
            # take max of non-entailment rather than taking their sum
            probs[:, 0] = probs[:, [0, 2]].max(axis=1)
            # probs[:, 0] = probs[:, 0] + probs[:, 2]
            probs = probs[:, :2]

        preds = np.argmax(probs, axis=1)

        result = {"acc": simple_accuracy(preds, all_label_ids)}
        if 'non' in name:
            result['f1'] = f1(preds, all_label_ids, True)
        else:
            result['f1'] = f1(preds, all_label_ids)
        result["loss"] = eval_loss

        conf_plot_file = os.path.join(output_dir, "eval_%s_confidence.png" % name)
        ECE, bins_acc, bins_conf, bins_num = visualize_predictions(probs, all_label_ids, conf_plot_file=conf_plot_file)
        result["ECE"] = ECE
        result["bins_acc"] = bins_acc
        result["bins_conf"] = bins_conf
        result["bins_num"] = bins_num

        output_eval_file = os.path.join(output_dir, "eval_%s_results.txt" % name)
        output_all_eval_file = os.path.join(output_dir, "eval_all_results.txt")
        with open(output_eval_file, "w") as writer, open(output_all_eval_file, "a") as all_writer:
            logging.info("***** Eval results *****")
            all_writer.write("eval results on %s:\n" % name)
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
                all_writer.write("%s = %s\n" % (key, str(result[key])))

        output_answer_file = os.path.join(output_dir, "eval_%s_answers.json" % name)
        answers = {ex.example_id: [float(x) for x in p] for ex, p in
                   zip(eval_features, probs)}
        with open(output_answer_file, "w") as f:
            json.dump(answers, f)

        # prepare submission file
        if args.do_test and ix >= len(eval_datasets) - len(test_datasets):
            with open(subm_paths.pop(0), "w") as subm_f:
                subm_f.write("pairID,gold_label\n")
                for sub_id, pred_label_id in zip(test_subm_ids, preds):
                    if args.dataset == 'mnli':
                        subm_f.write("{},{}\n".format(str(sub_id), REV_NLI_LABEL_MAP[pred_label_id]))
                    elif args.dataset == 'QQP':
                        subm_f.write("{},{}\n".format(str(sub_id), pred_label_id))
                    elif args.dataset == 'fever':
                        subm_f.write("{},{}\n".format(str(sub_id), REV_FEVER_LABEL_MAP[pred_label_id]))


if __name__ == "__main__":
    main()


# eval_datasets = [("snli_test", load_jsonl(file_path="snli_1.0_test.jsonl", data_dir="../dataset/snli")),
#                  ("snli_hard_test", load_jsonl(file_path="snli_1.0_test_hard.jsonl", data_dir="../dataset/snli")),
#                  ("snli_lovp_e", load_jsonl(file_path="snli_overlap_entailment.jsonl", data_dir="../dataset/snli")),
#                  ("snli_lovp_ne", load_jsonl(file_path="snli_overlap_non_entailment.jsonl", data_dir="../dataset/snli"))]
