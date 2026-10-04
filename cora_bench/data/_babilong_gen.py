"""BABILong generator that also records supporting facts and their token spans.

Vendored from booydar/babilong @7a6efee29f5cac03c3c410e6799c80fd2ffe3610
(babilong/babilong_utils.py). Changes: span and supporting-fact tracking, and a pandas NaN fix.
"""

from __future__ import annotations

import re

import nltk
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def compare_answers(target: str, output: str) -> bool:
    """Upstream answer check: gold appears in the output's first sentence."""
    target = target.lower()
    output = output.lower().split(".")[0].split("<context>")[0].split("<example>")[0]
    return target in output


def get_dataset_df(dataset_path: str, max_n_facts: int | None = None) -> pd.DataFrame:
    with open(dataset_path) as f:
        texts = f.read().strip().split("\n")
    df = pd.DataFrame(texts, columns=["text"])
    _null = lambda x: x is None or (isinstance(x, float) and pd.isna(x))  # arrow backend gives NaN, not None
    df["phrase_num"] = df.text.apply(lambda x: int(x.split(" ")[0]))
    df.text = df.text.apply(lambda x: x[x.index(" ") + 1:])
    df["answer"] = df.text.apply(lambda x: x[x.index("\t") + 1:] if "\t" in x else None)
    df["reference_num"] = df.answer.apply(
        lambda x: None if _null(x) else [int(n) for n in re.split("\t| ", x)[1:]])
    df.answer = df.answer.apply(lambda x: None if _null(x) else x.split("\t")[0])
    df.text = df.text.apply(lambda x: x.split("\t")[0] if "\t" in x else x)

    sample_start_inds = list(np.where(df.phrase_num == 1)[0]) + [df.shape[0]]
    for i, (start, end) in enumerate(zip(sample_start_inds, sample_start_inds[1:])):
        df.loc[start:end, "initial_sample_num"] = i
    df.initial_sample_num = df.initial_sample_num.astype(int)

    initial_samples = [df[df.initial_sample_num == sn] for sn in df.initial_sample_num.unique()]
    single_question_slices = []
    for sample in initial_samples:
        answer_positions = sample[~sample.answer.isna()].index
        slices = [sample.loc[:ans_pos].copy() for ans_pos in answer_positions]
        for i, slc in enumerate(slices):
            slices[i] = slc[(slc.answer.isna()) | (slc.index == slc.index[-1])]
        if max_n_facts is not None:
            slices = [slc for slc in slices if slc.shape[0] <= max_n_facts]
        single_question_slices += slices

    df = pd.concat(single_question_slices).reset_index(drop=True)
    sample_start_inds = list(np.where(df.phrase_num == 1)[0]) + [df.shape[0]]
    for i, (start, end) in enumerate(zip(sample_start_inds, sample_start_inds[1:])):
        df.loc[start:end, "sample_num"] = i
    df.sample_num = df.sample_num.astype(int)
    return df


class TaskDataset(Dataset):
    """bAbI task loader; `support_idx` comes from the question's annotated reference numbers."""

    def __init__(self, dataset_path: str, max_n_facts: int | None = None):
        self.fact_dataset = get_dataset_df(dataset_path, max_n_facts=max_n_facts)

    def __getitem__(self, ind: int) -> dict:
        slc = self.fact_dataset[self.fact_dataset.sample_num == ind]
        ref_nums = set(slc.reference_num.values[-1])          # supporting phrase numbers
        phrase_nums = list(slc.phrase_num.values[:-1])        # exclude the question row
        support_idx = [i for i, pn in enumerate(phrase_nums) if pn in ref_nums]
        return {"facts": slc.text.values[:-1], "question": slc.text.values[-1],
                "answer": slc.answer.values[-1], "support_idx": support_idx,
                "phrase_nums": [int(p) for p in phrase_nums],   # original bAbI phrase numbers
                "reference_nums": sorted(int(n) for n in ref_nums)}

    def __len__(self) -> int:
        return self.fact_dataset.sample_num.max()


class SentenceSampler:
    """PG19 background sampler, unchanged from upstream."""

    def __init__(self, dataset, tokenizer, min_sentence_len=10, max_sentence_len=None,
                 shuffle=False, random_seed=42):
        self.sample_ind = 0
        self.dataset = dataset
        self.sentences = []
        self.tokenizer = tokenizer
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.sentence_tokenizer = nltk.PunktSentenceTokenizer()
        self.shuffle = shuffle
        self.gen = np.random.default_rng(seed=random_seed)

    def get_sample(self, sample_size):
        sample, total_len = [], 0
        while True:
            for i, sent in enumerate(list(self.sentences)):
                tokenized = self.tokenizer.encode(" " + sent, add_special_tokens=False)
                if not self.length_is_ok(tokenized):
                    continue
                total_len += len(tokenized)
                sample.append(tokenized)
                if total_len >= sample_size:
                    self.sentences = self.sentences[i + 1:]
                    cutoff = total_len - sample_size
                    if cutoff > 0:
                        sample[-1] = sample[-1][:-cutoff]
                    return sample
            self.sentences = []
            self.sample_sentences_(sample_size)

    def sample_sentences_(self, sample_size):
        sentences = []
        while len(sentences) == 0:
            text = self.next_sample_()
            if self.shuffle:
                if len(text) == 0:
                    continue
                text = text[self.gen.choice(len(text)):]
                text = text[:sample_size * 10]
            sentences += self.sentence_tokenizer.tokenize(text)
            if self.shuffle:
                sentences = sentences[1:-1]
        self.sentences += sentences

    def next_sample_(self):
        if self.shuffle:
            self.total_tokens = 0
            sample_ind = self.gen.choice(len(self.dataset))
            sample = self.dataset[int(sample_ind)]["text"]
        else:
            sample = self.dataset[int(self.sample_ind)]["text"]
            self.sample_ind = (self.sample_ind + 1) % len(self.dataset)
        return sample

    def length_is_ok(self, tokenized):
        if self.max_sentence_len is not None and len(tokenized) > self.max_sentence_len:
            return False
        if self.min_sentence_len is not None and len(tokenized) < self.min_sentence_len:
            return False
        return True


def sum_lengths(sentences):
    return sum(len(s) for s in sentences)


class InstrumentedNoiseInjectionDataset(Dataset):
    """Upstream NoiseInjectionDataset that also returns `fact_token_spans` aligned with `facts`."""

    def __init__(self, task_dataset, noise_sampler, tokenizer, sample_size=1024, random_seed=42):
        self.task_dataset = task_dataset
        self.noise_sampler = noise_sampler
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.gen = np.random.default_rng(seed=random_seed)

    def __getitem__(self, ind: int) -> dict:
        sample = self.task_dataset[ind]
        facts = list(sample["facts"])
        facts_tok = self.tokenizer([" " + f for f in facts], add_special_tokens=False)["input_ids"]
        question_tok = self.tokenizer(sample["question"], add_special_tokens=False)["input_ids"]
        answer_tok = self.tokenizer(sample["answer"], add_special_tokens=False)["input_ids"]

        task_len = sum_lengths(facts_tok)
        background_text = self.noise_sampler.get_sample(self.sample_size - task_len)

        possible_positions = range(len(background_text) + 1)  # facts spread across full context
        fact_positions = self.gen.choice(possible_positions, len(facts_tok))
        fact_positions.sort()

        buckets = [[] for _ in range(len(background_text) + 1)]
        for fi, (fact, pos) in enumerate(zip(facts_tok, fact_positions)):
            buckets[pos].append(("fact", fi, fact))
        for i, s in enumerate(background_text):
            buckets[i].append(("bg", -1, s))

        tokens: list[int] = []
        spans: dict[int, tuple[int, int]] = {}
        for bucket in buckets:
            for kind, fi, toks in bucket:
                start = len(tokens)
                tokens.extend(toks)
                if kind == "fact":
                    spans[fi] = (start, len(tokens))

        sample.update({
            "facts": facts, "fact_token_spans": [spans[i] for i in range(len(facts))],
            "input_tokens": tokens, "question_tokens": question_tok, "target_tokens": answer_tok,
            "fact_positions": fact_positions.tolist(),
        })
        return sample

    def __len__(self) -> int:
        return len(self.task_dataset)
