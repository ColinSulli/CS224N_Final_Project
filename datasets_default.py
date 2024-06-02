#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions that load the three datasets
for training and evaluating multitask BERT.

Feel free to edit code in this file if you wish to modify the way in which the data
examples are preprocessed.
'''

import csv

import torch
from torch.utils.data import Dataset
from pals import tokenization
from pals.pal import _truncate_seq_pair
from tokenizer import BertTokenizer


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args, imdb_dataset=False):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.imdb_dataset = imdb_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        if(self.imdb_dataset):
            sent_ids = sent_ids * 4

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        token_type_ids = torch.LongTensor(encoding['token_type_ids'])
        labels = torch.LongTensor(labels)

        return token_ids, token_type_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, token_type_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


# Unlike SentenceClassificationDataset, we do not load labels in SentenceClassificationTestDataset.
class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        token_type_ids = torch.LongTensor(encoding['token_type_ids'])

        return token_ids, token_type_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, token_type_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SNLIDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False, max_sequence_length=128):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file='uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        batch_size = len(data)
        token_ids = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        token_ids_1 = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        token_ids_2 = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        token_type_ids = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        attention_mask_1 = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        attention_mask_2 = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        labels = torch.zeros(batch_size, 1, dtype=torch.float32 if self.isRegression else torch.long)

        sent_ids = []

        for index, training_case in enumerate(data):

            sent1 = training_case['premise']
            sent2 = training_case['hypothesis']
            label = training_case['label']
            #label = label * 5
            if label == 0:
                label = 1
            elif label == 1:
                continue
            else:
                label = 0

            tokens_1 = self.tokenizer.tokenize(sent1)
            tokens_2 = self.tokenizer.tokenize(sent2)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_1, tokens_2, self.max_sequence_length - 3)

            tokens = ["[CLS]"] + tokens_1 + ["[SEP]"] + tokens_2 + ["[SEP]"]
            segment_ids = [0] * (len(tokens_1) + 2) + [1] * (len(tokens_2) + 1)
            input_mask = [1] * len(tokens)

            ### Colin's ###
            input_mask_1 = [1] * len(tokens_1)
            input_mask_2 = [1] * len(tokens_2)
            ### Colin's ###

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            ### Colin's ###
            input_ids_1 = self.tokenizer.convert_tokens_to_ids(tokens_1)
            input_ids_2 = self.tokenizer.convert_tokens_to_ids(tokens_2)
            ### Colin's ###

            padding_length = self.max_sequence_length - len(tokens)
            input_ids.extend([0] * padding_length)
            segment_ids.extend([0] * padding_length)
            input_mask.extend([0] * padding_length)

            ### Colin's ###
            padding_length_1 = self.max_sequence_length - len(tokens_1)
            padding_length_2 = self.max_sequence_length - len(tokens_2)
            input_ids_1.extend([0] * padding_length_1)
            input_ids_2.extend([0] * padding_length_2)
            input_mask_1.extend([0] * padding_length_1)
            input_mask_2.extend([0] * padding_length_2)
            ### Colin's ###

            assert len(input_ids) == self.max_sequence_length
            assert len(input_mask) == self.max_sequence_length
            assert len(segment_ids) == self.max_sequence_length

            token_ids[index] = torch.Tensor(input_ids)
            token_type_ids[index] = torch.Tensor(segment_ids)
            attention_mask[index] = torch.Tensor(input_mask)
            labels[index] = torch.Tensor([label])

            ### Colin's ###
            token_ids_1[index] = torch.Tensor(input_ids_1)
            token_ids_2[index] = torch.Tensor(input_ids_2)
            attention_mask_1[index] = torch.Tensor(input_mask_1)
            attention_mask_2[index] = torch.Tensor(input_mask_2)
            ### Colin's ###
        return (token_ids, token_ids_1, token_ids_2, token_type_ids,
                attention_mask, attention_mask_1, attention_mask_2, labels.squeeze(1))

    def collate_fn(self, all_data):
        (token_ids, token_ids_1, token_ids_2, token_type_ids,
         attention_mask, attention_mask_1, attention_mask_2, labels) = self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'token_ids_1': token_ids_1,
                'token_ids_2': token_ids_2,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'attention_mask_1': attention_mask_1,
                'attention_mask_2': attention_mask_2,
                'labels': labels,
            }

        return batched_data

class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False, max_sequence_length=128):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression 
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenization.FullTokenizer(
                vocab_file='uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        batch_size = len(data)
        token_ids = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        token_ids_1 = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        token_ids_2 = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        token_type_ids = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        attention_mask_1 = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        attention_mask_2 = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        labels = torch.zeros(batch_size, 1, dtype=torch.float32 if self.isRegression else torch.long)
        
        sent_ids = []
        
        for index, training_case in enumerate(data):
            sent1 = training_case[0]
            sent2 = training_case[1]
            label = training_case[2]
            sent_id = training_case[3]
        
            tokens_1 = self.tokenizer.tokenize(sent1)
            tokens_2 = self.tokenizer.tokenize(sent2)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_1, tokens_2, self.max_sequence_length - 3)

            tokens = ["[CLS]"] + tokens_1 + ["[SEP]"] + tokens_2 + ["[SEP]"]
            segment_ids = [0] * (len(tokens_1) + 2) + [1] * (len(tokens_2) + 1)
            input_mask = [1] * len(tokens)

            ### Colin's ###
            input_mask_1 = [1] * len(tokens_1)
            input_mask_2 = [1] * len(tokens_2)
            ### Colin's ###

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            ### Colin's ###
            input_ids_1 = self.tokenizer.convert_tokens_to_ids(tokens_1)
            input_ids_2 = self.tokenizer.convert_tokens_to_ids(tokens_2)
            ### Colin's ###
            
            padding_length = self.max_sequence_length - len(tokens)
            input_ids.extend([0] * padding_length)
            segment_ids.extend([0] * padding_length)
            input_mask.extend([0] * padding_length)

            ### Colin's ###
            padding_length_1 = self.max_sequence_length - len(tokens_1)
            padding_length_2 = self.max_sequence_length - len(tokens_2)
            input_ids_1.extend([0] * padding_length_1)
            input_ids_2.extend([0] * padding_length_2)
            input_mask_1.extend([0] * padding_length_1)
            input_mask_2.extend([0] * padding_length_2)
            ### Colin's ###

            assert len(input_ids) == self.max_sequence_length
            assert len(input_mask) == self.max_sequence_length
            assert len(segment_ids) == self.max_sequence_length

            token_ids[index] = torch.Tensor(input_ids)
            token_type_ids[index] = torch.Tensor(segment_ids)
            attention_mask[index] = torch.Tensor(input_mask)
            labels[index] = torch.Tensor([label])
            sent_ids.append(sent_id)

            ### Colin's ###
            token_ids_1[index] = torch.Tensor(input_ids_1)
            token_ids_2[index] = torch.Tensor(input_ids_2)
            attention_mask_1[index] = torch.Tensor(input_mask_1)
            attention_mask_2[index] = torch.Tensor(input_mask_2)
            ### Colin's ###
        
        return (token_ids, token_ids_1, token_ids_2, token_type_ids, attention_mask,
                attention_mask_1, attention_mask_2, labels.squeeze(1), sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_ids_1, token_ids_2, token_type_ids,
         attention_mask, attention_mask_1, attention_mask_2, labels, sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'token_ids_1': token_ids_1,
                'token_ids_2': token_ids_2,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'attention_mask_1': attention_mask_1,
                'attention_mask_2': attention_mask_2,
                'labels': labels,
                'sent_ids': sent_ids,
            }

        return batched_data


# Unlike SentencePairDataset, we do not load labels in SentencePairTestDataset.
class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args, max_sequence_length=128):
        self.dataset = dataset
        self.p = args
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenization.FullTokenizer(
                vocab_file='uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        batch_size = len(data)
        token_ids = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        token_type_ids = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.long)
        sent_ids = []
        
        for index, testing_case in enumerate(data):
            sent1 = testing_case[0]
            sent2 = testing_case[1]
            sent_id = testing_case[2]
        
            tokens_1 = self.tokenizer.tokenize(sent1)
            tokens_2 = self.tokenizer.tokenize(sent2)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_1, tokens_2, self.max_sequence_length - 3)

            tokens = ["[CLS]"] + tokens_1 + ["[SEP]"] + tokens_2 + ["[SEP]"]
            segment_ids = [0] * (len(tokens_1) + 2) + [1] * (len(tokens_2) + 1)
            input_mask = [1] * len(tokens)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            padding_length = self.max_sequence_length - len(tokens)
            input_ids.extend([0] * padding_length)
            segment_ids.extend([0] * padding_length)
            input_mask.extend([0] * padding_length)

            assert len(input_ids) == self.max_sequence_length
            assert len(input_mask) == self.max_sequence_length
            assert len(segment_ids) == self.max_sequence_length

            # append the features
            token_ids[index] = torch.Tensor(input_ids)
            token_type_ids[index] = torch.Tensor(segment_ids)
            attention_mask[index] = torch.Tensor(input_mask)
            sent_ids.append(sent_id)
        
        return (token_ids, token_type_ids, attention_mask, sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask, sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'sent_ids': sent_ids
            }

        return batched_data


def load_multitask_data(sentiment_filename, imdb_filename, paraphrase_filename,similarity_filename,split='train'):
    sentiment_data = []
    imdb_sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent,sent_id))
    else:
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label,sent_id))
        if(split == 'train'):
            with open(imdb_filename, 'r') as fp:
                for record in csv.DictReader(fp, delimiter='\t'):
                    sent = record['sentence'].lower().strip()
                    sent_id = record['id'].lower().strip()
                    label = int(record['sentiment'].strip())
                    if label not in num_labels:
                        num_labels[label] = len(num_labels)
                    imdb_sentiment_data.append((sent, label, sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")
    print(f"Loaded {len(imdb_sentiment_data)} {split} examples from {imdb_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(record['sentence2']),
                                            int(float(record['is_duplicate'])),sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2'])
                                        ,sent_id))
    else:
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']),sent_id))

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, imdb_sentiment_data, num_labels, paraphrase_data, similarity_data
