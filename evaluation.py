#!/usr/bin/env python3

'''
Multitask BERT evaluation functions.

When training your multitask model, you will find it useful to call
model_eval_multitask to evaluate your model on the 3 tasks' dev sets.
'''

import os
import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np

from utils import get_model


TQDM_DISABLE = os.environ.get('TQDM_DISABLE', 'False').lower() == 'true'


# Evaluate multitask model on SST only.
def model_eval_sst(dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'sst eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                        batch['labels'], batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model.predict_sentiment(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids


# Evaluate multitask model on dev sets.
def model_eval_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device, arg):
    model = get_model(model)
    model.eval()  # Switch to eval model, will turn off randomness like dropout.

    with torch.no_grad():
        sentiment_accuracy, paraphrase_accuracy, sts_corr = 0, 0, 0

        # Evaluate sentiment classification.
        sst_y_true = []
        sst_y_pred = []
        sst_sent_ids = []
        if arg == 'sst' or arg == 'all':
            for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'sst eval', disable=TQDM_DISABLE)):
                b_ids, b_token_type_ids, b_mask, b_labels, b_sent_ids = batch['token_ids'], batch['token_type_ids'], batch['attention_mask'], batch['labels'], batch['sent_ids']

                b_ids = b_ids.to(device)
                b_token_type_ids = b_token_type_ids.to(device)
                b_mask = b_mask.to(device)

                logits = model.predict_sentiment(b_ids, b_token_type_ids, b_mask)
                y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
                
                b_labels = b_labels.flatten().cpu().numpy()

                sst_y_pred.extend(y_hat)
                sst_y_true.extend(b_labels)
                sst_sent_ids.extend(b_sent_ids)

            sentiment_accuracy = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))

        # Evaluate paraphrase detection.
        para_y_true = []
        para_y_pred = []
        para_sent_ids = []
        if arg == 'para' or arg == 'all':
            for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'para eval', disable=TQDM_DISABLE)):
                (b_ids, b_token_type_ids, b_mask,
                 b_labels, b_sent_ids) = (batch['token_ids'], batch['token_type_ids'], batch['attention_mask'],
                              batch['labels'], batch['sent_ids'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_token_type_ids = b_token_type_ids.to(device)
                logits = model.predict_paraphrase(b_ids, b_token_type_ids, b_mask)
                y_hat = logits.sigmoid().round().flatten().cpu().numpy()
                b_labels = b_labels.flatten().cpu().numpy()

                para_y_pred.extend(y_hat)
                para_y_true.extend(b_labels)
                para_sent_ids.extend(b_sent_ids)

            paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))

        # Evaluate semantic textual similarity.
        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []
        if arg == 'sts' or arg == 'all':
            for step, batch in enumerate(tqdm(sts_dataloader, desc=f'sts eval', disable=TQDM_DISABLE)):
                (
                    token_ids,
                    token_ids_1,
                    token_ids_2,
                    token_type_ids,
                    attention_mask,
                    attention_mask_1,
                    attention_mask_2,
                    b_labels,
                    b_sent_ids,
                ) = (
                    batch["token_ids"],
                    batch["token_ids_1"],
                    batch["token_ids_2"],
                    batch["token_type_ids"],
                    batch["attention_mask"],
                    batch["attention_mask_1"],
                    batch["attention_mask_2"],
                    batch["labels"],
                    batch["sent_ids"],
                )

                token_ids = token_ids.to(device)
                token_ids_1 = token_ids_1.to(device)
                token_ids_2 = token_ids_2.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                attention_mask_1 = attention_mask_1.to(device)
                attention_mask_2 = attention_mask_2.to(device)
                b_labels = b_labels.type(torch.float32).to(device)

                logits = model.predict_similarity(token_ids, token_ids_1, token_ids_2, token_type_ids, attention_mask,
                                                  attention_mask_1, attention_mask_2)

                y_hat = logits.flatten().cpu().numpy()

                b_labels = b_labels.flatten().cpu().numpy()

                sts_y_pred.extend(y_hat)
                sts_y_true.extend(b_labels)
                sts_sent_ids.extend(b_sent_ids)
            pearson_mat = np.corrcoef(sts_y_pred, sts_y_true)
            sts_corr = pearson_mat[1][0]

        print(f'Sentiment classification accuracy: {sentiment_accuracy:.3f}')
        print(f'Paraphrase detection accuracy: {paraphrase_accuracy:.3f}')
        print(f'Semantic Textual Similarity correlation: {sts_corr:.3f}')

        return (sentiment_accuracy, sst_y_pred, sst_sent_ids,
                paraphrase_accuracy, para_y_pred, para_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids)


# Evaluate multitask model on test sets.
def model_eval_test_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device):
    model = get_model(model)
    model.eval()  # Switch to eval model, will turn off randomness like dropout.

    with torch.no_grad():
        # Evaluate sentiment classification.
        sst_y_pred = []
        sst_sent_ids = []
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'sst test eval', disable=TQDM_DISABLE)):
            b_ids, b_token_type_ids, b_mask, b_sent_ids = batch['token_ids'], batch['token_type_ids'], batch['attention_mask'], batch['sent_ids']

            b_ids = b_ids.to(device)
            b_token_type_ids = b_token_type_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model.predict_sentiment(b_ids, b_token_type_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_sent_ids.extend(b_sent_ids)

        # Evaluate paraphrase detection.
        para_y_pred = []
        para_sent_ids = []
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'para test eval', disable=TQDM_DISABLE)):
            (b_ids, b_token_type_ids, b_mask,
                 b_sent_ids) = (batch['token_ids'], batch['token_type_ids'], batch['attention_mask'],
                              batch['sent_ids'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_token_type_ids = b_token_type_ids.to(device)
            
            logits = model.predict_paraphrase(b_ids, b_token_type_ids, b_mask)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_sent_ids.extend(b_sent_ids)

        # Evaluate semantic textual similarity.
        sts_y_pred = []
        sts_sent_ids = []
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'sts test eval', disable=TQDM_DISABLE)):
            (
                token_ids,
                #token_ids_1,
                #token_ids_2,
                token_type_ids,
                attention_mask,
                #attention_mask_1,
                #attention_mask_2,
                b_labels,
            ) = (
                batch["token_ids"],
                #batch["token_ids_1"],
                #batch["token_ids_2"],
                batch["token_type_ids"],
                batch["attention_mask"],
                #batch["attention_mask_1"],
                #batch["attention_mask_2"],
                batch["sent_ids"],
            )

            token_ids = token_ids.to(device)
            #token_ids_1 = token_ids_1.to(device)
            #token_ids_2 = token_ids_2.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            #attention_mask_1 = attention_mask_1.to(device)
            #attention_mask_2 = attention_mask_2.to(device)
            #b_sent_ids = b_labels.type(torch.float32).to(device)
            
            logits = model.predict_similarity(token_ids, None, None, token_type_ids, attention_mask, None, None)
            y_hat = logits.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_sent_ids.extend(b_sent_ids)

        return (sst_y_pred, sst_sent_ids,
                para_y_pred, para_sent_ids,
                sts_y_pred, sts_sent_ids)
