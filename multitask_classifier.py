'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import itertools
import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch.cuda.amp import autocast, GradScaler
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask
from utils import p_print


TQDM_DISABLE=True


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        
        # SST: 5 class classification
        # negative, somewhat negative, neutral, somewhat positive, or positive.

        # according to documentaiton of SST, there are 5 labels
        assert(len(config.sentiment_labels) == 5)
        self.sst_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ELU(alpha=0.1),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(512, len(config.sentiment_labels))
        )
        
        # Paraphrasing: Binary classification
        # we are concatenating the embeddings of the two sentences 
        # and then passing them through a linear layer.
        self.para_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 512),
            nn.ELU(alpha=0.1),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(512, 1)
        )
        
        # SST: 6 class classification The similarity scores vary from 0 to 5
        # with 0 being the least similar and 5 being the most similar.
        '''self.sts_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 512),
            nn.ELU(alpha=0.1),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(512, 1)
        )'''
        self.sts_classifier = nn.Linear(config.hidden_size * 2, 1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO

        # for now just returning bert embedding
        return self.bert.forward(input_ids, attention_mask)

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO

        # From the default handout: As a baseline, you should call the new forward() method above followed by a
        # dropout and linear layer as in classifier.py.'''
        pooler_output = self.bert.forward(input_ids, attention_mask)['pooler_output']
        pooler_output = self.dropout(pooler_output)
        logits = self.sst_classifier(pooler_output)
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO

        # concatenate inputs and attention masks
        output_1 = self.bert.forward(input_ids_1, attention_mask_1)['pooler_output']
        output_2 = self.bert.forward(input_ids_2, attention_mask_2)['pooler_output']

        # dimension
        output_cat = torch.cat((output_1, output_2), dim=1)
        output = self.dropout(output_cat)
        logits = self.para_classifier(output).squeeze()

        return logits

    def train_similarity(self,
                         input_ids_1, attention_mask_1,
                         input_ids_2, attention_mask_2, b_labels):
        att_1 = self.forward(input_ids_1, attention_mask_1)['pooler_output']
        att_2 = self.forward(input_ids_2, attention_mask_2)['pooler_output']

        output_cat = torch.cat((att_1, att_2), dim=1)
        output_cat = self.dropout(output_cat)
        sim_output = self.para_classifier(output_cat)

        loss = torch.nn.CosineEmbeddingLoss(reduction='sum')(sim_output, output_cat, b_labels.to(torch.float).view(-1)) / args.batch_size

        return loss

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''

        ### TODO
        # cosine similarity
        att_1 = self.bert.forward(input_ids_1, attention_mask_1)['pooler_output']
        att_2 = self.bert.forward(input_ids_2, attention_mask_2)['pooler_output']
        output_cat = torch.cat((output_1, output_2), dim=1)

        output_cat = self.dropout(output_cat)
        sim_output = self.para_classifier(output_cat)

        input_cos = F.cosine_similarity(sim_output, output_cat, dim=1)
        input_cos = 5 * torch.sigmoid(input_cos)
        input_cos = torch.round(input_cos)

        return input_cos

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    p_print(f"save the model to {filepath}")


def train(batch, device, optimizer, model, type, scaler):
    loss = None

    if type == 'sst':
        b_ids, b_mask, b_labels = (batch['token_ids'],
                                   batch['attention_mask'], batch['labels'])
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model.predict_sentiment(b_ids, b_mask)

            # logits dim: B, class_size. b_labels dim: B, (class indices)
            loss = nn.CrossEntropyLoss(reduction='mean')(logits, b_labels)

    elif type == 'para':
        (token_ids_1, token_type_ids_1, attention_mask_1, token_ids_2,
         token_type_ids_2, attention_mask_2, b_labels, sent_ids) = \
            (batch['token_ids_1'], batch['token_type_ids_1'], batch['attention_mask_1'], batch['token_ids_2'],
             batch['token_type_ids_2'], batch['attention_mask_2'], batch['labels'], batch['sent_ids'])

        token_ids_1 = token_ids_1.to(device)
        token_type_ids_1 = token_type_ids_1.to(device)  # need to modify bert embedding to use this later
        attention_mask_1 = attention_mask_1.to(device)

        token_ids_2 = token_ids_2.to(device)
        token_type_ids_2 = token_type_ids_2.to(device)  # need to modify bert embedding to use this later
        attention_mask_2 = attention_mask_2.to(device)
        b_labels = b_labels.to(device, dtype=torch.float32)

        with autocast():
            logits = model.predict_paraphrase(token_ids_1, attention_mask_1, token_ids_2, attention_mask_2)
            # logits dim: B, b_labels dim: B
            loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, b_labels)

    elif type == 'sts':
        (token_ids_1, token_type_ids_1, attention_mask_1, token_ids_2,
         token_type_ids_2, attention_mask_2, b_labels, sent_ids) = \
            (batch['token_ids_1'], batch['token_type_ids_1'], batch['attention_mask_1'], batch['token_ids_2'],
             batch['token_type_ids_2'], batch['attention_mask_2'], batch['labels'], batch['sent_ids'])

        token_ids_1 = token_ids_1.to(device)
        token_type_ids_1 = token_type_ids_1.to(device)  # need to modify bert embedding to use this later
        attention_mask_1 = attention_mask_1.to(device)

        token_ids_2 = token_ids_2.to(device)
        token_type_ids_2 = token_type_ids_2.to(device)  # need to modify bert embedding to use this later
        attention_mask_2 = attention_mask_2.to(device)
        b_labels = b_labels.to(device, dtype=torch.float32)

        with autocast():
            # logits dim: B, b_labels dim: B. value of logits should be between 0 to 5
            loss = model.train_similarity(token_ids_1, attention_mask_1, token_ids_2, attention_mask_2, b_labels)

    # Run backprop for the loss from the task
    scaler.scale(loss).backward()

    return loss


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    p_print('using device', device)

    summary_writer = SummaryWriter(f'runs/train-multitask-cycle-loader')
    scaler = GradScaler()

    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, sentiment_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train,
                                                                                            args.para_train,
                                                                                            args.sts_train,
                                                                                            split='train')
    sst_dev_data, sentiment_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev,
                                                                                      args.sts_dev, split='train')

    # size of sts_train_data: 6040
    # size of sst_train_data: 8544
    # size of para_train_data: 283003

    # smallest data is sts data, so lets loop over that and
    # create a cycle loader for the other two datasets

    # lets go with small batch size for sts and sst and increase the
    # batch size for para data. Since para data size is almost 46 times
    # sts data, with 15 times batch size, we should have covered all the 
    # para data in every 3rd epoch
    train_batch_size_sts_and_sst = 4
    train_batch_size_para = 32

    # SST Data
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=train_batch_size_sts_and_sst,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # Para Data
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=train_batch_size_para,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    # STS Data
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=train_batch_size_sts_and_sst,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'sentiment_labels': sentiment_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_sst_dev = 0
    best_para_dev = 0
    best_sts_corr = 0

    cycle_sst_loader = itertools.cycle(sst_train_dataloader)
    cycle_para_loader = itertools.cycle(para_train_dataloader)

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()

        sst_train_loss, sst_num_batches, para_train_loss, para_num_batches, sts_train_loss, sts_num_batches = 0, 0, 0, 0, 0, 0

        for step, sts_batch in enumerate(sts_train_dataloader):
            sst_batch = next(cycle_sst_loader)
            para_batch = next(cycle_para_loader)

            optimizer.zero_grad()

            # STS symantic textual simiarity training
            sts_training_loss = train(sts_batch, device, optimizer, model, 'sts', scaler)
            sts_train_loss += sts_training_loss.item()
            sts_num_batches += 1

            # paraphrase training
            para_training_loss = train(para_batch, device, optimizer, model, 'para', scaler)
            para_train_loss += para_training_loss.item()
            para_num_batches += 1

            # SST sentiment training
            sst_training_loss = train(sst_batch, device, optimizer, model, 'sst', scaler)
            sst_train_loss += sst_training_loss.item()
            sst_num_batches += 1

            scaler.step(optimizer)
            scaler.update()

            if step % 100 == 0:
                overall_steps = epoch * len(sts_train_dataloader) + step
                summary_writer.add_scalar('sts_train_loss', sts_training_loss.item(), overall_steps)
                summary_writer.add_scalar('sst_train_loss', sst_training_loss.item(), overall_steps)
                summary_writer.add_scalar('para_train_loss', para_training_loss.item(), overall_steps)

        # Skipping training accuracy since we already have training loss
        # sst_train_acc, _, _, \
        # para_train_acc, _, _, \
        # sts_train_corr, *_ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device, args.train)

        sst_dev_acc, _, sst_sent_ids, \
            para_dev_acc, _, para_sent_ids, \
            sts_dev_corr, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model,
                                                    device, args.train)

        # summary_writer.add_scalar('sst_train_acc', sst_train_acc, epoch)
        # summary_writer.add_scalar('para_train_acc', para_train_acc, epoch)
        summary_writer.add_scalar('sst_dev_acc', sst_dev_acc, epoch)
        summary_writer.add_scalar('para_dev_acc', para_dev_acc, epoch)
        summary_writer.add_scalar('sts_dev_corr', sts_dev_corr, epoch)

        # save new mode if at least one of the dev accuracy is better
        if sst_dev_acc > best_sst_dev or para_dev_acc > best_para_dev or sts_dev_corr > best_sts_corr:
            p_print(
                f"Saving model at epoch {epoch}, previous dev accuracies: {best_sst_dev, best_para_dev, best_sts_corr}, new dev accuracies: {sst_dev_acc, para_dev_acc, sts_dev_corr}")
            save_model(model, optimizer, args, config, args.filepath)
            if sst_dev_acc > best_sst_dev:
                best_sst_dev = sst_dev_acc
            if para_dev_acc > best_para_dev:
                best_para_dev = para_dev_acc
            if sts_dev_corr > best_sts_corr:
                best_sts_corr = sts_dev_corr

        sts_train_loss = sts_train_loss / sts_num_batches
        para_train_loss = para_train_loss / para_num_batches
        sst_train_loss = sst_train_loss / sst_num_batches

        p_print(
            f"Epoch {epoch}: sst train loss :: {sst_train_loss :.3f}, para train loss :: {para_train_loss :.3f}, sts train loss :: {sts_train_loss :.3f}, sst dev acc :: {sst_dev_acc :.3f}, para dev acc :: {para_dev_acc :.3f}, sts dev corr :: {sts_dev_corr :.3f}")


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        p_print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device, args.train)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            p_print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            p_print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            p_print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--train", type=str, help="sst, para, sts, or all", default="all")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
