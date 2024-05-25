"""
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
"""

import argparse
import datetime
import itertools
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bert import BertModel
from data import data_loaders_for_test, data_loaders_for_train_and_validation
from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, model_eval_sst, model_eval_test_multitask
from optimizer import AdamW
from utils import get_model, p_print
from dotenv import load_dotenv

load_dotenv(override=True)

TQDM_DISABLE = os.environ.get("TQDM_DISABLE", "False").lower() == "true"

# Set it to True to iterate over a small subset of the data to check for any implementation errors.
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        "nccl",  # NCCL backend optimized for NVIDIA GPUs
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


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
    """
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    """

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == "last-linear-layer":
                param.requires_grad = False
            elif config.fine_tune_mode == "full-model":
                param.requires_grad = True

        # SST: 5 class classification
        # negative, somewhat negative, neutral, somewhat positive, or positive.

        # according to documentaiton of SST, there are 5 labels
        assert len(config.sentiment_labels) == 5
        self.sst_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ELU(alpha=0.1),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(512, len(config.sentiment_labels)),
        )

        # Paraphrasing: Binary classification
        # we are concatenating the embeddings of the two sentences
        # and then passing them through a linear layer.
        self.para_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 512),
            nn.ELU(alpha=0.1),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(512, 1),
        )

        # SST: 6 class classification The similarity scores vary from 0 to 5
        # with 0 being the least similar and 5 being the most similar.
        '''self.sts_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ELU(alpha=0.1),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(512, 1),
        )'''
        self.sts_classifier = nn.Linear(config.hidden_size, config.hidden_size)

        # Cosine Implementation
        # self.sts_classifier = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ELU(alpha=0.1),
        #     nn.Dropout(config.hidden_dropout_prob),
        #     nn.Linear(config.hidden_size, config.hidden)
        # )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        "Takes a batch of sentences and produces embeddings for them."
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        # for now just returning bert embedding
        return self.bert.forward(input_ids, attention_mask)

    def predict_sentiment(self, input_ids, attention_mask):
        """Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        """
        # From the default handout: As a baseline, you should call the new forward() method above followed by a
        # dropout and linear layer as in classifier.py.'''
        pooler_output = self.bert.forward(input_ids, attention_mask)["pooler_output"]
        pooler_output = self.dropout(pooler_output)
        logits = self.sst_classifier(pooler_output)
        return logits

    def predict_paraphrase(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        """
        # concatenate inputs and attention masks
        output_1 = self.bert.forward(input_ids_1, attention_mask_1)["pooler_output"]
        output_2 = self.bert.forward(input_ids_2, attention_mask_2)["pooler_output"]

        # dimension
        output_cat = torch.cat((output_1, output_2), dim=1)
        output = self.dropout(output_cat)
        logits = self.para_classifier(output).squeeze()

        # we are using BCEWithLogitLoss, so no need to put sigmoid here
        return logits

    def predict_similarity(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        """
        output_1 = self.forward(input_ids_1, attention_mask_1)["pooler_output"]
        output_2 = self.forward(input_ids_2, attention_mask_2)["pooler_output"]

        output_1 = self.sts_classifier(output_1)
        output_2 = self.sts_classifier(output_2)

        cos_sim = torch.nn.functional.cosine_similarity(output_1, output_2, dim=-1)
        return 5 * torch.sigmoid(5 * cos_sim) # cover larger range all values between 0 and 1

        '''print(output_1.shape)

        # dimension
        output_cat = torch.cat((output_1, output_2), dim=1)
        output = self.dropout(output_cat)
        logits = self.sts_classifier(output).squeeze()
        # scale it between 0 and 5 so that we can calculate MSE
        logits = 5 * torch.sigmoid(logits)

        # cosine implementation
        # output_1 = self.dropout(output_1)
        # output_2 = self.dropout(output_2)
        # proj_1 = self.sts_classifier(output_1)
        # proj_2 = self.sts_classifier(output_2)
        # logits = F.cosine_similarity(proj_1, proj_2, dim=1).squeeze()
        # logits = (logits + 1) * 2.5

        return logits'''


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "model_config": config,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    p_print(f"save the model to {filepath}")


def train(batch, device, model, type):
    loss = None
    model = get_model(model)

    if type == "sst":
        b_ids, b_mask, b_labels = (
            batch["token_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)

        logits = model.predict_sentiment(b_ids, b_mask)
        # logits dim: B, class_size. b_labels dim: B, (class indices)
        loss = nn.CrossEntropyLoss(reduction="mean")(logits, b_labels)

    elif type == "para":
        (
            token_ids_1,
            token_type_ids_1,
            attention_mask_1,
            token_ids_2,
            token_type_ids_2,
            attention_mask_2,
            b_labels,
            sent_ids,
        ) = (
            batch["token_ids_1"],
            batch["token_type_ids_1"],
            batch["attention_mask_1"],
            batch["token_ids_2"],
            batch["token_type_ids_2"],
            batch["attention_mask_2"],
            batch["labels"],
            batch["sent_ids"],
        )

        token_ids_1 = token_ids_1.to(device)
        token_type_ids_1 = token_type_ids_1.to(
            device
        )  # need to modify bert embedding to use this later
        attention_mask_1 = attention_mask_1.to(device)

        token_ids_2 = token_ids_2.to(device)
        token_type_ids_2 = token_type_ids_2.to(
            device
        )  # need to modify bert embedding to use this later
        attention_mask_2 = attention_mask_2.to(device)
        b_labels = b_labels.type(torch.float32).to(device)

        logits = model.predict_paraphrase(
            token_ids_1, attention_mask_1, token_ids_2, attention_mask_2
        )
        # logits dim: B, b_labels dim: B
        loss = nn.BCEWithLogitsLoss(reduction="mean")(logits, b_labels)

    elif type == "sts":
        (
            token_ids_1,
            token_type_ids_1,
            attention_mask_1,
            token_ids_2,
            token_type_ids_2,
            attention_mask_2,
            b_labels,
            sent_ids,
        ) = (
            batch["token_ids_1"],
            batch["token_type_ids_1"],
            batch["attention_mask_1"],
            batch["token_ids_2"],
            batch["token_type_ids_2"],
            batch["attention_mask_2"],
            batch["labels"],
            batch["sent_ids"],
        )

        token_ids_1 = token_ids_1.to(device)
        token_type_ids_1 = token_type_ids_1.to(
            device
        )  # need to modify bert embedding to use this later
        attention_mask_1 = attention_mask_1.to(device)

        token_ids_2 = token_ids_2.to(device)
        token_type_ids_2 = token_type_ids_2.to(
            device
        )  # need to modify bert embedding to use this later
        attention_mask_2 = attention_mask_2.to(device)
        b_labels = b_labels.type(torch.float32).to(device)

        # logits dim: B, b_labels dim: B. value of logits should be between 0 to 5
        logits = model.predict_similarity(
            token_ids_1, attention_mask_1, token_ids_2, attention_mask_2
        )
        loss = nn.MSELoss(reduction="mean")(logits, b_labels)

    # Run backprop for the loss from the task
    loss.backward()

    return loss


def train_multitask(rank, world_size, args):
    """Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    """

    if args.use_gpu:
        setup(rank, world_size)
        device = torch.device(rank)
    else:
        device = torch.device("cpu")

    if rank == 0:
        run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-complex-repr-trimmed-para-per-epoch-final"
        summary_writer = SummaryWriter(
            f"runs/{run_name}"
        )
        p_print(f'\n\n\n*** Train multitask {run_name} ***')
        p_print('device: {}, debug: {}'.format(device, DEBUG))

    use_multi_gpu = False
    if world_size > 1:
        use_multi_gpu = True
    
    # Get data loaders for training and validation.
    (
        sentiment_labels,
        para_train_dataloader,
        sst_train_dataloader,
        sts_train_dataloader,
        para_dev_dataloader,
        sst_dev_dataloader,
        sts_dev_dataloader,
    ) = data_loaders_for_train_and_validation(args, rank, world_size, use_multi_gpu, debug=DEBUG)

    # Init model.
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "sentiment_labels": sentiment_labels,
        "hidden_size": 768,
        "data_dir": ".",
        "fine_tune_mode": args.fine_tune_mode,
    }

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    if world_size > 0:
        model = DDP(model, device_ids=[rank])

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_overall_accuracy = 0

    # cycle_sst_loader = itertools.cycle(sst_train_dataloader)
    cycle_para_loader = itertools.cycle(para_train_dataloader)

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()

        # Initialize variables
        (sst_train_loss,
        sst_num_batches,
        para_train_loss,
        para_num_batches,
        sts_train_loss,
        sts_num_batches,
        ) = (0, 0, 0, 0, 0, 0)

        # since paraphrase has lots of data, lets try to limit it to 1800 batches per epoch
        # this way, we can cover all the data at least twice in 10 epochs
        if DEBUG:
            loop_size = 10
        else:
            loop_size = 1800

        # Paraphrase training
        for step in tqdm(range(loop_size), desc="para train", disable=TQDM_DISABLE):
            para_batch = next(cycle_para_loader)
            optimizer.zero_grad()
            para_training_loss = train(para_batch, device, model, "para")
            para_train_loss += para_training_loss.item()
            para_num_batches += 1
            optimizer.step()

            if rank == 0 and step % 10 == 0:
                overall_steps = epoch * len(para_train_dataloader) + step
                summary_writer.add_scalar(
                    "para_train_loss", para_training_loss.item(), overall_steps
                )

        # SST training
        for step, sst_batch in enumerate(
            tqdm(sst_train_dataloader, desc="sst train", disable=TQDM_DISABLE)
        ):
            optimizer.zero_grad()
            sst_training_loss = train(sst_batch, device, model, "sst")
            sst_train_loss += sst_training_loss.item()
            sst_num_batches += 1
            optimizer.step()

            if rank == 0 and step % 10 == 0:
                overall_steps = epoch * len(sst_train_dataloader) + step
                summary_writer.add_scalar(
                    "sst_train_loss", sst_training_loss.item(), overall_steps
                )

        # STS training
        for step, sts_batch in enumerate(
            tqdm(sts_train_dataloader, desc="sts train", disable=TQDM_DISABLE)
        ):
            optimizer.zero_grad()
            sts_training_loss = train(sts_batch, device, model, "sts")
            sts_train_loss += sts_training_loss.item()
            sts_num_batches += 1
            optimizer.step()

            if rank == 0 and step % 10 == 0:
                overall_steps = epoch * len(sts_train_dataloader) + step
                summary_writer.add_scalar(
                    "sts_train_loss", sts_training_loss.item(), overall_steps
                )

        (
            sst_dev_acc,
            _,
            _,
            para_dev_acc,
            _,
            _,
            sts_dev_corr,
            *_,
        ) = model_eval_multitask(
            sst_dev_dataloader,
            para_dev_dataloader,
            sts_dev_dataloader,
            model,
            device,
            args.train,
        )

        if rank == 0:
            overall_accuracy = (sst_dev_acc + (sts_dev_corr + 1) / 2 + para_dev_acc) / 3

            summary_writer.add_scalar("sst_dev_acc", sst_dev_acc, epoch)
            summary_writer.add_scalar("para_dev_acc", para_dev_acc, epoch)
            summary_writer.add_scalar("sts_dev_corr", sts_dev_corr, epoch)
            summary_writer.add_scalar("overall_accuracy", overall_accuracy, epoch)

            # save new mode if at least one of the dev accuracy is better
            p_print(
                f"Epoch {epoch} overall accuracy: {overall_accuracy}, previous accuracy: {best_overall_accuracy}"
            )
            if overall_accuracy > best_overall_accuracy:
                p_print(
                    f"Saving model at epoch {epoch}, previous accuracy: {best_overall_accuracy}, new accuracy: {overall_accuracy}"
                )
                save_model(model, optimizer, args, config, args.filepath)
                best_overall_accuracy = overall_accuracy

        sts_train_loss = sts_train_loss / sts_num_batches
        para_train_loss = para_train_loss / para_num_batches
        sst_train_loss = sst_train_loss / sst_num_batches

        p_print(
            f"Epoch {epoch}: Rank: {rank} sst train loss :: {sst_train_loss :.3f}, para train loss :: {para_train_loss :.3f}, sts train loss :: {sts_train_loss :.3f}, sst dev acc :: {sst_dev_acc :.3f}, para dev acc :: {para_dev_acc :.3f}, sts dev corr :: {sts_dev_corr :.3f}"
        )
    
    if args.use_gpu:
        cleanup()


def test_multitask(args):
    """Test and save predictions on the dev and test sets of all three tasks."""
    with torch.no_grad():
        saved = torch.load(args.filepath)
        config = saved["model_config"]

        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

        model = MultitaskBERT(config)

        if args.use_gpu:
            model = nn.DataParallel(model)

        model.to(device)
        model.load_state_dict(saved["model"])

        p_print(f"Loaded model to test from {args.filepath}")

        (
            para_test_dataloader,
            sst_test_dataloader,
            sts_test_dataloader,
            para_dev_dataloader,
            sst_dev_dataloader,
            sts_dev_dataloader,
        ) = data_loaders_for_test(args, use_multi_gpu=False, debug=DEBUG)

        (
            dev_sentiment_accuracy,
            dev_sst_y_pred,
            dev_sst_sent_ids,
            dev_paraphrase_accuracy,
            dev_para_y_pred,
            dev_para_sent_ids,
            dev_sts_corr,
            dev_sts_y_pred,
            dev_sts_sent_ids,
        ) = model_eval_multitask(
            sst_dev_dataloader,
            para_dev_dataloader,
            sts_dev_dataloader,
            model,
            device,
            args.train,
        )

        (
            test_sst_y_pred,
            test_sst_sent_ids,
            test_para_y_pred,
            test_para_sent_ids,
            test_sts_y_pred,
            test_sts_sent_ids,
        ) = model_eval_test_multitask(
            sst_test_dataloader, para_test_dataloader, sts_test_dataloader, model, device
        )

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
    parser.add_argument(
        "--fine-tune-mode",
        type=str,
        help="last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well",
        choices=("last-linear-layer", "full-model"),
        default="last-linear-layer",
    )
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument(
        "--sst_dev_out", type=str, default="predictions/sst-dev-output.csv"
    )
    parser.add_argument(
        "--sst_test_out", type=str, default="predictions/sst-test-output.csv"
    )

    parser.add_argument(
        "--para_dev_out", type=str, default="predictions/para-dev-output.csv"
    )
    parser.add_argument(
        "--para_test_out", type=str, default="predictions/para-test-output.csv"
    )

    parser.add_argument(
        "--sts_dev_out", type=str, default="predictions/sts-dev-output.csv"
    )
    parser.add_argument(
        "--sts_test_out", type=str, default="predictions/sts-test-output.csv"
    )

    parser.add_argument(
        "--batch_size",
        help="sst: 64, cfimdb: 8 can fit a 12GB GPU",
        type=int,
        default=32,
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument(
        "--train", type=str, help="sst, para, sts, or all", default="all"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = (
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt"  # Save path.
    )
    seed_everything(args.seed)  # Fix the seed for reproducibility.

    # If CUDA is available, use it.
    world_size = torch.cuda.device_count()
    if world_size > 0:
        mp.spawn(
            train_multitask,
            args=(world_size, args),  # 10 epochs, for example
            nprocs=world_size,
            join=True,
        )
    else:
        train_multitask(0, 0, args)

    test_multitask(args)
