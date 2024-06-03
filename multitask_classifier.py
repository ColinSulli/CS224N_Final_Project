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
import torch.nn.functional as F
from dotenv import load_dotenv
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import data_loaders_for_test, data_loaders_for_train_and_validation
from evaluation import model_eval_multitask, model_eval_test_multitask
from optimizer import AdamW
from pals.pal import get_pal
from utils import get_model, p_print

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

        self.bert = get_pal(num_task=3)

        self.task_ids = {
            "para": 0,
            "sst": 1,
            "sts": 2,
        }

        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == "last-linear-layer":
                param.requires_grad = False
            elif config.fine_tune_mode == "full-model":
                param.requires_grad = True

        # Paraphrasing: Binary classification
        # we are concatenating the embeddings of the two sentences
        # and then passing them through a linear layer.
        self.para_classifier = nn.Linear(config.hidden_size * 2, 1)

        # SST: 5 class classification
        # negative, somewhat negative, neutral, somewhat positive, or positive.
        # according to documentaiton of SST, there are 5 labels
        assert len(config.sentiment_labels) == 5
        self.sst_classifier = nn.Linear(config.hidden_size, 5)

        # SST: regression between 0 and 6
        # with 0 being the least similar and 5 being the most similar.
        self.sts_classifier = nn.Linear(config.hidden_size * 2, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids, attention_mask, task_id):
        "Takes a batch of sentences and produces embeddings for them."
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        # for now just returning bert embedding
        return self.bert.forward(input_ids, token_type_ids, attention_mask, task_id)

    def predict_paraphrase(self, input_ids, token_type_ids, attention_mask):
        """Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        """
        # concatenate inputs and attention masks
        output = self.forward(
            input_ids, token_type_ids, attention_mask, self.task_ids["para"]
        )

        output_1 = self.dropout(output)
        output_2 = self.dropout(output)
        output = torch.cat((output_1, output_2), dim=1)

        logits = self.para_classifier(output).squeeze()

        # we are using BCEWithLogitLoss, so no need to put sigmoid here
        return logits

    def predict_sentiment(self, input_ids, token_type_ids, attention_mask):
        """Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        """
        # From the default handout: As a baseline, you should call the new forward() method above followed by a
        # dropout and linear layer as in classifier.py.'''
        pooler_output = self.forward(
            input_ids,
            token_type_ids,
            attention_mask,
            task_id=self.task_ids["sst"],
        )
        logits = self.sst_classifier(pooler_output).squeeze()

        # we are using CrossEntropyLoss, so no need to put softmax here
        return logits

    def predict_similarity(self, input_ids, token_type_ids, attention_mask):
        """Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        """
        # concatenate inputs and attention masks

        #print(input_ids)
        #print(input_ids.shape)

        output = self.forward(
            input_ids, token_type_ids, attention_mask, self.task_ids["sts"]
        )

        #print(output.shape)

        output_even = output[::2, :]
        output_odd = output[1::2, :]

        #print(output_even.shape)
        #print(output_odd.shape)

        output_1 = self.dropout(output)
        output_2 = self.dropout(output)
        output = torch.cat((output_1, output_2), dim=1)

        logits = self.sts_classifier(output)

        # normalise between 0 to 5
        logits = torch.sigmoid(logits).squeeze() * 5.0

        # we are using MSELoss, so no need to put sigmoid here
        return logits


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
        b_ids, b_token_type_ids, b_mask, b_labels = (
            batch["token_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
            batch["labels"],
        )
        b_ids = b_ids.to(device)
        b_token_type_ids = b_token_type_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)
        logits = model.predict_sentiment(b_ids, b_token_type_ids, b_mask)

        # logits dim: B, class_size. b_labels dim: B, (class indices)
        # expects un-normalised logits
        loss = nn.CrossEntropyLoss(reduction="mean")(logits, b_labels)

    elif type == "para":
        (
            token_ids,
            token_type_ids,
            attention_mask,
            b_labels,
        ) = (
            batch["token_ids"],
            batch["token_type_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        token_ids = token_ids.to(device)
        token_type_ids = token_type_ids.to(
            device
        )  # need to modify bert embedding to use this later
        attention_mask = attention_mask.to(device)
        b_labels = b_labels.type(torch.float32).to(device)

        logits = model.predict_paraphrase(token_ids, token_type_ids, attention_mask)

        # logits dim: B, b_labels dim: B
        loss = nn.BCEWithLogitsLoss(reduction="mean")(logits, b_labels)

    elif type == "sts":
        (
            token_ids,
            token_type_ids,
            attention_mask,
            b_labels,
        ) = (
            batch["token_ids"],
            batch["token_type_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        token_ids = token_ids.to(device)
        token_type_ids = token_type_ids.to(
            device
        )  # need to modify bert embedding to use this later
        attention_mask = attention_mask.to(device)

        b_labels = b_labels.type(torch.float32).to(device)

        # logits dim: B, b_labels dim: B. value of logits should be between 0 to 5
        logits = model.predict_similarity(token_ids, token_type_ids, attention_mask)
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
        run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-pal-annealed-sigmoid"
        summary_writer = SummaryWriter(f"runs/{run_name}")
        p_print(f"\n\n\n*** Train multitask {run_name} ***")
        p_print("device: {}, debug: {}".format(device, DEBUG))
        p_print("args: ", args)

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
    ) = data_loaders_for_train_and_validation(
        args, rank, world_size, use_multi_gpu, debug=DEBUG
    )

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
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_overall_accuracy = 0

    # cycle_sst_loader = itertools.cycle(sst_train_dataloader)
    # cycle_para_loader = itertools.cycle(para_train_dataloader)

    # para, sst, sts
    loaders = [
        itertools.cycle(para_train_dataloader),
        itertools.cycle(sst_train_dataloader),
        itertools.cycle(sts_train_dataloader),
    ]

    # Run for the specified number of epochs.

    # 600 steps per task
    # PAL paper was running it for 300 * tasks but since I have had to
    # reduce the batch size, I am increasing the steps per epoch
    if DEBUG:
        steps_per_epoch = 10
        probs = [1, 1, 1]
    else:
        steps_per_epoch = 600 * 3
        probs = [283003, 8544, 6040]

    ### Load previous Crash Begin ###
    saved = torch.load('/home/cmsstanfordhw/Final_Project/CS224N_Final_Project/2024-06-03_21-20-14-full-model-3-1e-05-multitask.pt')
        # .46 one from today saved = torch.load('/home/cmsstanfordhw/Final_Project/CS224N_Final_Project/2024-06-03_14-31-27-full-model-10-2e-05-multitask.pt')
    config = saved["model_config"]
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = MultitaskBERT(config)
    if args.use_gpu:
        model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(saved["model"])


    start_epoch = 100
    for epoch in range(start_epoch, args.epochs):
        model.train()

        # annealed sampling
        # para, sst, sts
        alpha = 1.0 - 0.8 * epoch / (args.epochs - 1)
        probs = [p**alpha for p in probs]
        tot = sum(probs)
        probs = [p / tot for p in probs]

        # Initialize variables
        (
            sst_train_loss,
            sst_num_batches,
            para_train_loss,
            para_num_batches,
            sts_train_loss,
            sts_num_batches,
        ) = (0, 0, 0, 0, 0, 0)

        # Paraphrase training
        for step in tqdm(
            range(steps_per_epoch), desc=f"epoch {epoch}", disable=TQDM_DISABLE
        ):
            overall_steps = epoch * steps_per_epoch + step

            # get task_id
            task_id = np.random.choice([0, 1, 2], p=probs)
            task_loader = loaders[task_id]
            task_batch = next(task_loader)

            # take a step
            optimizer.zero_grad()
            if task_id == 0:
                para_batch = task_batch
                para_training_loss = train(para_batch, device, model, "para")
                para_train_loss += para_training_loss.item()
                para_num_batches += 1

                if rank == 0 and step % 10 == 0:
                    summary_writer.add_scalar(
                        "para_train_loss", para_training_loss.item(), overall_steps
                    )
            elif task_id == 1:
                sst_batch = task_batch
                sst_training_loss = train(sst_batch, device, model, "sst")
                sst_train_loss += sst_training_loss.item()
                sst_num_batches += 1

                if rank == 0 and step % 10 == 0:
                    summary_writer.add_scalar(
                        "sst_train_loss", sst_training_loss.item(), overall_steps
                    )
            elif task_id == 2:
                sts_batch = task_batch
                sts_training_loss = train(sts_batch, device, model, "sts")
                sts_train_loss += sts_training_loss.item()
                sts_num_batches += 1

                if rank == 0 and step % 10 == 0:
                    summary_writer.add_scalar(
                        "sts_train_loss", sts_training_loss.item(), overall_steps
                    )
            else:
                raise Exception("invalid task_id")

            optimizer.step()

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

        # while debgging, we may not encounter batches, so avoid division by 0
        sts_train_loss = sts_train_loss / (sts_num_batches + 1e-9)
        para_train_loss = para_train_loss / (para_num_batches + 1e-9)
        sst_train_loss = sst_train_loss / (sst_num_batches + 1e-9)

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
            sst_test_dataloader,
            para_test_dataloader,
            sts_test_dataloader,
            model,
            device,
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
        default=16,
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
    args.filepath = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt"  # Save path.
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
