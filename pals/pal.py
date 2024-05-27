import torch
from pals.modeling import BertConfig, BertForMultiTask


def get_pal(num_task):
    bert_config = BertConfig.from_json_file("pals_config.json")
    bert_config.num_tasks = num_task

    model = BertForMultiTask(bert_config)

    # init_checkpoint
    init_checkpoint = "uncased_L-12_H-768_A-12/pytorch_model.bin"

    if init_checkpoint is not None:
        partial = torch.load(init_checkpoint, map_location="cpu")
        model_dict = model.bert.state_dict()
        update = {}
        for n, p in model_dict.items():
            if "aug" in n or "mult" in n:
                update[n] = p
                if "pooler.mult" in n and "bias" in n:
                    update[n] = partial["pooler.dense.bias"]
                if "pooler.mult" in n and "weight" in n:
                    update[n] = partial["pooler.dense.weight"]
            else:
                update[n] = partial[n]
        model.bert.load_state_dict(update)

    return model


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
