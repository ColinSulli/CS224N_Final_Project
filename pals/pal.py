import torch
from pals.modeling import BertConfig, BertForMultiTask


def get_pal(num_task):
    bert_config = BertConfig.from_json_file('pals_config.json')
    bert_config.num_tasks = num_task

    model = BertForMultiTask(bert_config)

    # init_checkpoint
    init_checkpoint = 'uncased_L-12_H-768_A-12/pytorch_model.bin'

    if init_checkpoint is not None:
        partial = torch.load(init_checkpoint, map_location='cpu')
        model_dict = model.bert.state_dict()
        update = {}
        for n, p in model_dict.items():
            if 'aug' in n or 'mult' in n:
                update[n] = p
                if 'pooler.mult' in n and 'bias' in n:
                    update[n] = partial['pooler.dense.bias']
                if 'pooler.mult' in n and 'weight' in n:
                    update[n] = partial['pooler.dense.weight']
            else:
                update[n] = partial[n]
        model.bert.load_state_dict(update)
    
    return model
        

    