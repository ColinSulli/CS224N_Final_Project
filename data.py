from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
)
from torch.utils.data import DistributedSampler, DataLoader


def data_loaders_for_train_and_validation(args, rank, world_size, use_multi_gpu=False, debug=False):
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, sentiment_labels, para_train_data, sts_train_data = (
        load_multitask_data(
            args.sst_train, args.para_train, args.sts_train, split="train"
        )
    )
    sst_dev_data, sentiment_labels, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split="train"
    )

    # size of sts_train_data: 6040
    # size of sst_train_data: 8544
    # size of para_train_data: 283003

    # If we wish to debug the code, we can reduce the size of the data
    if debug:
        sst_train_data = sst_train_data[:100]
        para_train_data = para_train_data[:100]
        sts_train_data = sts_train_data[:100]
        sst_dev_data = sst_dev_data[:100]
        para_dev_data = para_dev_data[:100]
        sts_dev_data = sts_dev_data[:100]

    # SST Data
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    # Para Data
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    # STS Data
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    # Configuration for each data set
    # format: name, data, batch_size, shuffle
    datasets = [
        ("sst_train", sst_train_data, args.batch_size, True),
        ("sst_dev", sst_dev_data, args.batch_size, False),
        ("para_train", para_train_data, args.batch_size, True),
        ("para_dev", para_dev_data, args.batch_size, False),
        ("sts_train", sts_train_data, args.batch_size, True),
        ("sts_dev", sts_dev_data, args.batch_size, False),
    ]

    # create dataloaders
    dataloaders = {}
    for name, data, batch_size, should_shuffle in datasets:
        dataloaders[name] = create_data_loader(
            data,
            batch_size,
            data.collate_fn,
            use_multi_gpu,
            world_size,
            rank,
            should_shuffle,
        )

    return (
        sentiment_labels,
        dataloaders["para_train"],
        dataloaders["sst_train"],
        dataloaders["sts_train"],
        dataloaders["para_dev"],
        dataloaders["sst_dev"],
        dataloaders["sts_dev"],
    )


def data_loaders_for_test(args, use_multi_gpu=False, debug=False):
    sst_test_data, num_labels, para_test_data, sts_test_data = load_multitask_data(
        args.sst_test, args.para_test, args.sts_test, split="test"
    )

    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split="dev"
    )

    if debug:
        sst_test_data = sst_test_data[:100]
        para_test_data = para_test_data[:100]
        sts_test_data = sts_test_data[:100]
        sst_dev_data = sst_dev_data[:100]
        para_dev_data = para_dev_data[:100]
        sts_dev_data = sts_dev_data[:100]

    sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    para_test_data = SentencePairTestDataset(para_test_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    sts_test_data = SentencePairTestDataset(sts_test_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    datasets = [
        ("sst_test", sst_test_data, args.batch_size, True),
        ("sst_dev", sst_dev_data, args.batch_size, False),
        ("para_test", para_test_data, args.batch_size, True),
        ("para_dev", para_dev_data, args.batch_size, False),
        ("sts_test", sts_test_data, args.batch_size, True),
        ("sts_dev", sts_dev_data, args.batch_size, False),
    ]

    # create dataloaders
    dataloaders = {}
    for name, data, batch_size, should_shuffle in datasets:
        dataloaders[name] = create_data_loader(
            data, batch_size, data.collate_fn, use_multi_gpu
        )

    return (
        dataloaders["para_test"],
        dataloaders["sst_test"],
        dataloaders["sts_test"],
        dataloaders["para_dev"],
        dataloaders["sst_dev"],
        dataloaders["sts_dev"],
    )


def create_data_loader(
    data, batch_size, collate_fn, use_multi_gpu, world_size=None, rank=None, shuffle=False
):
    # send a distributed data sampler if using GPU otherwise, just return a data loader with shuffle
    if use_multi_gpu:
        sampler = DistributedSampler(
            data, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        return DataLoader(
            data, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler
        )
    else:
        return DataLoader(
            data, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
        )
