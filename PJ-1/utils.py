import torch


def custom_collate(batch):
    """ padding """

    DEFAULT_PADDING_LABEL = 0
    input_ids, targets = [], []
    for x in batch:
        input_ids.append(x['input_ids'])
        targets.append(x['target'])
    max_len = max(len(x) for x in input_ids)
    batch_input_ids = [x + [DEFAULT_PADDING_LABEL] \
                       * (max_len - len(x)) for x in input_ids]
    batch_input_ids = torch.LongTensor(batch_input_ids)
    batch_targets = torch.LongTensor(targets)

    return batch_input_ids, batch_targets
