import os

def read_label_data(data_path):
    data_pairs = []
    with open(data_path, 'r') as f:
        is_header = True
        while True:
            line = f.readline()
            if not line:
                break
            if is_header:
                is_header = False
                continue
            line_split = line.split('\t')
            sentence, label = line_split[0].strip(), int(line_split[-1].strip())
            data_pairs.append((sentence, label))
        f.close()
    return data_pairs

def read_unlabel_data(data_path):
    data_pairs = []
    with open(data_path, 'r') as f:
        is_header = True
        while True:
            line = f.readline()
            if not line:
                break
            if is_header:
                is_header = False
                continue
            line_split = line.split('\t')
            index, sentence = line_split[0], line_split[-1].strip('\n')
            data_pairs.append((index, sentence))
        f.close()
    return data_pairs

def generate_dialogue(data_root_dir, data_split):
    traindata_path = os.path.join(data_root_dir, data_split['train'])
    valdata_path = os.path.join(data_root_dir, data_split['validation'])
    testdata_path = os.path.join(data_root_dir, data_split['test'])
    traindata = read_label_data(traindata_path)
    valdata = read_label_data(valdata_path)
    testdata = read_unlabel_data(testdata_path)
    dataset = {'train': [], 'validation': [], 'test': []}
    classes = ['negative', 'positive']
    def get_dialogue(data, is_test=False):
        dialogue_list = []
        if not is_test:
            for i, (sentence, label) in enumerate(data):
                dialogue = dict()
                dialogue['premise'] = sentence
                dialogue['hypothesis'] = 'it was {mask}'.format(mask=classes[label])
                dialogue['idx'] = i
                dialogue['label'] = label
                dialogue_list.append(dialogue)
        else:
            for index, sentence in data:
                dialogue = dict()
                dialogue['premise'] = sentence
                dialogue['hypothesis'] = 'it was unknown'
                dialogue['idx'] = index
                dialogue['label'] = -1
                dialogue_list.append(dialogue)
        return dialogue_list
    dataset['train'] = get_dialogue(traindata)
    dataset['validation'] = get_dialogue(valdata)
    dataset['test'] = get_dialogue(testdata, is_test=True)
    return dataset
    



if __name__ == '__main__':
    train_data_path = 'dataset/FSS/FewShotSST/dev.tsv'
    # print(read_label_data(train_data_path))
    test_data_path = 'dataset/FSS/FewShotSST/test.tsv'
    # print(read_unlabel_data(test_data_path))
    data_root_dir = 'dataset/FSS/FewShotSST/'
    split = {'train': 'train_32.tsv', 'validation': 'train_64.tsv', 'test': 'train_128.tsv'}
    print(generate_dialogue(data_root_dir, split))
