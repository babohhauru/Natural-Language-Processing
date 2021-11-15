from torch.utils.data import Dataset


class SSTDataSet(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                raw_words, target = line.strip().split('\t')
                # simple preprocessing
                if raw_words.endswith("\""):
                    raw_words = raw_words[:-1]
                if raw_words.startswith('"'):
                    raw_words = raw_words[1:]
                raw_words = raw_words.replace('""', '"')
                raw_words = raw_words.split(' ')
                self.data.append({
                    'raw_words': raw_words,
                    'target': int(target),
                })
        print("# samples: {}".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def convert_word_to_ids(self, vocab):
        for i in range(len(self.data)):
            ins = self.data[i]
            word_ids =  [vocab[x] for x in ins['raw_words']]
            self.data[i]['input_ids'] = word_ids