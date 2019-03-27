import configs
from vocab import Vocab
from collections import defaultdict
import numpy as np
from itertools import chain

names = ('train', 'valid', 'test')
raw_datasets = defaultdict(list)
id_to_class = set()

for name in names:
    with open(f'{configs.data_dir}/topicclass_{name}.txt') as dataset_file:
        for line in dataset_file.readlines():
            class_, words = line.lower().strip().split(' ||| ')
            # class_, words = line.strip().split(' ||| ')
            raw_datasets[name].append((class_, words.split()))
            id_to_class.add(class_)

id_to_class.remove('unk')
id_to_class = list(id_to_class)
class_to_id = {
    class_: id_
    for id_, class_ in enumerate(id_to_class)
}
class_to_id['unk'] = -1

with open(configs.classes_path, 'w') as classes_file:
    classes_file.writelines('\n'.join(id_to_class))


def get_words():
    # for _, words in chain(raw_datasets['train'], raw_datasets['valid']):
    for _, words in raw_datasets['train']:
        yield from words
    # for raw_dataset in raw_datasets.values():
    #     for _, words in raw_dataset:
    #         yield from words


vocab = Vocab.build(get_words())
vocab.build_embedding_mat(new=True)

for name in names:
    texts, labels = [], []

    for class_, words in raw_datasets[name]:
        texts.append(vocab.idify(words))
        labels.append(class_to_id[class_])

    np.save(f'{configs.data_dir}/labels.{name}.npy', labels)
    np.save(f'{configs.data_dir}/texts.id.{name}.npy', texts)


