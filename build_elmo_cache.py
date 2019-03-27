import numpy as np
import h5py
import json
from modules import ElmoEmbedder

elmo_embedder = ElmoEmbedder()

for name in ('train', 'dev', 'test'):
    with open(f'{name}.json') as examples_file, \
            h5py.File(f'{name}.elmo.cache.hdf5', 'w') as cache_file:
        for example in json.load(examples_file):
            doc_cache = cache_file.create_group(example['doc_key'].replace('/', ':'))
            sents = example['sentences']
            sent_lens = np.array([len(sent) for sent in sents])
            # [batch_size, max_sent_len, embedding_dim, layer_num]
            layer_outputs_batch, _ = elmo_embedder.embed(sents)
            layer_outputs_batch = layer_outputs_batch.cpu().numpy()

            for i, sent_len in enumerate(sent_lens):
                doc_cache[str(i)] = layer_outputs_batch[i][:sent_len, ...]
