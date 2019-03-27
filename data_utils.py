import configs
import os

from tqdm import tqdm
from model_utils import *
from collections import defaultdict, Counter
from itertools import chain
import time
import json
# import Levenshtein
import csv
from vocab import WordEmbedder, CharVocab
import bisect
# from PIL import Image
import pdb
import h5py

char_vocab = CharVocab()
glove_embedder = WordEmbedder(configs.glove_embeddings_path, configs.glove_embedding_dim)
head_embedder = WordEmbedder(configs.head_embeddings_path, configs.raw_head_embedding_dim)

id_to_genre = ('bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb')
genre_to_id = {genre: id_ for id_, genre in enumerate(id_to_genre)}
genre_num = len(id_to_genre)

names = ('train', 'dev') if configs.training else ('test',)


def build_mask_batch(
        # [batch_size], []
        len_batch, max_len
):
    batch_size, = len_batch.shape
    # [batch_size, max_len]
    idxes_batch = np.arange(max_len).reshape(1, -1).repeat(batch_size, axis=0)
    # [batch_size, max_len] = [batch_size, max_len] >= [batch_size, 1]
    return idxes_batch < len_batch.reshape(-1, 1)


class Dataset(tud.Dataset):
    def __init__(self, name):
        self.name = name
        self.examples = json.load(open(f'{self.name}.json'))
        self.elmo_cache = h5py.File(f'{self.name}.elmo.cache.hdf5', 'r')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.tensorize_example(idx)

    def get_gold_clusters(self, example_idx):
        return self.examples[example_idx]['clusters']

    def tensorize_example(self, example_idx):
        example = self.examples[example_idx]
        clusters = example['clusters']

        gold_spans = sorted(
            tuple(span) for cluster in clusters
            for span in cluster
        )

        gold_span_to_id = {
            span: id_
            for id_, span in enumerate(gold_spans)
        }

        gold_start_idxes, gold_end_idxes = map(
            np.array,
            zip(*gold_spans) if gold_spans else ([], [])
        )

        gold_cluster_ids = np.zeros(len(gold_spans))

        for cluster_id, cluster in enumerate(clusters):
            for span in cluster:
                # leave cluster_id of 0 for dummy
                gold_cluster_ids[gold_span_to_id[tuple(span)]] = cluster_id + 1

        sents = example['sentences']
        speakers = [
            speaker
            for speakers_of_sent in example['speakers']
            for speaker in speakers_of_sent
        ]

        sent_len_batch = np.array([len(sent) for sent in sents])
        max_sent_len = sent_len_batch.max()
        doc_len = sent_len_batch.sum()
        sent_num = len(sents)

        max_word_len = max(
            max(
                max(len(word) for word in sent)
                for sent in sents
            ),
            max(configs.cnn_kernel_widths)
        )

        char_ids_seq_batch = np.zeros((sent_num, max_sent_len, max_word_len))

        doc_key = example['doc_key'].replace('/', ':')
        doc_cache = self.elmo_cache[doc_key]
        elmo_layer_outputs_batch = np.zeros(
            (sent_num, max_sent_len, configs.elmo_embedding_dim, configs.elmo_layer_num),
            dtype=np.float32
        )

        for i, sent in enumerate(sents):
            for j, word in enumerate(sent):
                # sent_batch[i][j] = word
                char_ids_seq_batch[i, j, :len(word)] = [char_vocab[char] for char in word]

            elmo_layer_outputs_batch[i, :sent_len_batch[i], ...] = doc_cache[str(i)][...]

        glove_embedding_seq_batch = np.zeros([sent_num, max_sent_len, glove_embedder.dim])
        head_embedding_seq_batch = np.zeros([sent_num, max_sent_len, head_embedder.dim])

        for i, sent in enumerate(sents):
            for j, word in enumerate(sent):
                glove_embedding_seq_batch[i, j] = glove_embedder[word]
                head_embedding_seq_batch[i, j] = head_embedder[word]

        speaker_to_id = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = np.array([speaker_to_id[s] for s in speakers])

        genre_id = [genre_to_id[doc_key[:2]]]

        # [doc_len]
        sent_idxes = np.array(
            [
                i
                for i in range(sent_num)
                for _ in range(sent_len_batch[i])
            ]
        )

        # [doc_len, max_span_width]
        cand_start_idxes = np.arange(doc_len).reshape(-1, 1).repeat(configs.max_span_width, axis=-1)

        # [doc_len, max_span_width]
        cand_end_idxes = np.minimum(cand_start_idxes + np.arange(configs.max_span_width).reshape(1, -1), doc_len - 1)
        # [doc_len * max_span_width]
        cand_start_idxes = cand_start_idxes.reshape(-1)
        cand_end_idxes = cand_end_idxes.reshape(-1)

        # [doc_len * max_span_width]
        cand_start_sent_idxes = sent_idxes[cand_start_idxes]
        cand_end_sent_idxes = sent_idxes[cand_end_idxes]
        # [doc_len * max_span_width]
        cand_mask = (cand_start_sent_idxes == cand_end_sent_idxes)

        # [doc_len * max_span_width]
        cand_start_idxes = cand_start_idxes[cand_mask]
        cand_end_idxes = cand_end_idxes[cand_mask]
        cand_sent_idxes = cand_start_sent_idxes[cand_mask]

        # [doc_len * max_span_width]
        cand_cluster_ids = self.get_cand_labels(
            cand_start_idxes, cand_end_idxes,
            gold_start_idxes, gold_end_idxes, gold_cluster_ids
        )

        example_tensors = (
            example_idx,
            glove_embedding_seq_batch, head_embedding_seq_batch, elmo_layer_outputs_batch,
            char_ids_seq_batch, sent_len_batch, speaker_ids, genre_id,
            gold_start_idxes, gold_end_idxes, gold_cluster_ids,
            cand_start_idxes, cand_end_idxes, cand_cluster_ids, cand_sent_idxes,
        )

        if self.name == 'train' and sent_num > configs.max_sent_num:
            return self.truncate_example(*example_tensors)
        else:
            return example_tensors

    def get_cand_labels(self, cand_start_idxes, cand_end_idxes, gold_start_idxes, gold_end_idxes, gold_cluster_ids):
        # [gold_num, cand_num]
        indicator_mat = (gold_start_idxes.reshape(-1, 1) == cand_start_idxes.reshape(1, -1)) \
                        & (gold_end_idxes.reshape(-1, 1) == cand_end_idxes.reshape(1, -1))

        # [1, cand_num] = [1, gold_num] @ [gold_num, cand_num]
        cand_labels = gold_cluster_ids.reshape(1, -1) @ indicator_mat.astype(np.int32)

        # [cand_num]
        return cand_labels.reshape(-1)

    def truncate_example(
            self,
            example_idx,
            glove_embedding_seq_batch, head_embedding_seq_batch,
            elmo_layer_outputs_batch, char_ids_seq_batch, sent_len_batch, speaker_ids, genre,
            gold_start_idxes, gold_end_idxes, gold_cluster_ids,
            cand_start_idxes, cand_end_idxes, cand_cluster_ids, cand_sent_idxes,
    ):
        sent_num = glove_embedding_seq_batch.shape[0]
        assert sent_num > configs.max_sent_num

        start_sent_idx = random.randint(0, sent_num - configs.max_sent_num)
        end_sent_idx = start_sent_idx + configs.max_sent_num

        start_word_idx = sent_len_batch[:start_sent_idx].sum()
        end_word_idx = sent_len_batch[:end_sent_idx].sum()

        glove_embedding_seq_batch = glove_embedding_seq_batch[start_sent_idx:end_sent_idx, ...]
        head_embedding_seq_batch = head_embedding_seq_batch[start_sent_idx:end_sent_idx, ...]
        elmo_layer_outputs_batch = elmo_layer_outputs_batch[start_sent_idx:end_sent_idx, ...]
        char_ids_seq_batch = char_ids_seq_batch[start_sent_idx:end_sent_idx, ...]
        sent_len_batch = sent_len_batch[start_sent_idx:end_sent_idx]

        speaker_ids = speaker_ids[start_word_idx:end_word_idx]

        gold_mask = (gold_end_idxes >= start_word_idx) & (gold_start_idxes < end_word_idx)

        gold_start_idxes = gold_start_idxes[gold_mask] - start_word_idx
        gold_end_idxes = gold_end_idxes[gold_mask] - start_word_idx
        gold_cluster_ids = gold_cluster_ids[gold_mask]

        cand_mask = (cand_end_idxes >= start_word_idx) & (cand_start_idxes < end_word_idx)

        cand_start_idxes = cand_start_idxes[cand_mask] - start_word_idx
        cand_end_idxes = cand_end_idxes[cand_mask] - start_word_idx
        cand_cluster_ids = cand_cluster_ids[cand_mask]
        cand_sent_idxes = cand_sent_idxes[cand_mask] - start_sent_idx

        return (
            example_idx,
            glove_embedding_seq_batch, head_embedding_seq_batch,
            elmo_layer_outputs_batch, char_ids_seq_batch, sent_len_batch, speaker_ids,
            genre, gold_start_idxes, gold_end_idxes, gold_cluster_ids,
            cand_start_idxes,
            cand_end_idxes, cand_cluster_ids, cand_sent_idxes,
        )


datasets = {
    name: Dataset(name)
    for name in names
}


def get_dataset_size(name):
    return len(datasets[name])

def get_gold_clusters(name, example_idx):
    return datasets[name].get_gold_clusters(example_idx)


def collate(batch):
    # batch_size = 1
    tensors, = batch
    example_idx, *tensors = tensors
    return (
        example_idx,
        tuple(
            map(lambda tensor: torch.as_tensor(tensor).cuda(), tensors)
        )
    )


data_loaders = {
    name: tud.DataLoader(
        dataset=datasets[name],
        batch_size=1,
        shuffle=(name == 'train'),
        # pin_memory=True,
        collate_fn=collate,
        num_workers=4
    )
    for name in names
}


def gen_batches(name):
    instance_num = 0

    for example_idx, batch in data_loaders[name]:
        instance_num += len(batch[-1])
        pct = instance_num * 100. / len(datasets[name])
        yield pct, example_idx, batch


def save_predictions(name, predictions):
    # if name == 'valid':
    #     results = []
    #
    #     for prediction, word_ids, label in zip(predictions, datasets['valid'].texts, datasets['valid'].labels):
    #         results.append(
    #             {
    #                 'text': vocab.textify(word_ids),
    #                 'correct': bool(int(prediction) == label),
    #                 'label': id_to_class[label],
    #                 'prediction':  id_to_class[prediction]
    #             }
    #         )
    #
    #     json.dump(results, open('results.json', 'w'), indent=4)
    # else:
    #     np.save('predictions.npy', predictions)
    with open(f'predictions.{name}', 'w') as predictions_file:
        predictions_file.writelines(
            '\n'.join(map(lambda i: str(i.cpu().item()), predictions))
        )
