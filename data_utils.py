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

glove_embedder = WordEmbedder(configs.glove_embeddings_path, configs.glove_embedding_dim) \
    if configs.uses_glove_embeddings else None

head_embedder = WordEmbedder(configs.head_embeddings_path, configs.raw_head_embedding_dim)

id_to_genre = ('bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb')
genre_to_id = {genre: id_ for id_, genre in enumerate(id_to_genre)}
genre_num = len(id_to_genre)

names = ('train', 'dev') if configs.training \
    else ('test',) if configs.testing \
    else ('test', 'dev')


# names = 'test', 'dev'


# def build_mask_batch(
#         # [batch_size], []
#         len_batch, max_len
# ):
#     batch_size, = len_batch.shape
#     # [batch_size, max_len]
#     idxes_batch = np.arange(max_len).reshape(1, -1).repeat(batch_size, axis=0)
#     # [batch_size, max_len] = [batch_size, max_len] >= [batch_size, 1]
#     return idxes_batch < len_batch.reshape(-1, 1)


class Dataset(tud.Dataset):
    def __init__(self, name):
        self.name = name
        self.examples = json.load(open(f'{configs.data_dir}/{self.name}.json'))
        self.elmo_cache = h5py.File(f'{configs.data_dir}/{self.name}.elmo.cache.hdf5', 'r', swmr=True)

    def __len__(self):
        return len(self.examples)

    def get_gold_clusters(self, example_idx):
        return self.examples[example_idx]['clusters']

    def truncate_example(self, example):
        sents = example['sentences']
        sent_num = len(sents)
        sent_lens = [len(sent) for sent in sents]
        start_sent_idx = random.randint(0, sent_num - configs.max_sent_num)
        end_sent_idx = start_sent_idx + configs.max_sent_num
        start_word_idx = sum(sent_lens[:start_sent_idx])
        end_word_idx = sum(sent_lens[:end_sent_idx])

        clusters = [
            [
                (l - start_word_idx, r - start_word_idx)
                for l, r in cluster
                if start_word_idx <= l <= r < end_word_idx
            ] for cluster in example['clusters']
        ]
        clusters = [cluster for cluster in clusters if cluster]

        return {
            'sentences': example['sentences'][start_sent_idx:end_sent_idx],
            'clusters': clusters,
            'speakers': example['speakers'][start_sent_idx:end_sent_idx],
            'doc_key': example['doc_key'],
            'start_sent_idx': start_sent_idx
        }

    def __getitem__(self, example_idx):
        start_time = time.time()

        example = self.examples[example_idx]

        if self.name == 'train' and len(example['sentences']) > configs.max_sent_num:
            example = self.truncate_example(example)

        clusters = example['clusters']

        gold_spans = sorted(
            tuple(span) for cluster in clusters
            for span in cluster
        )

        # for s, e in gold_spans:
        #     try:
        #         assert isinstance(s, int)
        #         assert isinstance(e, int)
        #     except:
        #         print(gold_spans)
        #         exit()

        gold_span_to_id = {
            span: id_
            for id_, span in enumerate(gold_spans)
        }

        gold_start_idxes, gold_end_idxes = map(
            # np.array,
            torch.as_tensor,
            zip(*gold_spans) if gold_spans else ([], [])
        )

        # if gold_start_idxes.dtype != torch.long:
        #     print(gold_spans)
        #     exit()

        gold_cluster_ids = torch.zeros(len(gold_spans), dtype=torch.long)

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

        sent_len_batch = torch.as_tensor([len(sent) for sent in sents])

        # print(self.name)
        # print(example_idx)
        # print(sent_len_batch)

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

        char_ids_seq_batch = torch.zeros((sent_num, max_sent_len, max_word_len), dtype=torch.long)

        doc_key = example['doc_key'].replace('/', ':')

        # try:
        doc_cache = self.elmo_cache[doc_key]
        elmo_layer_outputs_batch = torch.zeros(
            (sent_num, max_sent_len, configs.elmo_embedding_dim, configs.elmo_layer_num),
            dtype=torch.float32
        )

        for i, sent in enumerate(sents):
            for j, word in enumerate(sent):
                # sent_batch[i][j] = word
                char_ids_seq_batch[i, j, :len(word)] = torch.as_tensor([char_vocab[char] for char in word])

            sent_key = str(i + example.get('start_sent_idx', 0))
            elmo_layer_outputs_batch[i, :sent_len_batch[i]] = torch.as_tensor(doc_cache[sent_key][...])

        # except:
        #     print(example_idx)
        #     print(doc_key)
        #     # pdb.set_trace()
        #     breakpoint()

        # assert elmo_layer_outputs_batch.dtype == np.float32
        glove_embedding_seq_batch = torch.zeros((sent_num, max_sent_len, glove_embedder.dim), dtype=torch.float32) \
            if configs.uses_glove_embeddings else None

        head_embedding_seq_batch = torch.zeros((sent_num, max_sent_len, head_embedder.dim), dtype=torch.float32)

        for i, sent in enumerate(sents):
            for j, word in enumerate(sent):
                if configs.uses_glove_embeddings:
                    glove_embedding_seq_batch[i, j] = glove_embedder[word]

                head_embedding_seq_batch[i, j] = head_embedder[word]

        speaker_to_id = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = torch.as_tensor([speaker_to_id[s] for s in speakers])

        genre_id = torch.as_tensor([genre_to_id[doc_key[:2]]])

        # [doc_len]
        sent_idxes = torch.as_tensor(
            [
                i
                for i in range(sent_num)
                for _ in range(sent_len_batch[i])
            ]
        )

        # [doc_len, max_span_width]
        cand_start_idxes = torch.arange(doc_len).view(-1, 1).repeat(1, configs.max_span_width)
        # [doc_len, max_span_width]
        cand_cluster_ids = torch.zeros_like(cand_start_idxes)

        if gold_spans:
            # try:
            gold_end_offsets = gold_end_idxes - gold_start_idxes
            gold_span_mask = gold_end_offsets < configs.max_span_width
            filtered_gold_start_idxes = gold_start_idxes[gold_span_mask]
            filtered_gold_end_offsets = gold_end_offsets[gold_span_mask]
            filtered_gold_cluster_ids = gold_cluster_ids[gold_span_mask]
            cand_cluster_ids[filtered_gold_start_idxes, filtered_gold_end_offsets] = filtered_gold_cluster_ids
            # except:
            #     breakpoint()

        # [doc_len * max_span_width]
        cand_end_idxes = (cand_start_idxes + torch.arange(configs.max_span_width).view(1, -1)).view(-1)

        # # [doc_len * max_span_width]
        # cand_end_idxes = torch.clamp(
        #     cand_start_idxes + torch.arange(configs.max_span_width).view(1, -1), max=(doc_len - 1)
        # ).view(-1)

        # [doc_len * max_span_width]
        cand_start_idxes = cand_start_idxes.view(-1)
        # [doc_len * max_span_width]
        cand_cluster_ids = cand_cluster_ids.view(-1)
        # [doc_len * max_span_width]
        cand_mask = cand_end_idxes < doc_len
        # [cand_num]
        cand_start_idxes = cand_start_idxes[cand_mask]
        # [cand_num]
        cand_end_idxes = cand_end_idxes[cand_mask]
        # [cand_num]
        cand_cluster_ids = cand_cluster_ids[cand_mask]
        # [cand_num]
        cand_start_sent_idxes = sent_idxes[cand_start_idxes]
        # [cand_num]
        cand_end_sent_idxes = sent_idxes[cand_end_idxes]
        # [cand_num]
        cand_mask = (cand_start_sent_idxes == cand_end_sent_idxes)
        # [cand_num]
        cand_start_idxes = cand_start_idxes[cand_mask]
        # [cand_num]
        cand_end_idxes = cand_end_idxes[cand_mask]
        # [cand_num]
        cand_sent_idxes = cand_start_sent_idxes[cand_mask]
        # [cand_num]
        cand_cluster_ids = cand_cluster_ids[cand_mask]

        # try:
        #     assert (cand_cluster_ids == (self.get_cand_labels(
        #         cand_start_idxes, cand_end_idxes,
        #         gold_start_idxes, gold_end_idxes, gold_cluster_ids
        #     ) if gold_spans else torch.zeros_like(cand_start_idxes, dtype=torch.long))).all()
        # except:
        #     breakpoint()

        # # [cand_num]
        # cand_cluster_ids = self.get_cand_labels(
        #     cand_start_idxes, cand_end_idxes,
        #     gold_start_idxes, gold_end_idxes, gold_cluster_ids
        # ) if gold_spans else torch.zeros_like(cand_start_idxes, dtype=torch.long)

        # example_tensors = (
        #     example_idx,
        #     glove_embedding_seq_batch, head_embedding_seq_batch, elmo_layer_outputs_batch,
        #     char_ids_seq_batch, sent_len_batch, speaker_ids, genre_id,
        #     gold_start_idxes, gold_end_idxes, gold_cluster_ids,
        #     cand_start_idxes, cand_end_idxes, cand_cluster_ids, cand_sent_idxes,
        # )

        if self.name == 'train':  # and sent_num > configs.max_sent_num:
            assert sent_num <= configs.max_sent_num
            # example_tensors = self.truncate(*example_tensors)

        return self.tensorize(
            example_idx,
            glove_embedding_seq_batch, head_embedding_seq_batch, elmo_layer_outputs_batch,
            char_ids_seq_batch, sent_len_batch, speaker_ids, genre_id,
            gold_start_idxes, gold_end_idxes, gold_cluster_ids,
            cand_start_idxes, cand_end_idxes, cand_cluster_ids, cand_sent_idxes
        )

        # print(f'__getitem__: {time.time() - start_time}')

        # return example_tensors

    # def get_cand_labels(self, cand_start_idxes, cand_end_idxes, gold_start_idxes, gold_end_idxes, gold_cluster_ids):
    #     start_time = time.time()
    #
    #     # [gold_num, cand_num]
    #     indicator_mat = (gold_start_idxes.reshape(-1, 1) == cand_start_idxes.reshape(1, -1)) \
    #                     & (gold_end_idxes.reshape(-1, 1) == cand_end_idxes.reshape(1, -1))
    #
    #     # [1, cand_num] = [1, gold_num] @ [gold_num, cand_num]
    #     cand_labels = gold_cluster_ids.reshape(1, -1) @ indicator_mat.to(torch.long)
    #
    #     # print(f'get_cand_labels: {time.time() - start_time}')
    #
    #     # [cand_num]
    #     return cand_labels.reshape(-1)

    # def truncate(
    #     self,
    #     example_idx,
    #     glove_embedding_seq_batch, head_embedding_seq_batch,
    #     elmo_layer_outputs_batch, char_ids_seq_batch, sent_len_batch, speaker_ids, genre_id,
    #     gold_start_idxes, gold_end_idxes, gold_cluster_ids,
    #     cand_start_idxes, cand_end_idxes, cand_cluster_ids, cand_sent_idxes,
    # ):
    #     sent_num = len(sent_len_batch)
    #     assert sent_num > configs.max_sent_num
    #
    #     start_sent_idx = random.randint(0, sent_num - configs.max_sent_num)
    #     end_sent_idx = start_sent_idx + configs.max_sent_num
    #
    #     start_word_idx = sent_len_batch[:start_sent_idx].sum()
    #     end_word_idx = sent_len_batch[:end_sent_idx].sum()
    #
    #     sent_len_batch = sent_len_batch[start_sent_idx:end_sent_idx]
    #     max_sent_len = sent_len_batch.max()
    #
    #     if configs.uses_glove_embeddings:
    #         glove_embedding_seq_batch = glove_embedding_seq_batch[start_sent_idx:end_sent_idx, :max_sent_len]
    #
    #     head_embedding_seq_batch = head_embedding_seq_batch[start_sent_idx:end_sent_idx, :max_sent_len]
    #     elmo_layer_outputs_batch = elmo_layer_outputs_batch[start_sent_idx:end_sent_idx, :max_sent_len]
    #     char_ids_seq_batch = char_ids_seq_batch[start_sent_idx:end_sent_idx, :max_sent_len]
    #
    #     speaker_ids = speaker_ids[start_word_idx:end_word_idx]
    #
    #     gold_mask = (gold_end_idxes >= start_word_idx) & (gold_start_idxes < end_word_idx)
    #
    #     gold_start_idxes = gold_start_idxes[gold_mask] - start_word_idx
    #     gold_end_idxes = gold_end_idxes[gold_mask] - start_word_idx
    #     gold_cluster_ids = gold_cluster_ids[gold_mask]
    #
    #     cand_mask = (cand_end_idxes >= start_word_idx) & (cand_start_idxes < end_word_idx)
    #
    #     cand_start_idxes = cand_start_idxes[cand_mask] - start_word_idx
    #     cand_end_idxes = cand_end_idxes[cand_mask] - start_word_idx
    #     cand_cluster_ids = cand_cluster_ids[cand_mask]
    #     cand_sent_idxes = cand_sent_idxes[cand_mask] - start_sent_idx
    #
    #     # if cand_start_idxes.max() < 0:
    #     #     breakpoint()
    #
    #     # breakpoint()
    #
    #     return (
    #         example_idx,
    #         glove_embedding_seq_batch, head_embedding_seq_batch,
    #         elmo_layer_outputs_batch, char_ids_seq_batch, sent_len_batch, speaker_ids,
    #         genre_id, gold_start_idxes, gold_end_idxes, gold_cluster_ids,
    #         cand_start_idxes,
    #         cand_end_idxes, cand_cluster_ids, cand_sent_idxes,
    #     )

    def tensorize(
        self,
        example_idx,
        glove_embedding_seq_batch, head_embedding_seq_batch,
        elmo_layer_outputs_batch, char_ids_seq_batch, sent_len_batch, speaker_ids, genre_id,
        gold_start_idxes, gold_end_idxes, gold_cluster_ids,
        cand_start_idxes, cand_end_idxes, cand_cluster_ids, cand_sent_idxes,
    ):
        # print('tensorizing')
        # print(char_ids_seq_batch.shape)
        return (
            example_idx,
            # (
            # [sent_num, max_sent_len, glove_embedding_dim]
            glove_embedding_seq_batch.cuda() if configs.uses_glove_embeddings else torch.tensor(0),
            # [sent_num, max_sent_len, raw_head_embedding_dim]
            head_embedding_seq_batch.cuda(),
            # [sent_num, max_sent_len, elmo_embedding_dim, elmo_layer_num]
            elmo_layer_outputs_batch.cuda(),
            # [sent_num, max_sent_len, max_word_len]
            char_ids_seq_batch.cuda(),
            # [sent_num]
            sent_len_batch,
            # [doc_len]
            speaker_ids,
            # [1]
            genre_id,
            # [gold_num]
            gold_start_idxes,
            # [gold_num]
            gold_end_idxes,
            # [gold_num]
            gold_cluster_ids,
            # [cand_num]
            cand_start_idxes,
            # [cand_num]
            cand_end_idxes,
            # [cand_num]
            cand_cluster_ids,
            # [cand_num]
            cand_sent_idxes
            # )
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
    # batch, = batch
    # breakpoint()

    # return batch
    (example_idx, *tensors), = batch
    # breakpoint()

    # assert tensors[2].dtype == np.float32

    # print(torch.as_tensor(tensors[2]).cuda().type())

    return (
        example_idx,
        tensors
        # tuple(
        #     # map(
        #     #     lambda tensor: torch.as_tensor(tensor).cuda(),
        #     #     # lambda tensor: tensor.cuda(),
        #     #     tensors
        #     # )
        # )
    )

data_loaders = {
    name: tud.DataLoader(
        dataset=datasets[name],
        batch_size=1,
        shuffle=(name == 'train'),
        # pin_memory=True,
        collate_fn=collate,
        # num_workers=4
    )
    for name in names
}

def gen_batches(name):
    instance_num = 0

    for example_idx, tensors in data_loaders[name]:
        # ((example_idx, tensors),) = b
        instance_num += 1
        pct = instance_num * 100. / len(datasets[name])
        yield pct, example_idx, tensors
        # torch.cuda.empty_cache()

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
