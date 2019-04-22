import configs
import os
# import numpy as np
import json
from collections import Counter, defaultdict
from itertools import chain
from model_utils import *
import time


class Vocab:
    path = f'{configs.data_dir}/vocab.txt'
    embedding_mat_path = f'{configs.data_dir}/embedding_mat.fasttext-wiki-word.npy'

    @staticmethod
    def build(words):
        word_to_id, id_to_word = {}, []

        # for word in ('<pad>', '<s>', '</s>', '<unk>'):
        for word in ('<pad>', '<unk>'):
            word_to_id[word] = len(id_to_word)
            id_to_word.append(word)

        word_cnts = Counter(words)

        # with open(f'{configs.data_dir}/word_cnts.json', 'w') as word_cnts_file:
        #     json.dump(word_cnts, word_cnts_file)

        for word, cnt in word_cnts.items():
            if cnt > 5:
                word_to_id[word] = len(id_to_word)
                id_to_word.append(word)

        with open(Vocab.path, 'w') as vocab_file:
            vocab_file.writelines('\n'.join(id_to_word))

        return Vocab()

    def __init__(self):
        assert os.path.exists(Vocab.path)

        self.word_to_id, self.id_to_word = defaultdict(lambda: self.unk_id), []

        with open(Vocab.path) as vocab_file:
            for word in map(lambda s: s.strip(), vocab_file.readlines()):
                self.word_to_id[word] = len(self.id_to_word)
                self.id_to_word.append(word)

        self.padding_id = self.word_to_id['<pad>']
        # self.start_id = self.word_to_id['<s>']
        # self.end_id = self.word_to_id['</s>']
        self.unk_id = self.word_to_id['<unk>']
        self.size = len(self.id_to_word)
        assert len(self.word_to_id) == self.size

    def build_embedding_mat(self, new=False):
        if new or not os.path.exists(Vocab.embedding_mat_path):
            embedding_mat = np.random.randn(self.size, configs.word_embedding_dim).astype(np.float32)
            embedding_mat[self.padding_id].fill(0.)
            assert len(self.word_to_id) == self.size

            print(embedding_mat.shape)

            with open(configs.word_embeddings_path) as word_embeddings_file:
                line = word_embeddings_file.readline()
                embedding_num, embedding_dim = map(int, line.strip().split())
                line_cnt = 0
                overlap_cnt = 0

                assert embedding_dim == configs.word_embedding_dim

                words = set()

                for line in word_embeddings_file.readlines():
                    word, *embedding = line.split()
                    line_cnt += 1
                    assert len(embedding) == embedding_dim
                    words.add(word)

                    if word in self.word_to_id:
                        overlap_cnt += 1
                        assert self.word_to_id[word] is not self.unk_id
                        embedding = np.array(
                            list(map(float, embedding))
                        )

                        np.copyto(
                            dst=embedding_mat[self.word_to_id[word]],
                            src=list(map(float, embedding))
                        )
                        assert np.allclose(embedding_mat[self.word_to_id[word]], np.array(list(map(float, embedding))))

                assert len(words) == embedding_num
                print(len(words & set(self.id_to_word)))
                print(sum(word in self.word_to_id for word in words))
                assert line_cnt == embedding_num

            np.save(Vocab.embedding_mat_path, embedding_mat)

            print(f'built embedding matrix from {configs.word_embeddings_path} '
                  f'with {overlap_cnt} overlaps / {self.size}')
        else:
            embedding_mat = np.load(Vocab.embedding_mat_path)
            print(f'loaded embedding matrix from {configs.word_embeddings_path}')

        return embedding_mat

    # def __contains__(self, word):
    #     return word in self.word_to_id

    def __getitem__(self, word_or_id):
        if isinstance(word_or_id, str):
            return self.word_to_id[word_or_id]
        else:
            return self.id_to_word[word_or_id]

    def textify(self, ids):
        return ' '.join(map(self.id_to_word.__getitem__, ids))

    def idify(self, words):
        # return list(map(self.word_to_id.__getitem__, words))
        return list(map(lambda word: self.word_to_id[word.lower()], words))


class CharVocab:
    path = f'char_vocab.txt'

    @staticmethod
    def build(words):
        ...
        return CharVocab()

    def __init__(self):
        assert os.path.exists(CharVocab.path)
        self.id_to_char = ['<pad>', '<unk>']
        self.padding_id, self.unk_id = range(len(self.id_to_char))

        with open(CharVocab.path) as vocab_file:
            self.id_to_char.extend(map(lambda s: s.strip(), vocab_file.readlines()))

        self.char_to_id = defaultdict(
            lambda: self.unk_id,
            {char: id_ for id_, char in enumerate(self.id_to_char)}
        )

        self.size = len(self.id_to_char)

        assert len(self.char_to_id) == self.size

    def __getitem__(self, char_or_id):
        if isinstance(char_or_id, str):
            return self.char_to_id[char_or_id]
        else:
            return self.id_to_char[char_or_id]


class WordEmbedder:
    def __init__(self, path, dim, normalizes=True):
        self.normalizes = normalizes
        self.path = path
        self.dim = dim
        self.embeddings = self.load_embeddings()
        self.size = len(self.embeddings)

    def __len__(self):
        return self.size

    def load_embeddings(self):
        default_embedding = torch.zeros(self.dim)
        embeddings = defaultdict(lambda: default_embedding)

        # vocab_size = None

        if configs.debugging or configs.testing_gpu:
            return embeddings

        start_time = time.time()

        with open(self.path) as embeddings_file:
            for i, line in enumerate(embeddings_file.readlines()):
                word, *embedding = line.split(' ')
                embedding = torch.as_tensor(
                    list(map(float, embedding))
                )
                # word_end = line.find(" ")
                # word = line[:word_end]
                # embedding = np.fromstring(line[word_end + 1:], np.float3232, sep=" ")
                # assert len(embedding) == self.size
                embeddings[word] = embedding

        print(f'loaded embeddings from {self.path} in {time.time() - start_time:.5f}s')

        # if vocab_size is not None:
        #     assert vocab_size == len(embeddings)

        return embeddings

    def __getitem__(self, word):
        embedding = self.embeddings[word]

        if self.normalizes:
            embedding = self.normalize(embedding)

        return embedding

    def normalize(self, embedding):
        norm = np.linalg.norm(embedding)

        if norm > 0.:
            return embedding / norm
        else:
            return embedding
