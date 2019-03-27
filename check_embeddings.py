import configs
import data_utils
from model import Model
import numpy as np

vocab = data_utils.vocab
model = Model()
embedding_mat = model.embedder.weight.detach().cpu().numpy()

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

        if word in vocab.word_to_id:
            overlap_cnt += 1
            assert vocab.word_to_id[word] is not vocab.unk_id
            embedding = np.array(
                list(map(float, embedding))
            )
            assert np.allclose(embedding_mat[vocab.word_to_id[word]], embedding)

    print(overlap_cnt)