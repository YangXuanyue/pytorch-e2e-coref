from model_utils import *
from collections import Counter, defaultdict
import data_utils
import configs

import logging
from typing import Union, List, Dict, Any
from allennlp.modules.elmo import _ElmoBiLm, remove_sentence_boundaries, batch_to_ids


class ElmoEmbedder(nn.Module):
    def __init__(self, device_id='cuda:0'):
        super().__init__()

        self.device = torch.device(device_id)
        self.elmo_lm = _ElmoBiLm(
            options_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/'
                         '2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json',
            weight_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/'
                        '2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
        ).to(self.device)

    def embed(self, sents):
        return self(batch_to_ids(sents).to(self.device))

    def forward(
            self,
            # [batch_size, sent_len, max_word_len]
            char_ids_seq_batch
    ):
        with torch.no_grad():
            _ = self.elmo_lm(char_ids_seq_batch)
            # layer_num * [batch_size, sent_len + 2, embedding_dim], [batch_size, sent_len + 2]
            orig_layer_output_batches, orig_mask_batch = _['activations'], _['mask']
            layer_output_batches, mask_batch = [], None

            for orig_layer_output_batch in orig_layer_output_batches:
                layer_output_batch, mask_batch = remove_sentence_boundaries(
                    orig_layer_output_batch, orig_mask_batch
                )
                layer_output_batches.append(layer_output_batch)

            # [batch_size, sent_len, embedding_dim, layer_num]
            layer_outputs_batch = torch.stack(layer_output_batches, dim=-1)

            return layer_outputs_batch, mask_batch


class ElmoLayerOutputMixer(nn.Module):
    def __init__(self):
        super().__init__()

        self.mix_weights = nn.Parameter(torch.randn((configs.elmo_layer_num), requires_grad=True))
        self.scale = nn.Parameter(torch.tensor(1., requires_grad=True))

    def forward(
            self,
            # [batch_size, sent_len, elmo_embedding_dim, elmo_layer_num]
            layer_outputs_batch
    ):
        # [batch_size, sent_len, elmo_embedding_dim]
        embedding_seq_batch = layer_outputs_batch @ F.softmax(self.mix_weights)
        # [batch_size, sent_len, elmo_embedding_dim]
        return embedding_seq_batch * self.scale


class CharCnnEmbedder(nn.Module):
    def __init__(
            self,
            vocab_size,
            padding_id=0
    ):
        super().__init__()
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=configs.char_embedding_dim,
            padding_idx=padding_id
        )

        self.feature_extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=configs.char_embedding_dim,
                        out_channels=kernel_num,
                        kernel_size=kernel_width,
                    ),
                    # nn.BatchNorm1d(num_features=kernel_num),
                    nn.ReLU(),
                    # nn.Tanh(),
                    # nn.Dropout(configs.dropout_prob),
                    nn.AdaptiveMaxPool1d(output_size=1),
                    Reshaper(-1, kernel_num)
                )
                for kernel_width, kernel_num in zip(configs.cnn_kernel_widths, configs.cnn_kernel_nums)
            ]
        )

    def forward(
            self,
            # [batch_size, max_sent_len, max_word_num]
            char_ids_seq_batch
    ):
        batch_size, max_sent_len, max_word_num = char_ids_seq_batch.shape
        # [batch_size * max_sent_len, max_word_num, char_embedding_dim]
        char_embedding_seq_batch = self.embedder(char_ids_seq_batch).view(-1, max_word_num, configs.char_embedding_dim)
        # [batch_size * max_sent_len, char_embedding_dim, max_word_num]
        char_embedding_seq_batch = char_embedding_seq_batch.transpose_(1, 2)

        # [batch_size, feature_num]
        return torch.cat(
            [
                # [batch_size * max_sent_len, kernel_num]
                feature_extractor(char_embedding_seq_batch)
                for feature_extractor in self.feature_extractors
            ], dim=-1
        ).view(batch_size, max_sent_len, configs.char_feature_num)


class LearnableRnnInitialState(nn.Module):
    def __init__(
            self,
            state_shape,
            rnn_type=nn.LSTM
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.state_shape = state_shape

        if self.rnn_type is nn.LSTM:
            self.initial_hidden_state = nn.Parameter(torch.randn(*self.state_shape))
            self.initial_cell_state = nn.Parameter(torch.randn(*self.state_shape))
        elif self.rnn_type is nn.GRU:
            self.initial_hidden_state = nn.Parameter(torch.randn(*self.state_shape))

    def get(self, batch_size):
        times = (1, batch_size, 1)[-len(self.state_shape):]

        if self.rnn_type is nn.LSTM:
            return (
                self.initial_hidden_state.repeat(*times),
                self.initial_cell_state.repeat(*times)
            )
        elif self.rnn_type is nn.GRU:
            return self.initial_hidden_state.repeat(*times)


class WeightDroppedRnn(nn.Module):
    def __init__(self, rnn, dropout_prob=0.):
        super().__init__()
        self.rnn = rnn
        self.weight_names = [
            name
            for name, param in self.rnn.named_parameters()
            # if 'weight' in name
            if 'weight_hh' in name
        ]
        self.dropout_prob = dropout_prob
        self._setup()

    def do_nothing(*args, **kwargs):
        return

    def _setup(self):
        if issubclass(type(self.rnn), nn.RNNBase):
            self.rnn.flatten_parameters = self.do_nothing

        for weight_name in self.weight_names:
            print(f'applying weight-drop of {self.dropout_prob} to {weight_name}')
            weight_param = getattr(self.rnn, weight_name)
            del self.rnn._parameters[weight_name]
            self.rnn.register_parameter(weight_name + '_raw', nn.Parameter(weight_param.data))

    def drop(self):
        for weight_name in self.weight_names:
            raw_weight_param = getattr(self.rnn, weight_name + '_raw')
            weight_param = F.dropout(raw_weight_param, p=self.dropout_prob, training=self.training)
            setattr(self.rnn, weight_name, weight_param)

    def forward(self, *args):
        self.drop()
        return self.rnn.forward(*args)


class HighwayGate(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(
            self,
            # [..., hidden_size]
            input_batch,
            # [..., hidden_size]
            output_batch
    ):
        packed = isinstance(input_batch, rnn_utils.PackedSequence)
        len_batch = None

        if packed:
            input_batch, len_batch = rnn_utils.pad_packed_sequence(input_batch)
            output_batch, len_batch = rnn_utils.pad_packed_sequence(output_batch)

        assert input_batch.shape == output_batch.shape
        *dims, hidden_size = output_batch.shape
        assert hidden_size == self.hidden_size

        input_batch = input_batch.view(-1, self.hidden_size)
        output_batch = output_batch.view(-1, self.hidden_size)
        # [*, hidden_size]
        g = F.sigmoid(self.projection(output_batch))
        # [..., hidden_size]
        output_batch = (g * output_batch + (1. - g) * input_batch).view(*dims, self.hidden_size)

        if packed:
            output_batch = rnn_utils.pack_padded_sequence(output_batch, len_batch)

        return output_batch


class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.input_size = input_size
        self.direction_num = 2
        self.hidden_size = configs.rnn_hidden_size // self.direction_num
        self.layer_num = configs.rnn_layer_num
        self.rnns = nn.ModuleList(
            *(
                WeightDroppedRnn(
                    nn.LSTM(
                        input_size=(
                            self.input_size if i == 0
                            else self.hidden_size * self.direction_num
                        ),
                        hidden_size=self.hidden_size,
                        bidirectional=True
                    ),
                    dropout_prob=configs.lstm_dropout_prob
                )
                for i in range(self.layer_num)
            )
        )
        self.initial_states = nn.ModuleList(
            *(
                LearnableRnnInitialState(
                    state_shape=(self.direction_num, 1, self.hidden_size),
                    rnn_type=nn.LSTM
                )
                for _ in range(self.layer_num)
            )
        )
        self.highway_gates = nn.ModuleList(
            *(
                HighwayGate(self.hidden_size * self.direction_num)
                for _ in range(self.layer_num - 1)
            )
        )

    def forward(
            self,
            # [batch_size, max_sent_len, tot_embedding_dim], [batch_size]
            embedding_seq_batch, sent_len_batch
    ):
        # [max_sent_len, batch_size, tot_embedding_dim]
        curr_input_batch = embedding_seq_batch.transpose(0, 1)
        max_sent_len, batch_size, tot_embedding_dim = curr_input_batch.shape
        sorted_idx_batch = sorted(range(batch_size), key=sent_len_batch.__getitem__, reverse=True)
        orig_idx_batch = [-1] * batch_size

        for sorted_idx, orig_idx in enumerate(sorted_idx_batch):
            orig_idx_batch[orig_idx] = sorted_idx

        # [max_sent_len, batch_size(decreasing), tot_embedding_dim]
        curr_input_batch = rnn_utils.pack_padded_sequence(
            curr_input_batch[sorted_idx_batch, ...], sent_len_batch[sorted_idx_batch]
        )
        curr_output_batch = None

        for i in range(self.layer_num):
            # [max_sent_len, batch_size(decreasing), hidden_size * layer_num]
            curr_output_batch, *_ = self.rnns[i](
                curr_input_batch, self.initial_states[i].get(batch_size)
            )
            # TODO: dropout?
            if i > 0:
                curr_output_batch = self.highway_gates[i - 1](curr_input_batch, curr_output_batch)
            # [max_sent_len, batch_size(decreasing), hidden_size * layer_num]
            curr_input_batch = curr_output_batch

        # [max_sent_len, batch_size, hidden_size * layer_num]
        curr_output_batch, _ = rnn_utils.pad_packed_sequence(curr_output_batch)
        # [batch_size, max_sent_len, hidden_size * layer_num]
        return curr_output_batch[orig_idx_batch, ...].transpose(0, 1)

# zoneout
# https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/models/modules/recurrent.py

class ResidualConnection(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, input):
        return self.layer(input) + input


class ScaledDotProdAttention(nn.Module):
    # http://nlp.seas.harvard.edu/2018/04/03/attention.html
    def __init__(
            self,
            raw_key_size=configs.rnn_hidden_size,
            raw_value_size=configs.rnn_hidden_size,
            raw_query_size=configs.rnn_hidden_size,
    ):
        super().__init__()
        self.raw_key_size = raw_key_size
        self.raw_value_size = raw_value_size
        self.raw_query_size = raw_query_size
        self.keys_batch, self.values_batch, self.masks_batch = None, None, None
        self.factor = configs.query_size ** -.5
        self.min_score = -1e30

        self.key_mapper = nn.Linear(self.raw_key_size, configs.key_size)
        self.value_mapper = nn.Linear(self.raw_value_size, configs.value_size)
        self.query_mapper = nn.Linear(self.raw_query_size, configs.query_size)

    def clear(self):
        self.keys_batch, self.values_batch, self.masks_batch = None, None, None

    def set(
            self,
            # [batch_size, seq_len, raw_key_size]
            raw_keys_batch,
            # [batch_size, seq_len, raw_value_size]
            raw_values_batch,
            # [batch_size, seq_len]
            masks_batch=None
    ):
        # [batch_size, seq_len, key_size]
        self.keys_batch = self.key_mapper(raw_keys_batch)
        # [batch_size, seq_len, key_size, 1]
        self.keys_batch = self.keys_batch.view(*self.keys_batch.shape, 1)
        self.batch_size, *_ = self.keys_batch.shape
        # [batch_size, seq_len, value_size]
        self.values_batch = self.value_mapper(raw_values_batch)
        # [batch_size, seq_len]
        self.masks_batch = masks_batch

    def append(
            self,
            # [batch_size, raw_key_size]
            raw_key_batch,
            # [batch_size, raw_value_size]
            raw_value_batch,
            # [batch_size]
            mask_batch=None
    ):
        self.batch_size, _ = raw_key_batch.shape
        # [batch_size, 1, key_size, 1]
        key_batch = self.key_mapper(raw_key_batch).view(self.batch_size, 1, -1, 1)
        # [batch_size, 1, key_size]
        value_batch = self.value_mapper(raw_value_batch).view(self.batch_size, 1, -1)

        if mask_batch is not None:
            mask_batch = mask_batch.view(self.batch_size, 1)

        if self.keys_batch is None:
            self.keys_batch, self.values_batch, self.masks_batch = key_batch, value_batch, mask_batch
        else:
            # [batch_size, seq_len, key_size, 1]
            self.keys_batch = torch.cat(
                (self.keys_batch, key_batch),
                dim=1
            )
            # [batch_size, seq_len, key_size]
            self.values_batch = torch.cat(
                (self.values_batch, value_batch),
                dim=1
            )

            if mask_batch is not None:
                # [batch_size, seq_len]
                self.masks_batch = torch.cat(
                    (self.masks_batch, mask_batch),
                    dim=1
                )

    # a workaround for reindexing in beam search
    def __getitem__(
            self,
            # [batch_size]
            idx_batch
    ):
        # [batch_size, seq_len, key_size, 1]
        self.keys_batch = self.keys_batch[idx_batch]
        # [batch_size, seq_len, key_size]
        self.values_batch = self.values_batch[idx_batch]
        # [batch_size, seq_len]
        self.masks_batch = self.masks_batch[idx_batch]

        return self

    def forward(
            self,
            # [batch_size, raw_query_size]
            raw_query_batch,
    ):
        # [batch_size, 1, 1, query_size]
        query_batch = self.query_mapper(raw_query_batch).view(self.batch_size, 1, 1, -1)
        # [batch_size, seq_len]
        # = ([batch_size, 1, 1, query_size] @ [batch_size, seq_len, key_size, 1]).view(batch_size, seq_len)
        scores_batch = (query_batch @ self.keys_batch).view(self.batch_size, -1)

        if self.masks_batch is not None:
            scores_batch[self.masks_batch] = self.min_score

        # [batch_size, 1, seq_len]
        self.scores_batch = F.softmax(
            scores_batch * self.factor,
            dim=-1
        ).view(self.batch_size, 1, -1)
        # [batch_size, value_size]
        # = ([batch_size, 1, seq_len] @ [batch_size, seq_len, value_size]).view(batch_size, value_size)
        return (self.scores_batch @ self.values_batch).view(self.batch_size, -1)

    # def get_scores_batch(self):
    #     # [batch_size, 1, seq_len]
    #     return self.scores_batch


class MultiHeadScaledDotProdAttention(nn.Module):
    # http://nlp.seas.harvard.edu/2018/04/03/attention.html
    def __init__(
            self,
            raw_key_size=configs.rnn_hidden_size,
            raw_value_size=configs.rnn_hidden_size,
            raw_query_size=configs.rnn_hidden_size,
            head_num=4
    ):
        super().__init__()
        self.raw_key_size = raw_key_size
        self.raw_value_size = raw_value_size
        self.raw_query_size = raw_query_size
        self.head_num = head_num
        self.keys_batches, self.values_batches, self.masks_batch = \
            [None] * self.head_num, [None] * self.head_num, None
        self.factor = (configs.query_size // self.head_num) ** -.5
        self.min_score = -1e30

        self.key_mappers = nn.ModuleList(
            (
                nn.Linear(self.raw_key_size, configs.key_size // self.head_num)
                for _ in range(self.head_num)
            )
        )
        self.value_mappers = nn.ModuleList(
            (
                nn.Linear(self.raw_value_size, configs.value_size // self.head_num)
                for _ in range(self.head_num)
            )
        )
        self.query_mappers = nn.ModuleList(
            (
                nn.Linear(self.raw_query_size, configs.query_size // self.head_num)
                for _ in range(self.head_num)
            )
        )

    def clear(self):
        self.keys_batches, self.values_batches, self.masks_batches = \
            [None] * self.head_num, [None] * self.head_num, None

    def set(
            self,
            # [batch_size, seq_len, raw_key_size]
            raw_keys_batch,
            # [batch_size, seq_len, raw_value_size]
            raw_values_batch,
            # [batch_size, seq_len]
            masks_batch=None
    ):
        self.batch_size, *_ = raw_keys_batch.shape

        for i in range(self.head_num):
            # [batch_size, seq_len, key_size / head_num]
            self.keys_batches[i] = self.key_mappers[i](raw_keys_batch)
            # [batch_size, seq_len, key_size / head_num, 1]
            self.keys_batches[i] = self.keys_batches[i].view(*self.keys_batches[i].shape, 1)
            # [batch_size, seq_len, value_size / head_num]
            self.values_batches[i] = self.value_mappers[i](raw_values_batch)
            # [batch_size, seq_len]

        self.masks_batch = masks_batch

    def append(
            self,
            # [batch_size, raw_key_size]
            raw_key_batch,
            # [batch_size, raw_value_size]
            raw_value_batch,
            # [batch_size]
            mask_batch=None
    ):
        self.batch_size, _ = raw_key_batch.shape

        if mask_batch is not None:
            mask_batch = mask_batch.view(self.batch_size, 1)

            if self.masks_batch is None:
                self.masks_batch = mask_batch
            else:
                self.masks_batch = torch.cat(
                    (self.masks_batch, mask_batch),
                    dim=1
                )

        for i in range(self.head_num):
            # [batch_size, 1, key_size / head_num, 1]
            key_batch = self.key_mappers[i](raw_key_batch).view(self.batch_size, 1, -1, 1)
            # [batch_size, 1, key_size / head_num]
            value_batch = self.value_mappers[i](raw_value_batch).view(self.batch_size, 1, -1)

            if self.keys_batches[i] is None:
                self.keys_batches[i], self.values_batches[i] = key_batch, value_batch
            else:
                # [batch_size, seq_len, key_size / head_num, 1]
                self.keys_batches[i] = torch.cat(
                    (self.keys_batches[i], key_batch),
                    dim=1
                )
                # [batch_size, seq_len, value_size / head_num]
                self.values_batches[i] = torch.cat(
                    (self.values_batches[i], value_batch),
                    dim=1
                )

    # a workaround for reindexing in beam search
    def __getitem__(
            self,
            # [batch_size]
            idx_batch
    ):
        for i in range(self.head_num):
            # [batch_size, seq_len, key_size / head_num, 1]
            self.keys_batches[i] = self.keys_batches[i][idx_batch]
            # [batch_size, seq_len, value_size / head_num]
            self.values_batches[i] = self.values_batches[i][idx_batch]
            # [batch_size, seq_len]
            self.masks_batches[i] = self.masks_batches[i][idx_batch]

        return self

    def forward(
            self,
            # [batch_size, raw_query_size]
            raw_query_batch,
    ):
        context_batches = []

        for i in range(self.head_num):
            # [batch_size, 1, 1, query_size / head_num]
            query_batch = self.query_mappers[i](raw_query_batch).view(self.batch_size, 1, 1, -1)
            # [batch_size, seq_len]
            # = ([batch_size, 1, 1, query_size / head_num] @ [batch_size, seq_len, key_size / head_num, 1])
            #   .view(batch_size, seq_len)
            scores_batch = (query_batch @ self.keys_batches[i]).view(self.batch_size, -1)

            if self.masks_batch is not None:
                scores_batch[self.masks_batch] = self.min_score

            # [batch_size, 1, seq_len]
            scores_batch = F.softmax(
                scores_batch * self.factor,
                dim=-1
            ).view(self.batch_size, 1, -1)
            # [batch_size, value_size / head_num]
            # = ([batch_size, 1, seq_len] @ [batch_size, seq_len, value_size]).view(batch_size, value_size)
            context_batches.append((scores_batch @ self.values_batches[i]).view(self.batch_size, -1))

        # [batch_size, value_size]
        return torch.cat(
            # head_num * [batch_size, value_size / head_num]
            context_batches,
            dim=-1
        )

    # def get_scores_batch(self):
    #     # [batch_size, 1, seq_len]
    #     return self.scores_batch


class LangModel(nn.Module):
    def __init__(self, vocab, device_id, requires_grad=False):
        super().__init__()
        self.device = torch.device(device_id)
        self.requires_grad = requires_grad
        self.vocab = vocab
        self.layer_num = 2
        self.rnn_cells = nn.ModuleList(
            [
                WeightDroppedRnnCell(
                    nn.LSTMCell(
                        input_size=configs.word_embedding_dim if l == 0 else configs.rnn_hidden_size,
                        hidden_size=configs.rnn_hidden_size
                    )
                )
                for l in range(self.layer_num)
            ]
        ).to(self.device)
        self.initial_rnn_states = nn.ModuleList(
            [
                LearnableRnnInitialState(
                    state_shape=(1, configs.rnn_hidden_size)
                )
                for l in range(self.layer_num)
            ]
        ).to(self.device)
        self.hidden_states = [
            None
            for l in range(self.layer_num)
        ]
        self.cell_states = [
            None
            for l in range(self.layer_num)
        ]

        self.classifier = nn.Linear(configs.rnn_hidden_size, self.vocab.size).to(self.device)

    def _run(
            self,
            # [batch_size, embedding_dim]
            embedding_batch,
            initial=False
    ):
        batch_size, _ = embedding_batch.shape

        if initial:
            for l in range(self.layer_num):
                self.rnn_cells[l].start()
                self.hidden_states[l], self.cell_states[l] = self.initial_rnn_states[l].get(batch_size)

        state_batch = embedding_batch.to(self.device)

        for l in range(self.layer_num):
            self.hidden_states[l], self.cell_states[l] = self.rnn_cells[l](
                state_batch,
                (self.hidden_states[l], self.cell_states[l])
            )
            state_batch = self.hidden_states[l]

        # [batch_size, vocab_size]
        logits_batch = self.classifier(state_batch)

        return logits_batch

    def run(
            self,
            # [batch_size, embedding_dim]
            embedding_batch,
            initial=False
    ):
        if not self.requires_grad:
            with torch.no_grad():
                return self._run(embedding_batch, initial)
        else:
            return self._run(embedding_batch, initial)


class Decoder(nn.Module):
    def __init__(self, vocab, device_id):
        super().__init__()
        self.device = torch.device(device_id)
        self.gumbel_distribution = torch.distributions.Gumbel(0, 1)

        self.vocab = vocab
        self.embedder = nn.Embedding(
            embedding_dim=configs.word_embedding_dim,
            num_embeddings=self.vocab.size,
            padding_idx=self.vocab.padding_id
        )

        if configs.uses_gumbel_softmax:
            self.embedder = self.embedder.to(self.device)

        # ia: image-attended
        self.iarnn_cell = WeightDroppedRnnCell(
            nn.LSTMCell(
                input_size=(configs.word_embedding_dim + configs.value_size),
                hidden_size=configs.rnn_hidden_size
            )
        ).to(self.device)
        self.initial_iarnn_state = LearnableRnnInitialState(
            state_shape=(1, configs.rnn_hidden_size)
        ).to(self.device)
        self.utterance_attention = ScaledDotProdAttention().to(self.device)
        # ta: transcript-attended
        self.initial_tarnn_state = LearnableRnnInitialState(
            state_shape=(1, configs.rnn_hidden_size)
        ).to(self.device)
        self.tarnn_cell = WeightDroppedRnnCell(
            nn.LSTMCell(
                # input_size=(configs.rnn_hidden_size + configs.value_size),
                input_size=configs.rnn_hidden_size,
                hidden_size=configs.rnn_hidden_size
            )
        ).to(self.device)
        # self.transcript_attention = ScaledDotProdAttention().to(self.device)
        self.classifier = nn.Linear(configs.rnn_hidden_size, self.vocab.size).to(self.device)

    def run_lang_model(
            self,
            # [seq_len, batch_size]
            transcript_batch,
            # [batch_size]
            transcript_len_batch
    ):
        max_transcript_len, batch_size = transcript_batch.shape
        # [seq_len, batch_size, embedding_dim]
        embeddings_batch = self.embedder(
            transcript_batch.to(self.device) if configs.uses_gumbel_softmax else transcript_batch
        ).to(self.device)
        embeddings_batch = F.dropout(embeddings_batch, p=configs.dropout_prob, training=self.training)
        iarnn_hidden_state_batch, iarnn_cell_state_batch = self.initial_iarnn_state.get(batch_size)
        tarnn_hidden_state_batch, tarnn_cell_state_batch = self.initial_tarnn_state.get(batch_size)
        next_embedding_batch = embeddings_batch[0]
        next_char_logits_seq_batch = []
        self.iarnn_cell.start()
        self.tarnn_cell.start()

        for t in range(max_transcript_len - 1):
            # [batch_size, hidden_size], [batch_size, hidden_size]
            iarnn_hidden_state_batch, iarnn_cell_state_batch = self.iarnn_cell(
                # [batch_size, embedding_dim + value_size]
                torch.cat(
                    (
                        # [batch_size, embedding_dim]
                        next_embedding_batch,
                        # [batch_size, value_size]
                        torch.zeros(batch_size, configs.value_size).to(self.device)

                    ), dim=-1
                ),
                (iarnn_hidden_state_batch, iarnn_cell_state_batch)
            )
            tarnn_hidden_state_batch, tarnn_cell_state_batch = self.tarnn_cell(
                iarnn_hidden_state_batch,
                (tarnn_hidden_state_batch, tarnn_cell_state_batch)
            )
            # [batch_size, vocab_size]
            next_char_logits_batch = self.classifier(tarnn_hidden_state_batch)
            next_char_logits_seq_batch.append(next_char_logits_batch)

            if self.training and random.random() < configs.sampling_rate:
                if configs.uses_gumbel_softmax:
                    # [batch_size, vocab_size]
                    next_char_probs_batch = F.gumbel_softmax(next_char_logits_batch, tau=.5)
                    # [batch_size, embedding] = [batch_size, vocab_size] @ [vocab_size, embedding_dim]
                    next_embedding_batch = next_char_probs_batch @ self.embedder.weight
                else:
                    # [batch_size, vocab_size]
                    gumbel_noises_batch = self.gumbel_distribution.sample(next_char_logits_batch.shape).to(self.device)
                    # [batch_size]
                    next_char_id_batch = torch.argmax(next_char_logits_batch.detach() + gumbel_noises_batch, dim=-1)
                    # [batch_size, embedding_dim]
                    next_embedding_batch = self.embedder(next_char_id_batch.cpu()).to(self.device)
            else:
                next_embedding_batch = embeddings_batch[t + 1]

        # [max_transcript_len - 1, batch_size, vocab_size]
        return torch.stack(
            # (max_transcript_len - 1) * [batch_size, vocab_size]
            next_char_logits_seq_batch,
            dim=0
        )

    def forward(
            self,
            # [seq_len, batch_size]
            report_batch,
            # [batch_size]
            report_len_batch,
            # [batch_size, feature_num, map_height, map_weight]
            feature_maps_batch,
    ):
        # [batch_size, max_encoded_utterance_len, hidden_size]
        feature_maps_batch = feature_maps_batch.transpose(0, 1).to(self.device)
        max_transcript_len, batch_size = report_batch.shape
        # [seq_len, batch_size, embedding_dim]
        embeddings_batch = self.embedder(
            report_batch.to(self.device) if configs.uses_gumbel_softmax else report_batch
        ).to(self.device)
        embeddings_batch = F.dropout(embeddings_batch, configs.dropout_prob, self.training)
        iarnn_hidden_state_batch, iarnn_cell_state_batch = self.initial_iarnn_state.get(batch_size)

        _, max_encoded_utterance_len, _ = feature_maps_batch.shape
        # [batch_size, max_len]
        idxes_batch = torch.arange(max_encoded_utterance_len).view(1, -1).repeat(batch_size, 1)
        # [batch_size, max_len] = [batch_size, max_len] >= [batch_size, 1]
        masks_batch = idxes_batch.ge(encoded_utterance_len_batch.view(-1, 1))

        self.utterance_attention.set(
            # [batch_size, max_encoded_utterance, hidden_size]
            raw_keys_batch=feature_maps_batch,
            # [batch_size, max_encoded_utterance, hidden_size]
            raw_values_batch=feature_maps_batch,
            masks_batch=masks_batch
        )
        tarnn_hidden_state_batch, tarnn_cell_state_batch = self.initial_tarnn_state.get(batch_size)
        # self.transcript_attention.clear()
        next_embedding_batch = embeddings_batch[0]
        next_char_logits_seq_batch = []
        self.iarnn_cell.start()
        self.tarnn_cell.start()
        # transcript_attention_map_batch = []

        for t in range(max_transcript_len - 1):
            # [batch_size, hidden_size], [batch_size, hidden_size]
            iarnn_hidden_state_batch, iarnn_cell_state_batch = self.iarnn_cell(
                # [batch_size, embedding_dim + value_size]
                torch.cat(
                    (
                        # [batch_size, embedding_dim]
                        next_embedding_batch,
                        # [batch_size, value_size]
                        self.utterance_attention(
                            raw_query_batch=iarnn_hidden_state_batch
                        )

                    ), dim=-1
                ),
                (iarnn_hidden_state_batch, iarnn_cell_state_batch)
            )
            # [batch_size, hidden_size], [batch_size, hidden_size]
            tarnn_hidden_state_batch, tarnn_cell_state_batch = self.tarnn_cell(
                iarnn_hidden_state_batch,
                (tarnn_hidden_state_batch, tarnn_cell_state_batch)
            )

            # [batch_size, vocab_size]
            next_char_logits_batch = self.classifier(tarnn_hidden_state_batch)
            next_char_logits_seq_batch.append(next_char_logits_batch)

            if self.training and random.random() < configs.sampling_rate:
                if configs.uses_gumbel_softmax:
                    # [batch_size, vocab_size]
                    next_char_probs_batch = F.gumbel_softmax(next_char_logits_batch, tau=.5)
                    # [batch_size, embedding] = [batch_size, vocab_size] @ [vocab_size, embedding_dim]
                    next_embedding_batch = next_char_probs_batch @ self.embedder.weight
                else:
                    # [batch_size, vocab_size]
                    gumbel_noises_batch = self.gumbel_distribution.sample(next_char_logits_batch.shape).to(self.device)
                    # [batch_size]
                    next_char_id_batch = torch.argmax(next_char_logits_batch.detach() + gumbel_noises_batch, dim=-1)
                    # [batch_size, embedding_dim]
                    next_embedding_batch = self.embedder(next_char_id_batch.cpu()).to(self.device)
            else:
                next_embedding_batch = embeddings_batch[t + 1]

        # [max_transcript_len - 1, batch_size, vocab_size]
        next_char_logits_seq_batch = torch.stack(
            # (max_transcript_len - 1) * [batch_size, vocab_size]
            next_char_logits_seq_batch,
            dim=0
        )

        return next_char_logits_seq_batch

    def decode(
            self,
            # [max_encoded_utterance, batch_size, hidden_size]
            encoded_utterance_batch,
            # [batch_size]
            encoded_utterance_len_batch,
            beam_width=configs.beam_width,
            per_node_beam_width=configs.per_node_beam_width,
            max_len=300
    ):
        batch_size, = encoded_utterance_len_batch.shape
        # [batch_size, max_encoded_utterance, hidden_size]
        encoded_utterance_batch = encoded_utterance_batch.transpose(0, 1).to(self.device)

        best_seq_batch = []
        # [vocab_size]
        end_char_log_probs = torch.full((self.vocab.size,), -1e30).to(self.device)
        # [vocab_size]
        end_char_log_probs[self.vocab.end_id] = 0.
        # [beam_width, vocab_size]
        end_char_log_probs_beam = end_char_log_probs.view(1, -1).repeat(beam_width, 1).to(self.device)

        for idx_in_batch in range(batch_size):
            encoded_utterance_len = encoded_utterance_len_batch[idx_in_batch]
            # [1, encoded_utterance_len, hidden_size]
            encoded_utterance = encoded_utterance_batch[idx_in_batch, :encoded_utterance_len]

            self.utterance_attention.set(
                # [beam_width, encoded_utterance_len, hidden_size]
                raw_keys_batch=encoded_utterance.repeat(beam_width, 1, 1),
                # [beam_width, encoded_utterance_len, hidden_size]
                raw_values_batch=encoded_utterance.repeat(beam_width, 1, 1),
            )

            iarnn_hidden_state_beam, iarnn_cell_state_beam = self.initial_iarnn_state.get(beam_width)
            tarnn_hidden_state_beam, tarnn_cell_state_beam = self.initial_tarnn_state.get(beam_width)

            # self.transcript_attention.clear()
            self.iarnn_cell.start()
            self.tarnn_cell.start()

            # [beam_width]
            next_char_id_beam = torch.full((beam_width,), self.vocab.start_id, dtype=torch.long)
            # [beam_width, embedding_dim]
            next_embedding_beam = self.embedder(next_char_id_beam).to(self.device)
            # [beam_width, seq_len(just 1 now)]
            char_id_seq_beam = next_char_id_beam.view(beam_width, 1)
            # [beam_width]
            char_id_seq_len_beam = torch.full((beam_width,), 0, dtype=torch.long).to(self.device)
            # [beam_width]
            seq_log_prob_beam = torch.zeros(beam_width).to(self.device)

            for t in range(max_len):
                # [beam_width, hidden_size], [beam_width, hidden_size]
                iarnn_hidden_state_beam, iarnn_cell_state_beam = self.iarnn_cell(
                    # [beam_width, embedding_dim + value_size]
                    torch.cat(
                        (
                            # [beam_width, embedding_dim]
                            next_embedding_beam,
                            # [beam_width, value_size]
                            self.utterance_attention(
                                raw_query_batch=iarnn_hidden_state_beam
                            )

                        ), dim=-1
                    ),
                    (iarnn_hidden_state_beam, iarnn_cell_state_beam)
                )

                end_flag_beam = next_char_id_beam.eq(self.vocab.end_id)

                # [beam_width, hidden_size], [beam_width, hidden_size]
                tarnn_hidden_state_beam, tarnn_cell_state_beam = self.tarnn_cell(
                    iarnn_hidden_state_beam,
                    (tarnn_hidden_state_beam, tarnn_cell_state_beam)
                )
                # [beam_width, vocab_size]
                next_char_logits_beam = self.classifier(tarnn_hidden_state_beam)
                # [beam_width, vocab_size]
                end_flags_beam = end_flag_beam.view(-1, 1).repeat(1, self.vocab.size).to(self.device)
                # [beam_width, vocab_size]
                next_char_log_probs_beam = torch.where(
                    # [beam_width, vocab_size]
                    end_flags_beam,
                    # [beam_width, vocab_size]
                    end_char_log_probs_beam,
                    # [beam_width, vocab_size]
                    F.log_softmax(next_char_logits_beam, dim=-1)
                )
                # [beam_width, vocab_size] = [beam_width, 1] + [beam_width, vocab_size]
                next_char_log_probs_beam = seq_log_prob_beam.view(-1, 1) + next_char_log_probs_beam
                # [beam_width, vocab_size]
                char_id_seq_lens_beam = char_id_seq_len_beam.view(-1, 1).repeat(1, self.vocab.size)
                # [beam_width, vocab_size]
                char_id_seq_lens_beam[1 - end_flags_beam] += 1
                # [beam_width, vocab_size] = [beam_width, vocab_size] / [beam_width, vocab_size]
                normalized_next_char_log_probs_beam = next_char_log_probs_beam / char_id_seq_lens_beam.float()
                # [beam_width, per_node_beam_width], [beam_width, per_node_beam_width]
                top_normalized_next_char_log_probs_beam, top_next_char_ids_beam = torch.topk(
                    normalized_next_char_log_probs_beam, per_node_beam_width
                )
                # [beam_width * per_node_beam_width]
                all_top_normalized_next_char_log_probs = top_normalized_next_char_log_probs_beam.view(-1)
                # [beam_width * per_node_beam_width]
                all_top_next_char_ids = top_next_char_ids_beam.view(-1)
                # [beam_width], [beam_width]
                normalized_next_char_log_prob_beam, idx_beam = all_top_normalized_next_char_log_probs.topk(beam_width)
                # [beam_width]
                idx_beam = idx_beam.cpu()
                # idx = idx_in_beam * per_node_beam_width + idx_in_next_beam
                # [beam_width]
                prev_idx_beam = idx_beam / per_node_beam_width
                # [beam_width]
                next_char_id_beam = all_top_next_char_ids[idx_beam].cpu()
                next_embedding_beam = self.embedder(next_char_id_beam).to(self.device)
                # [beam_width, hidden_size]
                iarnn_hidden_state_beam = iarnn_hidden_state_beam[prev_idx_beam]
                # [beam_width, hidden_size]
                iarnn_cell_state_beam = iarnn_cell_state_beam[prev_idx_beam]
                # [beam_width, seq_len - 1]
                char_id_seq_beam = char_id_seq_beam[prev_idx_beam]
                # [beam_width, seq_len]
                char_id_seq_beam = torch.cat(
                    (
                        # [beam_width, seq_len - 1]
                        char_id_seq_beam,
                        # [beam_width, 1]
                        next_char_id_beam.view(beam_width, 1)
                    ), dim=1
                )
                # [beam_width] = [beam_width, vocab_size][[beam_width], [beam_width]]
                seq_log_prob_beam = next_char_log_probs_beam[prev_idx_beam, next_char_id_beam]
                # [beam_width] = [beam_width, vocab_size][[beam_width], [beam_width]]
                char_id_seq_len_beam = char_id_seq_lens_beam[prev_idx_beam, next_char_id_beam]

                if next_char_id_beam.eq(self.vocab.end_id).all() or t == max_len - 1:
                    best_seq = char_id_seq_beam[0].numpy().tolist()

                    if self.vocab.end_id in best_seq:
                        best_seq = best_seq[:best_seq.index(self.vocab.end_id) + 1]
                    else:
                        best_seq.append(self.vocab.end_id)

                    best_seq_batch.append(best_seq)

                    break

        return best_seq_batch


class SelfAttendedDecoder(nn.Module):
    def __init__(self, vocab, device_id):
        super().__init__()
        self.device = torch.device(device_id)
        self.gumbel_distribution = torch.distributions.Gumbel(0, 1)

        self.vocab = vocab
        self.embedder = nn.Embedding(
            embedding_dim=configs.word_embedding_dim,
            num_embeddings=self.vocab.size,
            padding_idx=self.vocab.padding_id
        )

        if configs.uses_gumbel_softmax:
            self.embedder = self.embedder.to(self.device)
        # ia: utterance-attended
        self.iarnn_cell = WeightDroppedRnnCell(
            nn.LSTMCell(
                input_size=(configs.word_embedding_dim + configs.value_size),
                hidden_size=configs.rnn_hidden_size
            )
        ).to(self.device)
        self.initial_iarnn_state = LearnableRnnInitialState(
            state_shape=(1, configs.rnn_hidden_size)
        ).to(self.device)
        # self.utterance_attention = ScaledDotProdAttention().to(self.device)
        # if configs.use_multi_head_attention:
        #     self.image_attention = MultiHeadScaledDotProdAttention().to(self.device)
        # else:
        #     self.image_attention = ScaledDotProdAttention().to(self.device)
        self.image_attention = ScaledDotProdAttention(
            raw_key_size=configs.char_feature_num,
            raw_value_size=configs.char_feature_num
        ).to(self.device)
        # sa: self-attended
        self.initial_sarnn_state = LearnableRnnInitialState(
            state_shape=(1, configs.rnn_hidden_size)
        ).to(self.device)
        self.sarnn_cell = WeightDroppedRnnCell(
            nn.LSTMCell(
                input_size=(configs.rnn_hidden_size + configs.value_size),
                hidden_size=configs.rnn_hidden_size
            )
        ).to(self.device)
        self.self_attention = ScaledDotProdAttention().to(self.device)
        # self.mlp = nn.Sequential(
        #     nn.Linear(configs.rnn_hidden_size, configs.rnn_hidden_size),
        #     nn.BatchNorm1d(configs.rnn_hidden_size),
        #     nn.SELU(),
        #     nn.Dropout(configs.dropout_prob),
        #     nn.Linear(configs.rnn_hidden_size, configs.rnn_hidden_size),
        #     nn.BatchNorm1d(configs.rnn_hidden_size),
        #     nn.SELU()
        # ).to(self.device)
        self.classifier = nn.Linear(configs.rnn_hidden_size, self.vocab.size).to(self.device)

        # if configs.uses_lang_model:
        #     self.lang_model = LangModel(vocab=self.vocab, device_id=configs.decoder_device_id, requires_grad=False)
        #     self.score_merger = nn.Linear(2, 1, bias=False).to(self.device)
        #     self.score_merger.weight.data.copy_(torch.FloatTensor([[1., 0.008]]).to(self.device))

    def run_lang_model(
            self,
            # [seq_len, batch_size]
            report_batch,
            # [batch_size]
            report_len_batch
    ):
        max_report_len, batch_size = report_batch.shape
        # [seq_len, batch_size, embedding_dim]
        embeddings_batch = self.embedder(report_batch).to(self.device)
        iarnn_hidden_state_batch, iarnn_cell_state_batch = self.initial_iarnn_state.get(batch_size)
        sarnn_hidden_state_batch, sarnn_cell_state_batch = self.initial_sarnn_state.get(batch_size)
        self.self_attention.clear()
        next_embedding_batch = embeddings_batch[0]
        next_word_logits_seq_batch = []
        self.iarnn_cell.start()
        self.sarnn_cell.start()

        for t in range(max_report_len - 1):
            # [batch_size, hidden_size], [batch_size, hidden_size]
            iarnn_hidden_state_batch, iarnn_cell_state_batch = self.iarnn_cell(
                # [batch_size, embedding_dim + value_size]
                torch.cat(
                    (
                        # [batch_size, embedding_dim]
                        next_embedding_batch,
                        # [batch_size, value_size]
                        torch.zeros(batch_size, configs.value_size).to(self.device)

                    ), dim=-1
                ),
                (iarnn_hidden_state_batch, iarnn_cell_state_batch)
            )
            self.self_attention.append(
                raw_key_batch=iarnn_hidden_state_batch,
                raw_value_batch=iarnn_hidden_state_batch,
                # [batch_size]
                mask_batch=report_len_batch.le(t)
            )
            # [batch_size, hidden_size], [batch_size, hidden_size]
            sarnn_hidden_state_batch, sarnn_cell_state_batch = self.sarnn_cell(
                # [batch_size, hidden_size + value_size]
                torch.cat(
                    (
                        # [batch_size, hidden_size]
                        iarnn_hidden_state_batch,
                        # [batch_size, value_size]
                        # torch.zeros(batch_size, configs.value_size).to(self.device)
                        # [batch_size, value_size]
                        self.self_attention(
                            raw_query_batch=sarnn_hidden_state_batch
                        )
                    ), dim=-1
                ),
                (sarnn_hidden_state_batch, sarnn_cell_state_batch)
            )
            # [batch_size, vocab_size]
            next_word_logits_batch = self.classifier(
                sarnn_hidden_state_batch
                # self.mlp(sarnn_hidden_state_batch)
            )
            next_word_logits_seq_batch.append(next_word_logits_batch)

            if self.training and random.random() < configs.sampling_rate:
                if configs.uses_gumbel_softmax:
                    # [batch_size, vocab_size]
                    next_word_probs_batch = F.gumbel_softmax(next_word_logits_batch, tau=.5)
                    # [batch_size, embedding] = [batch_size, vocab_size] @ [vocab_size, embedding_dim]
                    next_embedding_batch = next_word_probs_batch @ self.embedder.weight
                else:
                    # [batch_size, vocab_size]
                    gumbel_noises_batch = self.gumbel_distribution.sample(next_word_logits_batch.shape).to(self.device)
                    # [batch_size]
                    next_word_id_batch = torch.argmax(next_word_logits_batch.detach() + gumbel_noises_batch, dim=-1)
                    # [batch_size, embedding_dim]
                    next_embedding_batch = self.embedder(next_word_id_batch.cpu()).to(self.device)
            else:
                next_embedding_batch = embeddings_batch[t + 1]

        # [max_report_len - 1, batch_size, vocab_size]
        return torch.stack(
            # (max_report_len - 1) * [batch_size, vocab_size]
            next_word_logits_seq_batch,
            dim=0
        )

    def forward(
            self,
            # [seq_len, batch_size]
            report_batch,
            # [batch_size]
            report_len_batch,
            # [batch_size, feature_num, map_height, map_weight]
            image_feature_maps_batch
    ):
        # [batch_size, map_height * map_weight, feature_num]
        image_feature_maps_batch = image_feature_maps_batch.view(
            # [batch_size, feature_num, map_height * map_weight]
            *image_feature_maps_batch.shape[:2], -1
        ).transpose(1, 2).to(self.device)
        # print(image_feature_maps_batch.shape)
        max_report_len, batch_size = report_batch.shape
        # [seq_len, batch_size, embedding_dim]
        embeddings_batch = self.embedder(
            report_batch.to(self.device) if configs.uses_gumbel_softmax else report_batch
        ).to(self.device)
        embeddings_batch = F.dropout(embeddings_batch, p=configs.dropout_prob, training=self.training)
        iarnn_hidden_state_batch, iarnn_cell_state_batch = self.initial_iarnn_state.get(batch_size)

        self.image_attention.set(
            # [batch_size, max_encoded_utterance, hidden_size]
            raw_keys_batch=image_feature_maps_batch,
            # [batch_size, max_encoded_utterance, hidden_size]
            raw_values_batch=image_feature_maps_batch
        )
        sarnn_hidden_state_batch, sarnn_cell_state_batch = self.initial_sarnn_state.get(batch_size)
        self.self_attention.clear()
        next_embedding_batch = embeddings_batch[0]
        next_word_logits_seq_batch = []
        self.iarnn_cell.start()
        self.sarnn_cell.start()
        transcript_attention_map_batch = []

        for t in range(max_report_len - 1):
            # [batch_size, hidden_size], [batch_size, hidden_size]
            iarnn_hidden_state_batch, iarnn_cell_state_batch = self.iarnn_cell(
                # [batch_size, embedding_dim + value_size]
                torch.cat(
                    (
                        # [batch_size, embedding_dim]
                        next_embedding_batch,
                        # [batch_size, value_size]
                        self.image_attention(
                            raw_query_batch=iarnn_hidden_state_batch
                        )

                    ), dim=-1
                ),
                (iarnn_hidden_state_batch, iarnn_cell_state_batch)
            )
            self.self_attention.append(
                raw_key_batch=iarnn_hidden_state_batch,
                raw_value_batch=iarnn_hidden_state_batch,
                # [batch_size]
                mask_batch=report_len_batch.le(t)
            )
            # [batch_size, hidden_size], [batch_size, hidden_size]
            sarnn_hidden_state_batch, sarnn_cell_state_batch = self.sarnn_cell(
                # [batch_size, hidden_size + value_size]
                torch.cat(
                    (
                        # [batch_size, hidden_size]
                        iarnn_hidden_state_batch,
                        # [batch_size, value_size]
                        self.self_attention(
                            raw_query_batch=sarnn_hidden_state_batch
                        )
                    ), dim=-1
                ),
                (sarnn_hidden_state_batch, sarnn_cell_state_batch)
            )

            # if not self.training:
            #     # [batch_size, 1, t + 1]
            #     scores_batch = self.transcript_attention.get_scores_batch().detach().cpu().numpy()
            #     # [batch_size, 1, max_report_len - 1]
            #     scores_batch = np.concatenate(
            #         (
            #             # [batch_size, 1, t + 1]
            #             scores_batch,
            #             # [batch_size, 1, max_report_len - 2 - t]
            #             np.zeros((batch_size, 1, max_report_len - 2 - t))
            #         ), axis=-1
            #     )
            #     transcript_attention_map_batch.append(scores_batch)

            # [batch_size, vocab_size]
            next_word_logits_batch = self.classifier(
                # self.mlp(sarnn_hidden_state_batch)
                sarnn_hidden_state_batch
            )

            if configs.uses_lang_model:
                # [batch_size, vocab_size]
                next_word_logits_batch = self.score_merger(
                    # [batch_size, vocab_size, 2]
                    torch.stack(
                        (
                            # [batch_size, vocab_size]
                            next_word_logits_batch,
                            # [batch_size, vocab_size]
                            self.lang_model.run(next_embedding_batch, initial=(t == 0))
                        ),
                        dim=-1
                    )
                ).view(batch_size, self.vocab.size)

            next_word_logits_seq_batch.append(next_word_logits_batch)

            if self.training and random.random() < configs.sampling_rate:
                if configs.uses_gumbel_softmax:
                    # [batch_size, vocab_size]
                    next_word_probs_batch = F.gumbel_softmax(next_word_logits_batch, tau=.5)
                    # print(next_word_probs_batch[0][transcript_batch[t + 1][0]])
                    # [batch_size, embedding] = [batch_size, vocab_size] @ [vocab_size, embedding_dim]
                    next_embedding_batch = next_word_probs_batch @ self.embedder.weight
                else:
                    # [batch_size, vocab_size]
                    gumbel_noises_batch = self.gumbel_distribution.sample(next_word_logits_batch.shape).to(self.device)
                    # [batch_size]
                    next_word_id_batch = torch.argmax(next_word_logits_batch.detach() + gumbel_noises_batch, dim=-1)
                    # [batch_size, embedding_dim]
                    next_embedding_batch = self.embedder(next_word_id_batch.cpu()).to(self.device)
            else:
                next_embedding_batch = embeddings_batch[t + 1]

        # [max_report_len - 1, batch_size, vocab_size]
        next_word_logits_seq_batch = torch.stack(
            # (max_report_len - 1) * [batch_size, vocab_size]
            next_word_logits_seq_batch,
            dim=0
        )

        # if self.training:
        return next_word_logits_seq_batch
        # else:
        #     # [batch_size, max_report_len - 1, max_report_len - 1]
        #     transcript_attention_map_batch = np.concatenate(
        #         # (max_report_len - 1) * [batch_size, 1, max_report_len - 1]
        #         transcript_attention_map_batch,
        #         axis=1
        #     )
        #
        #     return next_word_logits_seq_batch, transcript_attention_map_batch

    def decode(
            self,
            # [batch_size, feature_num, map_height, map_weight]
            image_feature_maps_batch,
            beam_width=configs.beam_width,
            per_node_beam_width=configs.per_node_beam_width,
            max_len=300
    ):
        # [batch_size, map_height * map_weight, feature_num]
        image_feature_maps_batch = image_feature_maps_batch.view(
            # [batch_size, feature_num, map_height * map_weight]
            *image_feature_maps_batch.shape[:2], -1
        ).transpose(1, 2).to(self.device)
        batch_size, *_ = image_feature_maps_batch.shape

        best_seq_batch = []
        # [vocab_size]
        end_word_log_probs = torch.full((self.vocab.size,), -1e30).to(self.device)
        # [vocab_size]
        end_word_log_probs[self.vocab.end_id] = 0.
        # [beam_width, vocab_size]
        end_word_log_probs_beam = end_word_log_probs.view(1, -1).repeat(beam_width, 1).to(self.device)

        for idx_in_batch in range(batch_size):
            # [1, map_height * map_weight, feature_num]
            image_feature_maps = image_feature_maps_batch[idx_in_batch]

            self.image_attention.set(
                # [1, map_height * map_weight, feature_num]
                raw_keys_batch=image_feature_maps.repeat(beam_width, 1, 1),
                # [1, map_height * map_weight, feature_num]
                raw_values_batch=image_feature_maps.repeat(beam_width, 1, 1),
            )

            iarnn_hidden_state_beam, iarnn_cell_state_beam = self.initial_iarnn_state.get(beam_width)
            sarnn_hidden_state_beam, sarnn_cell_state_beam = self.initial_sarnn_state.get(beam_width)

            self.self_attention.clear()
            self.iarnn_cell.start()
            self.sarnn_cell.start()

            # [beam_width]
            next_word_id_beam = torch.full((beam_width,), self.vocab.start_id, dtype=torch.long)
            # [beam_width, embedding_dim]
            next_embedding_beam = self.embedder(
                next_word_id_beam.to(self.device) if configs.uses_gumbel_softmax else next_word_id_beam
            ).to(self.device)
            # [beam_width, seq_len(just 1 now)]
            word_id_seq_beam = next_word_id_beam.view(beam_width, 1)
            # [beam_width]
            word_id_seq_len_beam = torch.full((beam_width,), 0, dtype=torch.long).to(self.device)
            # [beam_width]
            seq_log_prob_beam = torch.zeros(beam_width).to(self.device)

            for t in range(max_len):
                # [beam_width, hidden_size], [beam_width, hidden_size]
                iarnn_hidden_state_beam, iarnn_cell_state_beam = self.iarnn_cell(
                    # [beam_width, embedding_dim + value_size]
                    torch.cat(
                        (
                            # [beam_width, embedding_dim]
                            next_embedding_beam,
                            # [beam_width, value_size]
                            self.image_attention(
                                raw_query_batch=iarnn_hidden_state_beam
                            )

                        ), dim=-1
                    ),
                    (iarnn_hidden_state_beam, iarnn_cell_state_beam)
                )

                end_flag_beam = next_word_id_beam.eq(self.vocab.end_id)

                self.self_attention.append(
                    # [beam_width, hidden_size]
                    raw_key_batch=iarnn_hidden_state_beam,
                    # [beam_width, hidden_size]
                    raw_value_batch=iarnn_hidden_state_beam,
                    # [beam_width]
                    mask_batch=end_flag_beam
                )

                # [beam_width, hidden_size], [beam_width, hidden_size]
                sarnn_hidden_state_beam, sarnn_cell_state_beam = self.sarnn_cell(
                    # [beam_width, hidden_size + value_size]
                    torch.cat(
                        (
                            # [beam_width, hidden_size]
                            iarnn_hidden_state_beam,
                            # [beam_width, value_size]
                            self.self_attention(
                                raw_query_batch=sarnn_hidden_state_beam
                            )
                        ), dim=-1
                    ),
                    (sarnn_hidden_state_beam, sarnn_cell_state_beam)
                )
                # [beam_width, vocab_size]
                next_word_logits_beam = self.classifier(
                    sarnn_hidden_state_beam
                    # self.mlp(sarnn_hidden_state_beam)
                )

                if configs.uses_lang_model:
                    # [beam_width, vocab_size]
                    next_word_logits_beam = self.score_merger(
                        # [beam_width, vocab_size, 2]
                        torch.stack(
                            (
                                # [beam_width, vocab_size]
                                next_word_logits_beam,
                                # [beam_width, vocab_size]
                                self.lang_model.run(next_embedding_beam, initial=(t == 0))
                            ),
                            dim=-1
                        )
                    ).view(beam_width, self.vocab.size)

                # [beam_width, vocab_size]
                end_flags_beam = end_flag_beam.view(-1, 1).repeat(1, self.vocab.size).to(self.device)
                # [beam_width, vocab_size]
                next_word_log_probs_beam = torch.where(
                    # [beam_width, vocab_size]
                    end_flags_beam,
                    # [beam_width, vocab_size]
                    end_word_log_probs_beam,
                    # [beam_width, vocab_size]
                    F.log_softmax(next_word_logits_beam, dim=-1)
                )
                # [beam_width, vocab_size] = [beam_width, 1] + [beam_width, vocab_size]
                next_word_log_probs_beam = seq_log_prob_beam.view(-1, 1) + next_word_log_probs_beam
                # [beam_width, vocab_size]
                word_id_seq_lens_beam = word_id_seq_len_beam.view(-1, 1).repeat(1, self.vocab.size)
                # [beam_width, vocab_size]
                word_id_seq_lens_beam[1 - end_flags_beam] += 1
                # [beam_width, vocab_size] = [beam_width, vocab_size] / [beam_width, vocab_size]
                normalized_next_word_log_probs_beam = next_word_log_probs_beam / word_id_seq_lens_beam.float()
                # [beam_width, per_node_beam_width], [beam_width, per_node_beam_width]
                top_normalized_next_word_log_probs_beam, top_next_word_ids_beam = torch.topk(
                    normalized_next_word_log_probs_beam, per_node_beam_width
                )
                # [beam_width * per_node_beam_width]
                all_top_normalized_next_word_log_probs = top_normalized_next_word_log_probs_beam.view(-1)
                # [beam_width * per_node_beam_width]
                all_top_next_word_ids = top_next_word_ids_beam.view(-1)
                # [beam_width], [beam_width]
                normalized_next_word_log_prob_beam, idx_beam = all_top_normalized_next_word_log_probs.topk(beam_width)
                # [beam_width]
                idx_beam = idx_beam.cpu()
                # idx = idx_in_beam * per_node_beam_width + idx_in_next_beam
                # [beam_width]
                prev_idx_beam = idx_beam / per_node_beam_width
                # [beam_width]
                next_word_id_beam = all_top_next_word_ids[idx_beam].cpu()
                next_embedding_beam = self.embedder(
                    next_word_id_beam.to(self.device) if configs.uses_gumbel_softmax else next_word_id_beam
                ).to(self.device)
                # # [beam_width, seq_len, hidden_size]
                # iarnn_hidden_states_beam = iarnn_hidden_states_beam[prev_idx_beam]
                # [beam_width, hidden_size]
                iarnn_hidden_state_beam = iarnn_hidden_state_beam[prev_idx_beam]
                # [beam_width, hidden_size]
                iarnn_cell_state_beam = iarnn_cell_state_beam[prev_idx_beam]
                self.self_attention = self.self_attention[prev_idx_beam]
                # [beam_width, seq_len - 1]
                word_id_seq_beam = word_id_seq_beam[prev_idx_beam]
                # [beam_width, seq_len]
                word_id_seq_beam = torch.cat(
                    (
                        # [beam_width, seq_len - 1]
                        word_id_seq_beam,
                        # [beam_width, 1]
                        next_word_id_beam.view(beam_width, 1)
                    ), dim=1
                )
                # # [beam_width, seq_len]
                # masks_beam = masks_beam[prev_idx_beam]
                # [beam_width] = [beam_width, vocab_size][[beam_width], [beam_width]]
                seq_log_prob_beam = next_word_log_probs_beam[prev_idx_beam, next_word_id_beam]
                # [beam_width] = [beam_width, vocab_size][[beam_width], [beam_width]]
                word_id_seq_len_beam = word_id_seq_lens_beam[prev_idx_beam, next_word_id_beam]

                if next_word_id_beam.eq(self.vocab.end_id).all() or t == max_len - 1:
                    best_seq = word_id_seq_beam[0].numpy().tolist()

                    if self.vocab.end_id in best_seq:
                        best_seq = best_seq[:best_seq.index(self.vocab.end_id) + 1]
                    else:
                        best_seq.append(self.vocab.end_id)

                    best_seq_batch.append(best_seq)

                    break

        assert len(best_seq_batch) == batch_size

        return best_seq_batch


class HierarchicalDecoder(nn.Module):
    def __init__(self, vocab, device_id):
        super().__init__()

        self.device = torch.device(device_id)
        self.gumbel_distribution = torch.distributions.Gumbel(0, 1)

        self.vocab = vocab
        self.embedder = nn.Embedding(
            embedding_dim=configs.word_embedding_dim,
            num_embeddings=self.vocab.size,
            padding_idx=self.vocab.padding_id
        )

        self.topic_generator = WeightDroppedRnnCell(
            nn.LSTMCell(
                input_size=(configs.value_size),
                hidden_size=configs.rnn_hidden_size
            )
        ).to(self.device)

        self.end_predictor = nn.Sequential(
            nn.Linear(configs.rnn_hidden_size * 2, configs.rnn_hidden_size),
            nn.Tanh(),
            nn.Linear(configs.rnn_hidden_size, 2)
        ).to(self.device)

        self.initial_topic_state = LearnableRnnInitialState(
            state_shape=(1, configs.rnn_hidden_size)
        ).to(self.device)
        self.image_attention = ScaledDotProdAttention(
            raw_key_size=configs.char_feature_num,
            raw_value_size=configs.char_feature_num
        ).to(self.device)
        self.word_generator = WeightDroppedRnnCell(
            nn.LSTMCell(
                input_size=(configs.word_embedding_dim),  # + configs.value_size),
                hidden_size=configs.rnn_hidden_size
            )
        ).to(self.device)
        # self.self_attention = ScaledDotProdAttention().to(self.device)

        # self.mlp = nn.Sequential(
        #     nn.Linear(configs.rnn_hidden_size, configs.rnn_hidden_size),
        #     nn.BatchNorm1d(configs.rnn_hidden_size),
        #     nn.SELU(),
        #     nn.Dropout(configs.dropout_prob),
        #     nn.Linear(configs.rnn_hidden_size, configs.rnn_hidden_size),
        #     nn.BatchNorm1d(configs.rnn_hidden_size),
        #     nn.SELU()
        # ).to(self.device)
        self.classifier = nn.Linear(configs.rnn_hidden_size, self.vocab.size).to(self.device)

    def forward(
            self,
            # [max_sent_num, max_sent_len, batch_size]
            report_batch,
            # [batch_size]
            sent_num_batch,
            # [max_sent_num, batch_size]
            sent_lens_batch,
            # [batch_size, feature_num, map_height, map_weight]
            image_feature_maps_batch
    ):
        # [batch_size, map_height * map_weight, feature_num]
        image_feature_maps_batch = image_feature_maps_batch.view(
            # [batch_size, feature_num, map_height * map_weight]
            *image_feature_maps_batch.shape[:2], -1
        ).transpose(1, 2).to(self.device)

        self.image_attention.set(
            # [batch_size, max_encoded_utterance, hidden_size]
            raw_keys_batch=image_feature_maps_batch,
            # [batch_size, max_encoded_utterance, hidden_size]
            raw_values_batch=image_feature_maps_batch
        )

        max_sent_num, max_sent_len, batch_size = report_batch.shape

        topic_hidden_state_batch, topic_cell_state_batch = self.initial_topic_state.get(batch_size)
        self.topic_generator.start()
        next_word_logits_seqs_batch = []
        end_scores_seq_batch = []

        for i in range(max_sent_num):
            # [batch_size]
            sent_len_batch = sent_lens_batch[i]
            # [max_sent_len, batch_size, embedding_dim]
            embedding_seq_batch = self.embedder(report_batch[i]).to(self.device)
            embedding_seq_batch = F.dropout(embedding_seq_batch, p=configs.dropout_prob, training=self.training)
            image_context_batch = self.image_attention(
                raw_query_batch=topic_hidden_state_batch
            )
            last_topic_hidden_state_batch = topic_hidden_state_batch

            topic_hidden_state_batch, topic_cell_state_batch = self.topic_generator(
                image_context_batch,
                (topic_hidden_state_batch, topic_cell_state_batch)
            )
            # [batch_size, 2]
            end_scores_batch = self.end_predictor(
                torch.cat(
                    (last_topic_hidden_state_batch, topic_hidden_state_batch),
                    dim=-1
                )
            )
            end_scores_seq_batch.append(end_scores_batch)

            # self.self_attention.clear()

            self.word_generator.start()

            word_hidden_state_batch, word_cell_state_batch = self.word_generator(
                # [batch_size, hidden_size + context_size]
                # torch.cat(
                #     (
                #         # [batch_size, hidden_size]
                #         topic_hidden_state_batch,
                #         # [batch_size, context_size]
                #         image_context_batch
                #
                #     ), dim=-1
                # ),
                topic_hidden_state_batch,
                None
            )

            # self.self_attention.append(
            #     raw_key_batch=word_hidden_state_batch,
            #     raw_value_batch=word_hidden_state_batch,
            #     # [batch_size]
            #     mask_batch=sent_len_batch.le(-1)
            # )

            next_embedding_batch = embedding_seq_batch[0]
            next_word_logits_seq_batch = []

            for t in range(max_sent_len - 1):
                # [batch_size, hidden_size], [batch_size, hidden_size]
                word_hidden_state_batch, word_cell_state_batch = self.word_generator(
                    next_embedding_batch,
                    # [batch_size, embedding_dim + value_size]
                    # torch.cat(
                    #     (
                    #         # [batch_size, embedding_dim]
                    #         next_embedding_batch,
                    #         # [batch_size, value_size]
                    #         self.image_attention(
                    #             raw_query_batch=word_hidden_state_batch
                    #         )
                    #
                    #     ), dim=-1
                    # ),
                    (word_hidden_state_batch, word_cell_state_batch)
                )
                # self.self_attention.append(
                #     raw_key_batch=word_hidden_state_batch,
                #     raw_value_batch=word_hidden_state_batch,
                #     # [batch_size]
                #     mask_batch=report_len_batch.le(t - 1)
                # )

                # [batch_size, vocab_size]
                next_word_logits_batch = self.classifier(
                    # self.mlp(sarnn_hidden_state_batch)
                    word_hidden_state_batch
                )

                next_word_logits_seq_batch.append(next_word_logits_batch)

                if self.training and random.random() < configs.sampling_rate:
                    # [batch_size, vocab_size]
                    gumbel_noises_batch = self.gumbel_distribution.sample(next_word_logits_batch.shape).to(self.device)
                    # [batch_size]
                    next_word_id_batch = torch.argmax(next_word_logits_batch.detach() + gumbel_noises_batch, dim=-1)
                    # [batch_size, embedding_dim]
                    next_embedding_batch = self.embedder(next_word_id_batch.cpu()).to(self.device)
                else:
                    next_embedding_batch = embedding_seq_batch[t + 1]

            # [max_sent_len - 1, batch_size, vocab_size]
            next_word_logits_seq_batch = torch.stack(
                # (max_report_len - 1) * [batch_size, vocab_size]
                next_word_logits_seq_batch,
                dim=0
            )

            next_word_logits_seqs_batch.append(next_word_logits_seq_batch)

        # [max_sent_num, max_sent_len - 1, batch_size, vocab_size]
        next_word_logits_seqs_batch = torch.stack(
            # max_sent_num * [max_sent_len - 1, batch_size, vocab_size]
            next_word_logits_seqs_batch,
            dim=0
        )

        # [batch_size, max_sent_num, 2]
        end_scores_seq_batch = torch.stack(
            # max_sent_num * [batch_size, 2]
            end_scores_seq_batch,
            dim=1
        )

        # [max_sent_num, max_sent_len - 1, batch_size, vocab_size], [batch_size, max_sent_num, 2]
        return next_word_logits_seqs_batch, end_scores_seq_batch

    def decode(
            self,
            # # [max_sent_num, max_sent_len, batch_size]
            # report_batch,
            # # [batch_size]
            # sent_num_batch,
            # # [max_sent_num, batch_size]
            # sent_lens_batch,
            # [batch_size, feature_num, map_height, map_weight]
            image_feature_maps_batch,
            max_sent_num=7,
            max_sent_len=40
    ):
        # [batch_size, map_height * map_weight, feature_num]
        image_feature_maps_batch = image_feature_maps_batch.view(
            # [batch_size, feature_num, map_height * map_weight]
            *image_feature_maps_batch.shape[:2], -1
        ).transpose(1, 2).to(self.device)

        batch_size, *_ = image_feature_maps_batch.shape

        self.image_attention.set(
            # [batch_size, max_encoded_utterance, hidden_size]
            raw_keys_batch=image_feature_maps_batch,
            # [batch_size, max_encoded_utterance, hidden_size]
            raw_values_batch=image_feature_maps_batch
        )

        # max_sent_num, max_sent_len, batch_size = report_batch.shape

        topic_hidden_state_batch, topic_cell_state_batch = self.initial_topic_state.get(batch_size)
        self.topic_generator.start()
        next_word_id_seqs_batch = []
        # end_scores_seq_batch = []

        for i in range(max_sent_num):
            # [batch_size]
            # sent_len_batch = sent_lens_batch[i]
            # [max_sent_len, batch_size, embedding_dim]
            # embedding_seq_batch = self.embedder(report_batch[i]).to(self.device)
            # embedding_seq_batch = F.dropout(embedding_seq_batch, p=configs.dropout_prob, training=self.training)
            image_context_batch = self.image_attention(
                raw_query_batch=topic_hidden_state_batch
            )
            last_topic_hidden_state_batch = topic_hidden_state_batch

            topic_hidden_state_batch, topic_cell_state_batch = self.topic_generator(
                image_context_batch,
                (topic_hidden_state_batch, topic_cell_state_batch)
            )
            # [batch_size, 2]
            end_scores_batch = self.end_predictor(
                torch.cat(
                    (last_topic_hidden_state_batch, topic_hidden_state_batch),
                    dim=-1
                )
            )
            # end_scores_seq_batch.append(end_scores_batch)

            # [batch_size]
            end_flag_batch = torch.argmax(end_scores_batch, dim=-1)

            # self.self_attention.clear()

            self.word_generator.start()

            word_hidden_state_batch, word_cell_state_batch = self.word_generator(
                # [batch_size, hidden_size + context_size]
                # torch.cat(
                #     (
                #         # [batch_size, hidden_size]
                #         topic_hidden_state_batch,
                #         # [batch_size, context_size]
                #         image_context_batch
                #
                #     ), dim=-1
                # ),
                topic_hidden_state_batch,
                None
            )

            # self.self_attention.append(
            #     raw_key_batch=word_hidden_state_batch,
            #     raw_value_batch=word_hidden_state_batch,
            #     # [batch_size]
            #     mask_batch=sent_len_batch.le(-1)
            # )

            next_word_id_batch = torch.full((batch_size,), self.vocab.start_id, dtype=torch.long)

            next_embedding_batch = self.embedder(next_word_id_batch).to(self.device)
            next_word_id_seq_batch = []

            for t in range(max_sent_len - 1):
                # [batch_size, hidden_size], [batch_size, hidden_size]
                word_hidden_state_batch, word_cell_state_batch = self.word_generator(
                    next_embedding_batch,
                    # [batch_size, embedding_dim + value_size]
                    # torch.cat(
                    #     (
                    #         # [batch_size, embedding_dim]
                    #         next_embedding_batch,
                    #         # [batch_size, value_size]
                    #         self.image_attention(
                    #             raw_query_batch=word_hidden_state_batch
                    #         )
                    #
                    #     ), dim=-1
                    # ),
                    (word_hidden_state_batch, word_cell_state_batch)
                )
                # self.self_attention.append(
                #     raw_key_batch=word_hidden_state_batch,
                #     raw_value_batch=word_hidden_state_batch,
                #     # [batch_size]
                #     mask_batch=report_len_batch.le(t - 1)
                # )

                # [batch_size, vocab_size]
                next_word_logits_batch = self.classifier(
                    # self.mlp(sarnn_hidden_state_batch)
                    word_hidden_state_batch
                )

                next_word_id_batch = torch.argmax(next_word_logits_batch.detach())
                next_word_id_seq_batch.append(next_word_id_batch)

                if next_word_id_batch.eq(self.vocab.end_id):
                    break

                next_embedding_batch = self.embedder(next_word_id_batch.cpu()).to(self.device)

            # [max_sent_len - 1, batch_size, vocab_size]
            # next_word_id_seq_batch = torch.stack(
            #     # (max_report_len - 1) * [batch_size, vocab_size]
            #     next_word_id_seq_batch,
            #     dim=0
            # )

            next_word_id_seqs_batch.append(next_word_id_seq_batch)

            if end_flag_batch.eq(1).all():
                break

        # [max_sent_num, max_sent_len - 1, batch_size, vocab_size]
        # next_word_id_seqs_batch = torch.stack(
        #     # max_sent_num * [max_sent_len - 1, batch_size, vocab_size]
        #     next_word_id_seqs_batch,
        #     dim=0
        # )

        # [batch_size, max_sent_num, 2]
        # end_scores_seq_batch = torch.stack(
        #     # max_sent_num * [batch_size, 2]
        #     end_scores_seq_batch,
        #     dim=1
        # )

        # [max_sent_num, max_sent_len - 1, batch_size, vocab_size], [batch_size, max_sent_num, 2]
        return next_word_id_seqs_batch  # , end_scores_seq_batch


class Reshaper(nn.Module):
    def __init__(self, *output_shape):
        super().__init__()

        self.output_shape = output_shape

    def forward(self, input: torch.Tensor):
        return input.view(*self.output_shape)


class Normalizer(nn.Module):
    def __init__(self, target_norm=1.):
        super().__init__()
        self.target_norm = target_norm

    def forward(self, input: torch.Tensor):
        return input * self.target_norm / input.norm(p=2, dim=1, keepdim=True)


class Squeezer(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return torch.squeeze(input, dim=self.dim)