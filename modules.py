from model_utils import *
from collections import Counter, defaultdict
# import data_utils
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
        # print(layer_outputs_batch.type())

        # [batch_size, sent_len, elmo_embedding_dim]
        embedding_seq_batch = layer_outputs_batch @ F.softmax(self.mix_weights, dim=-1)
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

        # for weight_name in self.weight_names:
        #     print(f'applying weight-drop of {self.dropout_prob} to {weight_name}')
        #     weight_param = getattr(self.rnn, weight_name)
        #     del self.rnn._parameters[weight_name]
        #     self.rnn.register_parameter(weight_name + '_raw', nn.Parameter(weight_param.data))

    def drop(self):
        # for weight_name in self.weight_names:
        #     raw_weight_param = getattr(self.rnn, weight_name + '_raw')
        #     weight_param = F.dropout(raw_weight_param, p=self.dropout_prob, training=self.training)
        #     setattr(self.rnn, weight_name, weight_param)
        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight_hh' in name:
                    param.data = F.dropout(param.data, p=self.dropout_prob, training=self.training)

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
        # packs_input_batch = isinstance(input_batch, rnn_utils.PackedSequence)
        # packs_output_batch = isinstance(output_batch, rnn_utils.PackedSequence)
        # len_batch = None
        #
        #
        if isinstance(input_batch, rnn_utils.PackedSequence):
            input_batch, len_batch = rnn_utils.pad_packed_sequence(input_batch)

        if isinstance(output_batch, rnn_utils.PackedSequence):
            output_batch, len_batch = rnn_utils.pad_packed_sequence(output_batch)

        assert input_batch.shape == output_batch.shape
        *dims, hidden_size = output_batch.shape
        assert hidden_size == self.hidden_size

        input_batch = input_batch.view(-1, self.hidden_size)
        output_batch = output_batch.view(-1, self.hidden_size)
        # [*, hidden_size]
        g = torch.sigmoid(self.projection(output_batch))
        # [..., hidden_size]
        output_batch = (g * output_batch + (1. - g) * input_batch).view(*dims, self.hidden_size)

        # if packs_output_batch:
        #     output_batch = rnn_utils.pack_padded_sequence(output_batch, len_batch)

        return output_batch


from torchnlp.nn import WeightDropLSTM


class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.input_size = input_size
        self.direction_num = 2
        self.hidden_size = configs.rnn_hidden_size // self.direction_num
        self.layer_num = configs.rnn_layer_num
        self.mask_sampler = torch.distributions.bernoulli.Bernoulli(1. - configs.lstm_dropout_prob)

        # weight_names = [name for name, param in self.rnn.named_parameters() if 'weight' in name]

        self.rnns = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=(
                        self.input_size if i == 0
                        else self.hidden_size * self.direction_num
                    ),
                    hidden_size=self.hidden_size,
                    bidirectional=True
                )
                # WeightDroppedRnn(
                #     nn.LSTM(
                #         input_size=(
                #             self.input_size if i == 0
                #             else self.hidden_size * self.direction_num
                #         ),
                #         hidden_size=self.hidden_size,
                #         bidirectional=True
                #     ),
                #     dropout_prob=configs.lstm_dropout_prob
                # )
                # WeightDropLSTM(
                #     input_size=(
                #         self.input_size if i == 0
                #         else self.hidden_size * self.direction_num
                #     ),
                #     hidden_size=self.hidden_size,
                #     bidirectional=True,
                #     weight_dropout=configs.lstm_dropout_prob
                # )
                for i in range(self.layer_num)
            ]
        )
        self.initial_states = nn.ModuleList(
            [
                LearnableRnnInitialState(
                    state_shape=(self.direction_num, 1, self.hidden_size),
                    rnn_type=nn.LSTM
                )
                for _ in range(self.layer_num)
            ]

        )

        self.highway_gates = nn.ModuleList(
            [
                HighwayGate(self.hidden_size * self.direction_num)
                for _ in range(self.layer_num - 1)
            ]

        )

    def forward(
        self,
        # [batch_size, max_sent_len, tot_embedding_dim], [batch_size]
        embedding_seq_batch, sent_len_batch
    ):
        batch_size, max_sent_len, _ = embedding_seq_batch.shape
        sorted_idx_batch = sorted(range(batch_size), key=sent_len_batch.__getitem__, reverse=True)
        orig_idx_batch = [-1] * batch_size

        for sorted_idx, orig_idx in enumerate(sorted_idx_batch):
            orig_idx_batch[orig_idx] = sorted_idx

        # [max_sent_len, batch_size, tot_embedding_dim]
        curr_input_batch = embedding_seq_batch[sorted_idx_batch, ...].transpose(0, 1)

        sent_len_batch = sent_len_batch[sorted_idx_batch]

        # [max_sent_len, batch_size(decreasing), tot_embedding_dim]
        curr_input_batch = rnn_utils.pack_padded_sequence(
            curr_input_batch, sent_len_batch
        )
        curr_output_batch = None

        for i in range(self.layer_num):
            # print(i)

            # try:
            # print(type(curr_input_batch))
            # [max_sent_len, batch_size(decreasing), hidden_size]
            curr_output_batch, *_ = self.rnns[i](
                curr_input_batch, self.initial_states[i].get(batch_size)
            )
            curr_output_batch, sent_len_batch = rnn_utils.pad_packed_sequence(curr_output_batch)
            # print(type(curr_output_batch))

            if self.training:
                # locked dropout
                # [max_sent_len, batch_size, hidden_size]

                mask_batch = self.mask_sampler.sample((1, batch_size, self.hidden_size * self.direction_num))
                mask_batch /= (1. - configs.lstm_dropout_prob)
                curr_output_batch *= mask_batch.cuda()

            if i > 0:
                curr_output_batch = self.highway_gates[i - 1](curr_input_batch, curr_output_batch)

            # [max_sent_len, batch_size(decreasing), hidden_size]
            curr_input_batch = rnn_utils.pack_padded_sequence(curr_output_batch, sent_len_batch)
            # except:
            #     breakpoint()

        # # [max_sent_len, batch_size, hidden_size]
        # curr_output_batch, _ = rnn_utils.pad_packed_sequence(curr_output_batch)

        # try:
        #     assert curr_output_batch.shape[0] == max_sent_len
        #     assert curr_output_batch.shape[1] == batch_size
        # except AssertionError as x:
        #     print(x)
        #     breakpoint()

        # [batch_size, max_sent_len, hidden_size]
        return curr_output_batch.transpose(0, 1)[orig_idx_batch, ...]


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
