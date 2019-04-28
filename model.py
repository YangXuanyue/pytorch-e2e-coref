import configs
from model_utils import *
from modules import *
import data_utils
from functools import cmp_to_key
import time


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_cnn_embedder = CharCnnEmbedder(
            vocab_size=data_utils.char_vocab.size, padding_id=data_utils.char_vocab.padding_id
        ) if configs.uses_char_embeddings else None
        self.elmo_layer_output_mixer = ElmoLayerOutputMixer().cuda()
        self.encoder = DocEncoder(input_size=configs.tot_embedding_dim).cuda()
        self.span_width_embedder = nn.Embedding(
            num_embeddings=configs.max_span_width,
            embedding_dim=configs.span_width_embedding_dim
        )
        self.head_scorer = nn.Sequential(
            nn.Linear(configs.rnn_hidden_size, 1),
            Squeezer(dim=-1)
        )
        self.mention_scorer = nn.Sequential(
            nn.Linear(configs.span_embedding_dim, configs.ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(configs.dropout_prob),
            nn.Linear(configs.ffnn_hidden_size, configs.ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(configs.dropout_prob),
            nn.Linear(configs.ffnn_hidden_size, 1),
            Squeezer(dim=-1)
        )

        self.fast_ant_scoring_mat = nn.Parameter(
            torch.randn(configs.span_embedding_dim, configs.span_embedding_dim, requires_grad=True)
        )
        self.genre_embedder = nn.Embedding(
            num_embeddings=data_utils.genre_num,
            embedding_dim=configs.genre_embedding_dim
        )

        self.speaker_pair_embedder = nn.Embedding(
            num_embeddings=2,
            embedding_dim=configs.speaker_pair_embedding_dim
        )

        self.ant_offset_embedder = nn.Embedding(
            num_embeddings=10,
            embedding_dim=configs.ant_offset_embedding_dim
        )

        self.slow_ant_scorer = nn.Sequential(
            nn.Linear(configs.pair_embedding_dim, configs.ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(configs.dropout_prob),
            nn.Linear(configs.ffnn_hidden_size, configs.ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(configs.dropout_prob),
            nn.Linear(configs.ffnn_hidden_size, 1),
            Squeezer(dim=-1)
        )

        self.attended_span_embedding_gate = nn.Sequential(
            nn.Linear(configs.span_embedding_dim * 2, configs.span_embedding_dim),
            nn.Sigmoid()
        )

        self.init_params()

        # # print(self)

    def init_params(self):
        self.apply(init_params)

    def get_trainable_params(self):
        yield from filter(lambda param: param.requires_grad, self.parameters())

    def embed_spans(self, head_embedding_seq, encoded_doc, start_idxes, end_idxes):
        doc_len, _ = encoded_doc.shape

        start_embeddings, end_embeddings = encoded_doc[start_idxes], encoded_doc[end_idxes]

        # [span_num]
        span_widths = end_idxes - start_idxes + 1

        # [span_num]
        span_width_ids = span_widths - 1  # [k]

        # [span_num, span_width_embedding_dim]
        span_width_embeddings = F.dropout(
            self.span_width_embedder(span_width_ids.cuda()),
            p=configs.dropout_prob, training=self.training
        )

        # [span_num, max_span_width]
        idxes_of_spans = torch.clamp(
            torch.arange(configs.max_span_width).view(1, -1) + start_idxes.view(-1, 1),
            max=(doc_len - 1)
        )

        # [span_num, max_span_width, span_width_embedding_dim]
        embeddings_of_spans = head_embedding_seq[idxes_of_spans]

        # [doc_len]
        self.head_scores = self.head_scorer(encoded_doc)

        # [span_num, max_span_width]
        head_scores_of_spans = self.head_scores[idxes_of_spans]

        # [span_num, max_span_width]
        span_masks = build_len_mask_batch(span_widths, configs.max_span_width).view(-1, configs.max_span_width)
        # [span_num, max_span_width]
        head_scores_of_spans += torch.log(span_masks.float()).cuda()
        # [span_num, max_span_width, 1]
        attentions_of_spans = F.softmax(head_scores_of_spans, dim=1).view(-1, configs.max_span_width, 1)

        # [span_num, span_width_embedding_dim]
        attended_head_embeddings = (attentions_of_spans * embeddings_of_spans).sum(dim=1)

        # [span_num, span_embedding_dim]
        span_embeddings = torch.cat(
            (
                start_embeddings, end_embeddings, span_width_embeddings, attended_head_embeddings
            ), dim=-1
        )

        return span_embeddings

    @staticmethod
    def extract_top_spans(
        # [cand_num]
        span_scores,
        # [cand_num]
        cand_start_idxes,
        # [cand_num]
        cand_end_idxes,
        top_span_num,
    ):
        span_num, = span_scores.shape

        sorted_span_idxes = torch.argsort(span_scores, descending=True).tolist()

        top_span_idxes = []
        end_idx_to_min_start_dix, start_idx_to_max_end_idx = {}, {}
        selected_span_num = 0

        # while selected_span_num < top_span_num and curr_span_idx < span_num:
        #     i = sorted_span_idxes[curr_span_idx]

        for span_idx in sorted_span_idxes:
            crossed = False
            start_idx = cand_start_idxes[span_idx]
            end_idx = cand_end_idxes[span_idx]

            if end_idx == start_idx_to_max_end_idx.get(start_idx, -1):
                continue

            for j in range(start_idx, end_idx + 1):
                if j in start_idx_to_max_end_idx and j > start_idx and start_idx_to_max_end_idx[j] > end_idx:
                    crossed = True
                    break

                if j in end_idx_to_min_start_dix and j < end_idx and end_idx_to_min_start_dix[j] < start_idx:
                    crossed = True
                    break

            if not crossed:
                top_span_idxes.append(span_idx)
                selected_span_num += 1

                if start_idx not in start_idx_to_max_end_idx or end_idx > start_idx_to_max_end_idx[start_idx]:
                    start_idx_to_max_end_idx[start_idx] = end_idx

                if end_idx not in end_idx_to_min_start_dix or start_idx < end_idx_to_min_start_dix[end_idx]:
                    end_idx_to_min_start_dix[end_idx] = start_idx

            if selected_span_num == top_span_num:
                break

        def compare_span_idxes(i1, i2):
            if cand_start_idxes[i1] < cand_start_idxes[i2]:
                return -1
            elif cand_start_idxes[i1] > cand_start_idxes[i2]:
                return 1
            elif cand_end_idxes[i1] < cand_end_idxes[i2]:
                return -1
            elif cand_end_idxes[i1] > cand_end_idxes[i2]:
                return 1
            # elif i1 < i2:
            #     return -1
            # elif i1 > i2:
            #     return 1
            else:
                return 0

        top_span_idxes.sort(key=cmp_to_key(compare_span_idxes))

        # for span_idx in range(1, len(top_span_idxes)):
        #     assert compare_span_idxes(span_idx - 1, span_idx) == -1

        # last_end_idx = -1
        #
        # for i in range(len(top_span_idxes)):
        #     span_idx = top_span_idxes[i]
        #     start_idx, end_idx = cand_start_idxes[span_idx], cand_end_idxes[span_idx]
        #
        #     assert start_idx <= end_idx
        #
        #     if i:
        #         assert start_idx > last_end_idx
        #
        #     last_end_idx = end_idx

        return torch.as_tensor(
            top_span_idxes + [top_span_idxes[0]] * (top_span_num - selected_span_num)
        )

    def forward(
        self,
        # [sent_num, max_sent_len, glove_embedding_dim]
        glove_embedding_seq_batch,
        # [sent_num, max_sent_len, raw_head_embedding_dim]
        head_embedding_seq_batch,
        # [sent_num, max_sent_len, elmo_embedding_dim, elmo_layer_num]
        elmo_layer_outputs_batch,
        # [sent_num, max_sent_len, max_word_len]
        char_ids_seq_batch,
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
    ):
        start_time = time.time()

        sent_num, max_sent_len, *_ = char_ids_seq_batch.shape

        char_embedding_seq_batch = self.char_cnn_embedder(char_ids_seq_batch) \
            if configs.uses_char_embeddings else None
        elmo_embedding_seq_batch = self.elmo_layer_output_mixer(elmo_layer_outputs_batch)

        embedding_seq_batches = []

        if configs.uses_glove_embeddings:
            embedding_seq_batches.append(glove_embedding_seq_batch)

        head_embedding_seq_batches = [head_embedding_seq_batch]

        if configs.uses_char_embeddings:
            embedding_seq_batches.append(char_embedding_seq_batch)
            head_embedding_seq_batches.append(char_embedding_seq_batch)

        embedding_seq_batches.append(elmo_embedding_seq_batch)

        # [sent_num, max_sent_len, tot_embedding_dim]
        word_embedding_seq_batch = torch.cat(embedding_seq_batches, dim=-1)
        # [sent_num, max_sent_len, head_embedding_dim]
        head_embedding_seq_batch = torch.cat(head_embedding_seq_batches, dim=-1)
        # [sent_num, max_sent_len, tot_embedding_dim]
        word_embedding_seq_batch = F.dropout(
            word_embedding_seq_batch, p=configs.embedding_dropout_prob, training=self.training
        )
        # [sent_num, max_sent_len, head_embedding_dim]
        head_embedding_seq_batch = F.dropout(
            head_embedding_seq_batch, p=configs.embedding_dropout_prob, training=self.training
        )

        # try:
        # [sent_num, max_sent_len]
        len_mask_batch = build_len_mask_batch(sent_len_batch, max_sent_len)
        # except:
        #     breakpoint()

        # [doc_len, hidden_size]
        encoded_doc = self.encoder(word_embedding_seq_batch, sent_len_batch, len_mask_batch)

        doc_len, _ = encoded_doc.shape

        assert doc_len == sent_len_batch.sum().item()

        # [doc_len, head_emb]
        head_embedding_seq = head_embedding_seq_batch[len_mask_batch]

        # [cand_num, span_embedding_dim]
        cand_span_embeddings = self.embed_spans(
            head_embedding_seq, encoded_doc,
            cand_start_idxes, cand_end_idxes
        )

        # [cand_num]
        cand_mention_scores = self.mention_scorer(cand_span_embeddings)

        top_cand_num = int(doc_len * configs.top_span_ratio)

        # print('extracting top spans')

        # [top_cand_num]
        top_span_idxes = self.extract_top_spans(
            # [cand_num]
            cand_mention_scores,
            # [cand_num]
            cand_start_idxes,
            # [cand_num]
            cand_end_idxes,
            top_cand_num,
        )

        top_start_idxes = cand_start_idxes[top_span_idxes]
        top_end_idxes = cand_end_idxes[top_span_idxes]
        # [top_cand_num, span_embedding_dim]
        top_span_embeddings = cand_span_embeddings[top_span_idxes]
        # [top_cand_num]
        top_span_cluster_ids = cand_cluster_ids[top_span_idxes]
        # [top_cand_num]
        top_span_mention_scores = cand_mention_scores[top_span_idxes]

        top_span_sent_idxes = cand_sent_idxes[top_span_idxes]

        # try:
        # [top_cand_num]
        top_span_speaker_ids = speaker_ids[top_start_idxes]
        # except:
        #     breakpoint()

        pruned_ant_num = min(configs.max_ant_num, top_cand_num)

        # print('pruning ants')

        (
            # [top_span_num, pruned_ant_num], [top_span_num, pruned_ant_num]
            top_ant_idxes_of_spans, top_ant_mask_of_spans,
            # [top_span_num, pruned_ant_num], [top_span_num, pruned_ant_num]
            top_fast_ant_scores_of_spans, top_ant_offsets_of_spans
        ) = self.prune(
            # [top_cand_num, span_embedding_dim]
            top_span_embeddings,
            # [top_cand_num]
            top_span_mention_scores,
            pruned_ant_num
        )

        # # print(top_fast_ant_scores_of_spans.device)

        # top_fast_ant_scores_of_spans = top_fast_ant_scores_of_spans.to(torch.device(1))

        # [top_cand_num, 1]
        dummy_scores = torch.zeros(top_cand_num, 1).cuda()
        # top_span_embeddings = top_span_embeddings.to(torch.device(1))

        top_ant_scores_of_spans = None

        # [genre_embedding_dim]
        genre_embedding = self.genre_embedder(genre_id.view(1, 1).cuda()).view(-1)

        for i in range(configs.coref_depth):
            # for i in range(1):
            # print(f'depth {i}')

            # [top_span_num, pruned_ant_num, span_embedding_dim]
            top_ant_embeddings_of_spans = top_span_embeddings[top_ant_idxes_of_spans]
            # [top_cand_num, pruned_ant_num]
            top_ant_scores_of_spans = top_fast_ant_scores_of_spans
            top_ant_scores_of_spans += self.get_slow_ant_scores_of_spans(
                # [top_cand_num, span_embedding_dim]
                top_span_embeddings,
                # [top_span_num, pruned_ant_num]
                top_ant_idxes_of_spans,
                # [top_span_num, pruned_ant_num, span_embedding_dim]
                top_ant_embeddings_of_spans,
                # [top_span_num, pruned_ant_num]
                top_ant_offsets_of_spans,
                # [top_cand_num]
                top_span_speaker_ids,
                # [genre_embedding_dim]
                genre_embedding
            )

            if i == configs.coref_depth - 1:
                break

            # [top_cand_num, 1 + pruned_ant_num]
            top_ant_attentions_of_spans = F.softmax(
                # [top_cand_num, 1 + pruned_ant_num]
                torch.cat(
                    (dummy_scores, top_ant_scores_of_spans), dim=1
                ), dim=-1
            )
            # [top_cand_num, 1 + pruned_ant_num, span_embedding_dim]
            top_ant_embeddings_of_spans = torch.cat(
                (top_span_embeddings.view(top_cand_num, 1, -1), top_ant_embeddings_of_spans), dim=1
            )
            # [top_cand_num, span_embedding_dim]
            attended_top_span_embeddings = \
                (
                    # [top_cand_num, 1 + pruned_ant_num, 1]
                    top_ant_attentions_of_spans.view(top_cand_num, -1, 1)
                    # [top_cand_num, 1 + pruned_ant_num, span_embedding_dim]
                    * top_ant_embeddings_of_spans
                ).sum(1)

            # [top_cand_num, span_embedding_dim]
            g = self.attended_span_embedding_gate(
                # [top_cand_num, span_embedding_dim + span_embedding_dim]
                torch.cat(
                    (top_span_embeddings, attended_top_span_embeddings), dim=1
                )
            )

            top_span_embeddings = g * attended_top_span_embeddings + (1. - g) * top_span_embeddings
            # top_span_embeddings = attended_top_span_embeddings

        # [top_cand_num, 1 + pruned_ant_num]
        top_ant_scores_of_spans = torch.cat(
            (
                # [top_cand_num, 1]
                dummy_scores,
                # [top_cand_num, pruned_ant_num]
                top_ant_scores_of_spans

                # # [top_span_num, pruned_ant_num]
                # top_fast_ant_scores_of_spans
            ), dim=1
        )

        # [top_cand_num, pruned_ant_num]
        top_ant_cluster_ids_of_spans = top_span_cluster_ids[top_ant_idxes_of_spans]

        return (
            # [cand_num]
            cand_mention_scores,
            # [top_cand_num]
            top_start_idxes,
            # [top_cand_num]
            top_end_idxes,
            # [top_cand_num]
            top_span_cluster_ids,
            # [top_span_num, pruned_ant_num]
            top_ant_idxes_of_spans,
            # [top_cand_num, pruned_ant_num]
            top_ant_cluster_ids_of_spans,
            # [top_cand_num, 1 + pruned_ant_num]
            top_ant_scores_of_spans,
            # [top_span_num, pruned_ant_num]
            top_ant_mask_of_spans
        )

    def get_slow_ant_scores_of_spans(
        self,
        # [top_cand_num, span_embedding_dim]
        top_span_embeddings,
        # [top_span_num, pruned_ant_num]
        top_ant_idxes_of_spans,
        # [top_span_num, pruned_ant_num, span_embedding_dim]
        top_ant_embeddings_of_spans,
        # [top_span_num, pruned_ant_num]
        top_ant_offsets_of_spans,
        # [top_cand_num]
        top_span_speaker_ids,
        # [genre_embedding_dim]
        genre_embedding
    ):
        top_span_num, pruned_ant_num = top_ant_idxes_of_spans.shape
        # [top_span_num, pruned_ant_num]
        top_ant_speaker_ids_of_spans = top_span_speaker_ids[top_ant_idxes_of_spans]
        # [top_span_num, pruned_ant_num, speaker_pair_embedding_dim]
        speaker_pair_embeddings_of_spans = self.speaker_pair_embedder(
            # [top_span_num, pruned_ant_num]
            (top_span_speaker_ids.view(-1, 1) == top_ant_speaker_ids_of_spans).long().cuda()
        )
        # [top_span_num, pruned_ant_num, ant_offset_embedding_dim]
        ant_offset_embeddings_of_spans = self.ant_offset_embedder(
            self.get_offset_bucket_idxes_batch(top_ant_offsets_of_spans).cuda()
        )
        feature_embeddings_of_spans = torch.cat(
            (
                speaker_pair_embeddings_of_spans,
                # [top_span_num, pruned_ant_num, feature_size]
                genre_embedding.view(1, 1, -1).repeat(top_span_num, pruned_ant_num, 1),
                ant_offset_embeddings_of_spans
            ), dim=-1
        )
        # [top_span_num, pruned_ant_num, feature_size * 3]
        feature_embeddings_of_spans = F.dropout(
            feature_embeddings_of_spans, p=configs.dropout_prob, training=self.training
        )
        # [top_span_num, pruned_ant_num, span_embedding_dim] * [top_cand_num, 1, span_embedding_dim]
        similarity_embeddings_of_spans = top_ant_embeddings_of_spans \
                                         * top_span_embeddings.view(top_span_num, 1, -1)

        pair_embeddings_of_spans = torch.cat(
            (
                # [top_span_num, pruned_ant_num, span_embedding_dim]
                top_span_embeddings.view(top_span_num, 1, -1).expand(-1, pruned_ant_num, -1),
                # [top_span_num, pruned_ant_num, span_embedding_dim]
                top_ant_embeddings_of_spans,
                # [top_span_num, pruned_ant_num, span_embedding_dim]
                similarity_embeddings_of_spans,
                # [top_span_num, pruned_ant_num, feature_size * 3]
                feature_embeddings_of_spans
            ), dim=-1
        )

        # print(pair_embeddings_of_spans.shape)
        # [top_span_num, pruned_ant_num]
        slow_ant_scores_of_spans = self.slow_ant_scorer(pair_embeddings_of_spans.cuda()).cuda()
        # [top_span_num, pruned_ant_num]
        return slow_ant_scores_of_spans

    def prune(
        self,
        # [top_cand_num, span_embedding_dim]
        top_span_embeddings,
        # [top_cand_num]
        top_span_mention_scores,
        pruned_ant_num
    ):
        top_span_num, _ = top_span_embeddings.shape

        span_idxes = torch.arange(top_span_num)
        # [top_span_num, top_span_num]
        ant_offsets_of_spans = span_idxes.view(-1, 1) - span_idxes.view(1, -1)
        # [top_span_num, top_span_num]
        ants_mask_of_spans = ant_offsets_of_spans >= 1
        # [top_span_num, top_span_num]
        fast_ant_scores_of_spans = top_span_mention_scores.view(-1, 1) + top_span_mention_scores.view(1, -1)
        # fast_ant_scores_of_spans = fast_ant_scores_of_spans.cuda(1)
        fast_ant_scores_of_spans += torch.log(ants_mask_of_spans.float()).cuda()
        fast_ant_scores_of_spans += self.get_fast_ant_scores_of_spans(top_span_embeddings)

        # [top_span_num, pruned_ant_num]
        _, top_ant_idxes_of_spans = torch.topk(
            # [top_span_num, top_span_num]
            fast_ant_scores_of_spans, k=pruned_ant_num, dim=-1, sorted=False
        )
        top_ant_idxes_of_spans = top_ant_idxes_of_spans.cpu()
        # [top_span_num, 1]
        span_idxes = span_idxes.view(-1, 1)
        # [top_span_num, pruned_ant_num]
        top_ant_mask_of_spans = ants_mask_of_spans[span_idxes, top_ant_idxes_of_spans]
        # [top_span_num, pruned_ant_num]
        top_fast_ant_scores_of_spans = fast_ant_scores_of_spans[span_idxes, top_ant_idxes_of_spans]
        # [top_span_num, pruned_ant_num]
        top_ant_offsets_of_spans = ant_offsets_of_spans[span_idxes, top_ant_idxes_of_spans]

        return (
            # [top_span_num, pruned_ant_num], [top_span_num, pruned_ant_num]
            top_ant_idxes_of_spans, top_ant_mask_of_spans,
            # [top_span_num, pruned_ant_num], [top_span_num, pruned_ant_num]
            top_fast_ant_scores_of_spans, top_ant_offsets_of_spans
        )

    def get_fast_ant_scores_of_spans(
        self,
        # [top_cand_num, span_embedding_dim]
        top_span_embeddings
    ):
        # # print(top_span_embeddings.shape)
        # top_span_embeddings = top_span_embeddings.cuda(1)
        # [top_cand_num, span_embedding_dim]
        top_src_span_embeddings = F.dropout(top_span_embeddings, p=configs.dropout_prob, training=self.training)
        # [span_embedding_dim, top_cand_num]
        top_tgt_span_embeddings = F.dropout(top_span_embeddings.t(), p=configs.dropout_prob, training=self.training)
        # [top_span_num, top_span_num]
        return top_src_span_embeddings @ self.fast_ant_scoring_mat @ top_tgt_span_embeddings

        # # [top_span_num, top_span_num]
        # return (
        #         F.dropout(
        #             top_span_embeddings @ self.fast_ant_scoring_mat,
        #
        #         ) @ F.dropout(top_span_embeddings, p=configs.dropout_prob, training=self.training).t()
        # ).cuda()

    def get_offset_bucket_idxes_batch(self, offsets_batch):
        """
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        log_space_idxes_batch = (torch.log(offsets_batch.float()) / math.log(2)).floor().long() + 3

        identity_mask_batch = (offsets_batch <= 4).long()

        return torch.clamp(
            identity_mask_batch * offsets_batch + (1 - identity_mask_batch) * log_space_idxes_batch,
            min=0, max=9
        )

    def compute_loss(self, *input_tensors):
        start_time = time.time()

        (
            # [cand_num]
            cand_mention_scores,
            # [top_cand_num]
            top_start_idxes,
            # [top_cand_num]
            top_end_idxes,
            # [top_cand_num]
            top_span_cluster_ids,
            # [top_span_num, pruned_ant_num]
            top_ant_idxes_of_spans,
            # [top_cand_num, pruned_ant_num]
            top_ant_cluster_ids_of_spans,
            # [top_cand_num, 1 + pruned_ant_num]
            top_ant_scores_of_spans,
            # [top_span_num, pruned_ant_num]
            top_ant_mask_of_spans
        ) = self(*input_tensors)

        # print(f'forward: {time.time() - start_time}')

        top_ant_cluster_ids_of_spans += torch.log(top_ant_mask_of_spans.float()).long()
        # [top_cand_num, pruned_ant_num]
        ant_indicators_of_spans = top_ant_cluster_ids_of_spans == top_span_cluster_ids.view(-1, 1)
        # [top_cand_num, 1]
        non_dummy_span_mask = (top_span_cluster_ids > 0).view(-1, 1)
        # [top_cand_num, pruned_ant_num]
        non_dummy_ant_indicators_of_spans = ant_indicators_of_spans & non_dummy_span_mask
        # [top_cand_num, 1]
        dummy_span_indicators = ~non_dummy_ant_indicators_of_spans.any(dim=1, keepdim=True)
        # [top_cand_num, 1 + pruned_ant_num]
        ant_indicators_of_spans = torch.cat(
            (dummy_span_indicators, non_dummy_ant_indicators_of_spans), dim=1
        )
        # [top_cand_num]
        log_marginalized_prob_of_spans = (
            torch.logsumexp(
                top_ant_scores_of_spans + torch.log(ant_indicators_of_spans.float()).cuda(),
                dim=1
            ) - torch.logsumexp(top_ant_scores_of_spans, dim=1)
        )

        return -log_marginalized_prob_of_spans.sum()

    def predict(self, *input_tensors):
        (
            # [cand_num]
            cand_mention_scores,
            # [top_cand_num]
            top_start_idxes,
            # [top_cand_num]
            top_end_idxes,
            # [top_cand_num]
            top_span_cluster_ids,
            # [top_span_num, pruned_ant_num]
            top_ant_idxes_of_spans,
            # [top_cand_num, pruned_ant_num]
            top_ant_cluster_ids_of_spans,
            # [top_cand_num, 1 + pruned_ant_num]
            top_ant_scores_of_spans,
            # [top_span_num, pruned_ant_num]
            top_ant_mask_of_spans
        ) = self(*input_tensors)

        predicted_ant_idxes = []

        for span_idx, loc in enumerate(torch.argmax(top_ant_scores_of_spans, dim=1) - 1):
            if loc < 0:
                predicted_ant_idxes.append(-1)
            else:
                predicted_ant_idxes.append(top_ant_idxes_of_spans[span_idx, loc].item())

        span_to_predicted_cluster_id = {}
        predicted_clusters = []

        for span_idx, ant_idx in enumerate(predicted_ant_idxes):
            if ant_idx < 0:
                continue

            assert span_idx > ant_idx

            ant_span = top_start_idxes[ant_idx].item(), top_end_idxes[ant_idx].item()

            if ant_span in span_to_predicted_cluster_id:
                predicted_cluster_id = span_to_predicted_cluster_id[ant_span]
            else:
                predicted_cluster_id = len(predicted_clusters)
                predicted_clusters.append([ant_span])
                span_to_predicted_cluster_id[ant_span] = predicted_cluster_id

            span = top_start_idxes[span_idx].item(), top_end_idxes[span_idx].item()
            predicted_clusters[predicted_cluster_id].append(span)
            span_to_predicted_cluster_id[span] = predicted_cluster_id

        predicted_clusters = [tuple(cluster) for cluster in predicted_clusters]
        span_to_predicted_cluster = {
            span: predicted_clusters[cluster_id]
            for span, cluster_id in span_to_predicted_cluster_id.items()
        }

        # [top_cand_num], [top_cand_num], [top_cand_num]
        return top_start_idxes, top_end_idxes, predicted_ant_idxes, \
               predicted_clusters, span_to_predicted_cluster
