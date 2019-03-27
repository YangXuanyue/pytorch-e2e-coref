import configs
from model_utils import *
from modules import *
import data_utils
from functools import cmp_to_key


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        # self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
        # self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)
        self.char_cnn_embedder = CharCnnEmbedder(
            vocab_size=data_utils.char_vocab.size, padding_id=data_utils.char_vocab.padding_id
        )
        self.elmo_layer_output_mixer = ElmoLayerOutputMixer()
        self.genre_embedder = nn.Embedding(
            num_embeddings=data_utils.genre_num,
            embedding_dim=configs.genre_embedding_dim
        )
        self.encoder = Encoder(input_size=configs.tot_embedding_dim)
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

        # with tf.variable_scope("mention_scores"):
        #     return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout)  # [k, 1]

        # -> [k]

        self.attended_span_embedding_gate = nn.Sequential(
            nn.Linear(configs.span_width_embedding_dim * 2, configs.span_width_embedding_dim),
            nn.Sigmoid()
        )

        self.fast_antecedent_scoring_mat = nn.Parameter(
            torch.randn(configs.span_embedding_dim, configs.span_embedding_dim, requires_grad=True)
        )

        # source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)),
        #                                     self.dropout)

        self.speaker_pair_embedder = nn.Embedding(
            num_embeddings=2,
            embedding_dim=configs.speaker_pair_embedding_dim
        )

        self.antecedent_offset_embedder = nn.Embedding(
            num_embeddings=10,
            embedding_dim=configs.antecedent_offset_embedding_dim
        )
        self.slow_antecedent_scorer = nn.Sequential(
            nn.Sequential(configs.pair_embedding_dim, configs.ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(configs.dropout_prob),
            nn.Linear(configs.ffnn_hidden_size, configs.ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(configs.dropout_prob),
            nn.Linear(configs.ffnn_hidden_size, 1),
            Squeezer(dim=-1)
        )

        # slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
        #                                    self.dropout)  # [k, c, 1]

        # slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)  # [k, c]

    def init_params(self):
        self.apply(init_params)

    def get_trainable_params(self):
        yield from filter(lambda param: param.requires_grad, self.model.parameters())

    def embed_spans(self, head_embedding_seq, encoded_doc, start_idxes, end_idxes):
        doc_len, _ = encoded_doc.shape

        start_embeddings, end_embeddings = encoded_doc[start_idxes], encoded_doc[end_idxes]

        # [span_num]
        span_widths = 1 + end_idxes - start_idxes

        # [span_num]
        span_width_ids = span_widths - 1  # [k]

        # [span_num, span_width_embedding_dim]
        span_width_embeddings = F.dropout(
            self.span_width_embedder(span_width_ids),
            p=configs.dropout_prob, training=self.training
        )

        # [span_num, max_span_width]
        idxes_of_spans = torch.min(
            torch.arange(configs.max_span_width).view(1, -1) + start_idxes.view(-1, 1),
            doc_len - 1
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
        head_scores_of_spans += torch.log(span_masks)
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

    def extract_top_spans(
            self,
            span_scores, cand_start_idxes, cand_end_idxes, top_span_num
    ):
        span_num = span_scores.shape

        sorted_span_idxes = torch.argsort(span_scores, descending=True)

        top_span_idxes = []
        end_idx_to_min_start_dix, start_idx_to_max_end_idx = {}, {}
        curr_span_idx, selected_span_num = 0, 0

        while selected_span_num < top_span_num and curr_span_idx < span_num:
            i = sorted_span_idxes[curr_span_idx]
            crossed = False
            start_idx = cand_start_idxes[i]
            end_idx = cand_end_idxes[i]

            for j in range(start_idx, end_idx + 1):
                if j in start_idx_to_max_end_idx and j > start_idx and start_idx_to_max_end_idx[j] > end_idx:
                    crossed = True
                    break

                if j in end_idx_to_min_start_dix and j < end_idx and end_idx_to_min_start_dix[j] < start_idx:
                    crossed = True
                    break

            if not crossed:
                top_span_idxes.append(i)
                selected_span_num += 1

                if start_idx not in start_idx_to_max_end_idx or end_idx > start_idx_to_max_end_idx[start_idx]:
                    start_idx_to_max_end_idx[start_idx] = end_idx

                if end_idx not in end_idx_to_min_start_dix or start_idx < end_idx_to_min_start_dix[end_idx]:
                    end_idx_to_min_start_dix[end_idx] = start_idx

            curr_span_idx += 1

        def compare_span_idxes(i1, i2):
            if cand_start_idxes[i1] < cand_start_idxes[i2]:
                return -1
            elif cand_start_idxes[i1] > cand_start_idxes[i2]:
                return 1
            elif cand_end_idxes[i1] < cand_end_idxes[i2]:
                return -1
            elif cand_end_idxes[i1] > cand_end_idxes[i2]:
                return 1
            elif i1 < i2:
                return -1
            elif i1 > i2:
                return 1
            else:
                return 0

        top_span_idxes.sort(key=cmp_to_key(compare_span_idxes))

        return torch.as_tensor(
            top_span_idxes + [top_span_idxes[0]] * (top_span_num - selected_span_num)
        )

    def forward(
            self, glove_embedding_seq_batch, head_embedding_seq_batch, elmo_layer_outputs_batch,
            char_ids_seq_batch, sent_len_batch, speaker_ids, genre_id,
            gold_start_idxes, gold_end_idxes, gold_cluster_ids,
            cand_start_idxes, cand_end_idxes, cand_cluster_ids, cand_sent_idxes
    ):
        sent_num, max_sent_len, *_ = char_ids_seq_batch.shape

        char_embedding_seq_batch = self.char_cnn_embedder(char_ids_seq_batch)
        elmo_embedding_seq_batch = self.elmo_layer_output_mixer(elmo_layer_outputs_batch)

        word_embedding_seq_batch = torch.cat(
            (
                glove_embedding_seq_batch, char_embedding_seq_batch, elmo_embedding_seq_batch
            ), dim=-1
        )
        head_embedding_seq_batch = torch.cat(
            (
                head_embedding_seq_batch, char_embedding_seq_batch
            ), dim=-1
        )
        word_embedding_seq_batch = F.dropout(word_embedding_seq_batch, p=configs.embedding_dropout_prob,
                                             training=self.training)  # [sent_num, max_sent_len, emb]
        head_embedding_seq_batch = F.dropout(head_embedding_seq_batch, p=configs.embedding_dropout_prob,
                                             training=self.training)  # [sent_num, max_sent_len, emb]

        # [sent_num, max_sent_len]
        len_mask_batch = build_len_mask_batch(sent_len_batch, max_sent_len)

        # [doc_len, hidden_size]
        encoded_doc = self.encoder(word_embedding_seq_batch, sent_len_batch)[len_mask_batch]

        doc_len, _ = encoded_doc.shape

        # [doc_len, head_emb]
        head_embedding_seq = head_embedding_seq_batch[len_mask_batch]

        # [genre_embedding_dim]
        genre_embedding = self.genre_embedder(genre_id.view(1, 1)).view(-1)

        # [cand_num, span_embedding_dim]
        cand_span_embeddings = self.embed_spans(
            head_embedding_seq, encoded_doc,
            cand_start_idxes, cand_end_idxes
        )

        # [cand_num]
        cand_mention_scores = self.mention_scorer(cand_span_embeddings)

        top_cand_num = int(doc_len * configs.top_span_ratio)

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

        top_span_embeddings = cand_span_embeddings[top_span_idxes]

        top_span_cluster_ids = cand_cluster_ids[top_span_idxes]

        top_span_mention_scores = cand_mention_scores[top_span_idxes]

        top_span_sent_idxes = cand_sent_idxes[top_span_idxes]

        top_span_speaker_ids = speaker_ids[top_start_idxes]

        pruned_antecedent_num = min(configs.max_antecedent_num, top_cand_num)

        (
            top_antecedent_idxes_of_spans, top_antecedent_mask_of_spans,
            top_fast_antecedent_scores_of_spans, top_antecedent_offsets_of_spans
        ) = self.prune(top_span_embeddings, top_span_mention_scores, pruned_antecedent_num)

        dummy_scores = torch.zeros(top_cand_num, 1)

        top_antecedent_scores_of_spans = None

        for i in range(configs.coref_depth):
            top_antecedent_embeddings_of_spans = top_span_embeddings[top_antecedent_idxes_of_spans]
            # [top_cand_num, pruned_antecedent_num]
            top_antecedent_scores_of_spans = top_fast_antecedent_scores_of_spans + self.get_slow_antecedent_scores_of_spans(
                top_span_embeddings,
                top_antecedent_idxes_of_spans,
                top_antecedent_embeddings_of_spans,
                top_antecedent_offsets_of_spans,
                top_span_speaker_ids,
                genre_embedding
            )

            # [top_cand_num, 1 + pruned_antecedent_num]
            top_antecedent_attentions_of_spans = F.softmax(
                torch.cat(
                    (dummy_scores, top_antecedent_scores_of_spans), dim=1
                )
            )
            # [top_cand_num, 1 + pruned_antecedent_num, span_embedding_dim]
            top_antecedent_embeddings_of_spans = torch.cat(
                (top_span_embeddings.view(top_cand_num, 1, -1), top_antecedent_embeddings_of_spans), dim=1
            )
            # [top_cand_num, span_embedding_dim]
            attended_top_span_embeddings = \
                (
                        top_antecedent_attentions_of_spans.view(top_cand_num, -1,
                                                                1) * top_antecedent_embeddings_of_spans
                ).sum(1)

            g = self.attended_span_embedding_gate(
                torch.cat(
                    (top_span_embeddings, attended_top_span_embeddings), dim=1
                )
            )

            top_span_embeddings = g * attended_top_span_embeddings + (1. - g) * top_span_embeddings

        # [top_cand_num, 1 + pruned_antecedent_num]
        top_antecedent_scores_of_spans = torch.cat(
            (dummy_scores, top_antecedent_scores_of_spans), dim=1
        )

        # [top_cand_num, pruned_antecedent_num]
        top_antecedent_cluster_ids_of_spans = top_span_cluster_ids[top_antecedent_idxes_of_spans]

        return (
            cand_mention_scores, top_start_idxes, top_end_idxes, top_span_cluster_ids,
            top_antecedent_idxes_of_spans, top_antecedent_cluster_ids_of_spans,
            top_antecedent_scores_of_spans, top_antecedent_mask_of_spans
        )

    def prune(self, top_span_embeddings, top_span_mention_scores, pruned_antecedent_num):
        top_span_num, _ = top_span_embeddings.shape

        span_idxes = torch.arange(top_span_num)
        # [top_span_num, top_span_num]
        antecedent_offsets_of_spans = span_idxes.view(-1, 1) - span_idxes.view(1, -1)
        # [top_span_num, top_span_num]
        antecedents_mask_of_spans = antecedent_offsets_of_spans >= 1
        # [top_span_num, top_span_num]
        fast_antecedent_scores_of_spans = top_span_mention_scores.view(-1, 1) + top_span_mention_scores.view(1, -1)
        fast_antecedent_scores_of_spans += torch.log(antecedents_mask_of_spans.float())
        fast_antecedent_scores_of_spans += self.get_fast_antecedent_scores_of_spans(top_span_embeddings)

        # [top_span_num, pruned_antecedent_num]
        _, top_antecedent_idxes_of_spans = torch.topk(
            fast_antecedent_scores_of_spans, k=pruned_antecedent_num, dim=-1, sorted=False
        )

        span_idxes = span_idxes.view(-1, 1)
        top_antecedent_mask_of_spans = antecedents_mask_of_spans[span_idxes, top_antecedent_idxes_of_spans]
        top_fast_antecedent_scores_of_spans = fast_antecedent_scores_of_spans[span_idxes, top_antecedent_idxes_of_spans]
        top_antecedent_offsets_of_spans = antecedent_offsets_of_spans[span_idxes, top_antecedent_idxes_of_spans]

        return (
            top_antecedent_idxes_of_spans, top_antecedent_mask_of_spans,
            top_fast_antecedent_scores_of_spans, top_antecedent_offsets_of_spans
        )

    def get_slow_antecedent_scores_of_spans(
            self, top_span_embeddings, top_antecedent_idxes_of_spans, top_antecedent_embeddings_of_spans,
            top_antecedent_offsets_of_spans, top_span_speaker_ids, genre_embedding
    ):

        top_span_num, pruned_antecedent_num = top_antecedent_idxes_of_spans.shape

        top_antecedent_speaker_ids_of_spans = top_span_speaker_ids[top_antecedent_idxes_of_spans]

        speaker_pair_embeddings_of_spans = self.speaker_pair_embedder(
            # [top_span_num, pruned_antecedent_num]
            top_span_speaker_ids.view(-1, 1) == top_antecedent_speaker_ids_of_spans
        )

        antecedent_distance_embeddings_of_spans = self.antecedent_offset_embedder(
            self.get_offset_bucket_idxes_batch(top_antecedent_offsets_of_spans)
        )

        feature_embeddings_of_spans = torch.cat(
            (
                speaker_pair_embeddings_of_spans,
                # [top_span_num, pruned_antecedent_num, feature_size]
                genre_embedding.view(1, 1, -1).repeat(top_span_num, pruned_antecedent_num, 1),
                antecedent_distance_embeddings_of_spans
            ), dim=-1
        )
        # [top_span_num, pruned_antecedent_num, feature_size * 3]
        feature_embeddings_of_spans = F.dropout(
            feature_embeddings_of_spans, p=configs.dropout_prob, training=self.training
        )

        similarity_embeddings_of_spans = top_antecedent_embeddings_of_spans \
                                         * top_span_embeddings.view(top_span_num, -1, 1)

        pair_embeddings_of_spans = torch.cat(
            (
                top_span_embeddings.view(top_span_num, 1, -1).expand(1, pruned_antecedent_num, 1),
                top_antecedent_embeddings_of_spans,
                similarity_embeddings_of_spans,
                feature_embeddings_of_spans
            ), dim=-1
        )

        slow_antecedent_scores_of_spans = self.slow_antecedent_scorer(pair_embeddings_of_spans)

        return slow_antecedent_scores_of_spans  # [top_span_num, pruned_antecedent_num]

    def get_fast_antecedent_scores_of_spans(self, top_span_emb):
        # [top_span_num, top_span_num]
        return F.dropout(
            top_span_emb @ self.fast_antecedent_scoring_mat,
            p=configs.dropout_prob, training=self.training
        ) @ F.dropout(top_span_emb, p=configs.dropout_prob, training=self.training).T

    def get_offset_bucket_idxes_batch(self, offsets_batch):
        """
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        log_space_idxes_batch = (torch.log(offsets_batch.float()) / math.log(2)).floor().int() + 3

        identity_mask_batch = (offsets_batch <= 4).int()

        return torch.clamp(
            identity_mask_batch * offsets_batch + (1 - identity_mask_batch) * log_space_idxes_batch,
            min=0, max=9
        )

    def compute_loss(self, *input_tensors):
        (
            cand_mention_scores, top_start_idxes, top_end_idxes, top_span_cluster_ids,
            top_antecedent_idxes_of_spans, top_antecedent_cluster_ids_of_spans,
            top_antecedent_scores_of_spans, top_antecedent_mask_of_spans
        ) = self(*input_tensors)

        top_antecedent_cluster_ids_of_spans += torch.log(top_antecedent_mask_of_spans.float()).int()
        # [top_cand_num, pruned_antecedent_num]
        antecedent_indicators_of_spans = top_antecedent_cluster_ids_of_spans == top_span_cluster_ids.view(-1, 1)
        # [top_cand_num, 1]
        non_dummy_span_mask = (top_span_cluster_ids > 0).view(-1, 1)
        # [top_cand_num, pruned_antecedent_num]
        non_dummy_antecedent_indicators_of_spans = antecedent_indicators_of_spans & non_dummy_span_mask
        # [top_cand_num, 1]
        dummy_span_indicators = ~non_dummy_antecedent_indicators_of_spans.any(dim=1, keepdim=True)
        # [top_cand_num, 1 + pruned_antecedent_num]
        antecedent_indicators_of_spans = torch.cat(
            (dummy_span_indicators, non_dummy_antecedent_indicators_of_spans), dim=1
        )
        top_antecedent_scores_of_spans += torch.log(antecedent_indicators_of_spans.float())
        # [top_cand_num]
        log_marginalized_prob_of_spans = (
                torch.logsumexp(top_antecedent_scores_of_spans, dim=1)
                - torch.logsumexp(top_antecedent_scores_of_spans, dim=1)
        )

        return -log_marginalized_prob_of_spans.sum()

    def predict(self, *input_tensors):
        (
            cand_mention_scores, top_start_idxes, top_end_idxes, top_span_cluster_ids,
            top_antecedent_idxes_of_spans, top_antecedent_cluster_ids_of_spans,
            top_antecedent_scores_of_spans, top_antecedent_mask_of_spans
        ) = self(*input_tensors)

        predicted_antecedent_idxes = []

        for span_idx, loc in enumerate(torch.argmax(top_antecedent_scores_of_spans, dim=1) - 1):
            if loc < 0:
                predicted_antecedent_idxes.append(-1)
            else:
                predicted_antecedent_idxes.append(top_antecedent_idxes_of_spans[span_idx, loc].item())

        span_to_predicted_cluster_id = {}
        predicted_clusters = []

        for span_idx, antecedent_idx in enumerate(predicted_antecedent_idxes):
            if antecedent_idx < 0:
                continue

            assert span_idx > antecedent_idx

            antecedent_span = top_start_idxes[antecedent_idx], top_end_idxes[antecedent_idx]

            if antecedent_span in span_to_predicted_cluster_id:
                predicted_cluster_id = span_to_predicted_cluster_id[antecedent_span]
            else:
                predicted_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent_span])
                span_to_predicted_cluster_id[antecedent_span] = predicted_cluster_id

            span = top_start_idxes[span_idx], top_end_idxes[span_idx]
            predicted_clusters[predicted_cluster_id].append(span)
            span_to_predicted_cluster_id[span] = predicted_cluster_id

        predicted_clusters = [tuple(cluster) for cluster in predicted_clusters]
        span_to_predicted_cluster = {
            span: predicted_clusters[cluster_id]
            for span, cluster_id in span_to_predicted_cluster_id.items()
        }

        # [top_cand_num], [top_cand_num], [top_cand_num]
        return top_start_idxes, top_end_idxes, predicted_antecedent_idxes, \
               predicted_clusters, span_to_predicted_cluster
