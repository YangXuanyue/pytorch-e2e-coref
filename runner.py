import configs
from model import Model
from model_utils import *
import os
from log import log
import time
import data_utils
import re
from itertools import chain
from allennlp.training.optimizers import DenseSparseAdam
import metrics
import traceback
import conll


class Runner:
    def __init__(self):
        self.model = Model().cuda()

        self.optimizer = optim.Adam(
            # filter(lambda p: p.requires_grad, self.model.parameters()),

            # (param for name, param in self.model.named_parameters() if 'embedder' not in name),
            self.model.get_trainable_params(),
            lr=configs.initial_lr,
            # weight_decay=configs.l2_weight_decay
        )

        self.lr_scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=configs.lr_decay_freq, gamma=configs.lr_decay_rate
        )
        # self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, 'max',
        #     patience=configs.lr_scheduler_patience,
        #     factor=configs.lr_scheduler_factor, verbose=True
        # )

        self.epoch_idx = 0
        self.max_f1 = 0.

    @staticmethod
    def compute_prediction_batch(logits_batch):
        # [batch_size]
        return torch.argmax(logits_batch.detach(), dim=-1)

    @staticmethod
    def compute_accuracy(logits_batch, label_batch):
        prediction_batch = Runner.compute_prediction_batch(logits_batch)
        return (prediction_batch == label_batch).type(torch.cuda.FloatTensor).mean().item()

    @staticmethod
    def compute_mention_loss(
        # [cand_num], [cand_num]
        cand_mention_scores, cand_mention_labels, margin=1.
    ):
        # log_mention_probs = F.logsigmoid(cand_mention_scores)
        # log_true_mention_probs = log_mention_probs[cand_mention_labels]
        # min_log_true_mention_prob = log_true_mention_probs.min()
        # log_false_mention_probs = log_mention_probs[~cand_mention_labels]
        # true_mention_num, = log_true_mention_probs.shape
        # false_mention_num, = log_false_mention_probs.shape
        #
        # if not true_mention_num:
        #     return None
        #
        # max_log_false_mention_probs = log_false_mention_probs.max()
        #
        # mention_loss = F.relu(
        #     max_log_false_mention_probs - min_log_true_mention_prob + margin
        # ) - log_true_mention_probs.sum()

        # [min(true_mention_num, false_mention_num)]
        # top_log_false_mention_probs, _ = log_false_mention_probs.topk(
        #     k=min(true_mention_num, false_mention_num)
        # )
        # top_log_false_mention_probs = top_log_false_mention_probs[
        #     top_log_false_mention_probs > min_log_true_mention_prob - margin
        # ]
        #
        # mention_loss = (
        #     top_log_false_mention_probs.sum() if top_log_false_mention_probs.nelement() else 0.
        # ) - log_true_mention_probs.sum()

        # if mention_loss.item() < -10.:
        #     breakpoint()

        log_normalized_mention_probs = F.log_softmax(cand_mention_scores)
        log_normalized_true_mention_probs = log_normalized_mention_probs[cand_mention_labels]

        if log_normalized_true_mention_probs.nelement():
            return -log_normalized_true_mention_probs.sum()
        else:
            return None


    @staticmethod
    def compute_ant_loss(
        # self,
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
        # # [top_cand_num, 1 + pruned_ant_num]
        # top_ant_scores_of_spans,
        # 4 * [top_cand_num, 1 + pruned_ant_num]
        list_of_top_ant_scores_of_spans,
        # [top_span_num, pruned_ant_num]
        top_ant_mask_of_spans,
        # [doc_len, pos_tag_num]
        # pos_tag_logits
        # # pos_tags
        # [top_span_num, 1 + top_span_num], [top_span_num, top_span_num]
        full_fast_ant_scores_of_spans, full_ant_mask_of_spans
    ):
        # start_time = time.time()

        # (
        #
        # ) = self.model(*input_tensors)

        # print(f'forward: {time.time() - start_time}')

        # top_span_cluster_ids

        top_span_num, = top_span_cluster_ids.shape

        # [top_cand_num, 1]
        non_dummy_span_mask = (top_span_cluster_ids > 0).view(-1, 1)

        top_ant_cluster_ids_of_spans += torch.log(top_ant_mask_of_spans.float()).long()
        # [top_cand_num, pruned_ant_num]
        top_ant_indicators_of_spans = top_ant_cluster_ids_of_spans == top_span_cluster_ids.view(-1, 1)

        # [top_cand_num, pruned_ant_num]
        non_dummy_top_ant_indicators_of_spans = top_ant_indicators_of_spans & non_dummy_span_mask

        # [top_cand_num, 1 + pruned_ant_num]
        top_ant_indicators_of_spans = torch.cat(
            (
                # [top_cand_num, 1]
                ~non_dummy_top_ant_indicators_of_spans.any(dim=1, keepdim=True),
                # [top_cand_num, pruned_ant_num]
                non_dummy_top_ant_indicators_of_spans
            ), dim=1
        )
        # [top_cand_num]
        # neg_log_marginalized_prob_of_spans

        loss = sum(
            -(
                torch.logsumexp(
                    top_ant_scores_of_spans + torch.log(top_ant_indicators_of_spans.float()).cuda(),
                    dim=1
                ) - torch.logsumexp(top_ant_scores_of_spans, dim=1)
            ).sum()
            for top_ant_scores_of_spans in list_of_top_ant_scores_of_spans
        )

        if configs.supervises_unpruned_fast_ant_scores:
            # [top_span_num, top_span_num]
            full_ant_cluster_ids_of_spans = top_span_cluster_ids.view(-1, 1).repeat(1, top_span_num)
            full_ant_cluster_ids_of_spans += torch.log(full_ant_mask_of_spans.float()).long()
            # [top_span_num, top_span_num]
            full_ant_indicators_of_spans = full_ant_cluster_ids_of_spans == top_span_cluster_ids.view(1, -1)
            # [top_cand_num, top_span_num]
            non_dummy_full_ant_indicators_of_spans = full_ant_indicators_of_spans & non_dummy_span_mask

            # [top_span_num, 1 + top_span_num]
            full_ant_indicators_of_spans = torch.cat(
                (
                    # [top_span_num, 1]
                    ~non_dummy_full_ant_indicators_of_spans.any(dim=1, keepdim=True),
                    # [top_span_num, top_span_num]
                    non_dummy_full_ant_indicators_of_spans
                ), dim=1
            )
            loss += -(
                torch.logsumexp(
                    full_fast_ant_scores_of_spans + torch.log(full_ant_indicators_of_spans.float()).cuda(),
                    dim=1
                ) - torch.logsumexp(full_fast_ant_scores_of_spans, dim=1)
            ).sum()

        return loss

    @staticmethod
    def predict(
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
        # # [top_cand_num, 1 + pruned_ant_num]
        # top_ant_scores_of_spans,
        # 4 * [top_cand_num, 1 + pruned_ant_num]
        list_of_top_ant_scores_of_spans,
        # [top_span_num, pruned_ant_num]
        top_ant_mask_of_spans,
        # # [doc_len, pos_tag_num]
        # pos_tag_logits
    ):
        # (
        #
        # ) = self.model(*input_tensors)

        predicted_ant_idxes = []

        for span_idx, loc in enumerate(torch.argmax(list_of_top_ant_scores_of_spans[-1], dim=1) - 1):
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

    # @staticmethod
    # def compute_loss(ant_scores, ant_labels):
    #     gold_scores = ant_scores + torch.log(ant_labels.float())
    #
    #     # gold_scores = ant_scores + tf.log(tf.to_float(ant_labels))  # [k, max_ant + 1]
    #
    #     marginalized_gold_scores = F.log_softmax(gold_scores, dim=1)
    #
    #     # marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
    #     log_norm = F.log_softmax(ant_scores, dim=1)
    #     # log_norm = tf.reduce_logsumexp(ant_scores, [1])  # [k]
    #
    #     return log_norm - marginalized_gold_scores  # [k]

    def test_gpu(self):
        example_idx, input_tensors = data_utils.datasets['train'][2674]
        loss = self.model.compute_loss(input_tensors)
        print(loss.item())
        self.optimizer.zero_grad()
        # torch.cuda.empty_cache()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.get_trainable_params(), max_norm=configs.max_grad_norm)
        self.optimizer.step()

    def train(self):
        if configs.ckpt_id or configs.loads_ckpt or configs.loads_best_ckpt:
            self.load_ckpt()

        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)

        start_epoch_idx = self.epoch_idx

        for epoch_idx in range(start_epoch_idx, configs.epoch_num):
            self.epoch_idx = epoch_idx

            log(f'starting epoch {epoch_idx}')
            log('training')

            self.model.train()

            avg_epoch_ant_loss = 0.
            avg_epoch_pos_loss = 0.
            avg_epoch_mention_loss = 0.
            avg_epoch_pos_acc = 0.
            avg_epoch_loss = 0.
            batch_num = 0
            next_logging_pct = .5
            next_evaluating_pct = 20.
            start_time = time.time()

            for pct, example_idx, input_tensors, pos_tags, cand_mention_labels in data_utils.gen_batches(
                'train' if configs.training else 'test'):
                batch_num += 1

                self.optimizer.zero_grad()

                # print(example_idx)
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
                    # # [top_cand_num, 1 + pruned_ant_num]
                    # top_ant_scores_of_spans,
                    # 4 * [top_cand_num, 1 + pruned_ant_num]
                    list_of_top_ant_scores_of_spans,
                    # [top_span_num, pruned_ant_num]
                    top_ant_mask_of_spans,
                    # [doc_len, pos_tag_num]
                    pos_tag_logits,
                    # [top_span_num, 1 + top_span_num], [top_span_num, top_span_num]
                    full_fast_ant_scores_of_spans, full_ant_mask_of_spans
                ) = self.model(*input_tensors)

                loss = 0.

                # try:
                ant_loss = Runner.compute_ant_loss(
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
                    # # [top_cand_num, 1 + pruned_ant_num]
                    # top_ant_scores_of_spans,
                    # 4 * [top_cand_num, 1 + pruned_ant_num]
                    list_of_top_ant_scores_of_spans,
                    # [top_span_num, pruned_ant_num]
                    top_ant_mask_of_spans,
                    # [top_span_num, 1 + top_span_num], [top_span_num, top_span_num]
                    full_fast_ant_scores_of_spans, full_ant_mask_of_spans
                )

                avg_epoch_ant_loss += ant_loss.item()

                loss = ant_loss

                if configs.predicts_pos_tags:
                    pos_tags = pos_tags.cuda()

                    pos_loss = F.cross_entropy(
                        pos_tag_logits, pos_tags, ignore_index=data_utils.pos_tag_vocab.padding_id
                    )
                    avg_epoch_pos_loss += pos_loss.item()
                    loss += pos_loss
                    avg_epoch_pos_acc += Runner.compute_accuracy(pos_tag_logits, pos_tags)

                if configs.supervises_mention_scores:
                    mention_loss = Runner.compute_mention_loss(cand_mention_scores, cand_mention_labels)

                    if mention_loss is not None:
                        loss += mention_loss
                        avg_epoch_mention_loss += mention_loss.item()

                avg_epoch_loss += loss.item()
                loss.backward()

                # nn.utils.clip_grad_norm_(self.model.get_trainable_params(), max_norm=configs.max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step()

                # if not configs.freezes_embeddings and epoch_idx < configs.embedder_training_epoch_num:
                #     self.embedder_optimizer.step()

                if pct >= next_logging_pct:
                    na_str = 'N/A'

                    log(
                        f'{int(pct)}%, time: {time.time() - start_time}\n'
                        f'avg_epoch_loss: {avg_epoch_loss / batch_num}\n'
                        f'avg_mention_loss: '
                        f'{avg_epoch_mention_loss / batch_num if configs.supervises_mention_scores else na_str}\n'
                        f'avg_pos_loss: {avg_epoch_pos_loss / batch_num if configs.predicts_pos_tags else na_str}\n'
                        f'avg_ant_loss: {avg_epoch_ant_loss / batch_num}\n'
                        f'avg_epoch_pos_acc: {avg_epoch_pos_acc / batch_num if configs.predicts_pos_tags else na_str}\n'
                    )

                    next_logging_pct += 5.

                if pct >= next_evaluating_pct:
                    avg_conll_f1 = self.evaluate()

                    if avg_conll_f1 > self.max_f1:
                        self.max_f1 = avg_conll_f1
                        # self.save_ckpt()

                        max_f1_file = open(configs.max_f1_path)

                        if avg_conll_f1 > float(max_f1_file.readline().strip()):
                            max_f1_file.close()
                            max_f1_file = open(configs.max_f1_path, 'w')
                            print(avg_conll_f1, file=max_f1_file)
                            self.save_ckpt()

                        max_f1_file.close()

                    next_evaluating_pct += 20.

                    # self.evaluate()

            avg_epoch_loss /= batch_num
            avg_epoch_pos_loss /= batch_num
            avg_epoch_mention_loss /= batch_num
            avg_epoch_ant_loss /= batch_num
            avg_epoch_pos_acc /= batch_num

            na_str = 'N/A'

            log(
                f'100%,\ttime:\t{time.time() - start_time}\n'
                f'avg_epoch_loss:\t{avg_epoch_loss}\n'
                f'avg_mention_loss:\t{avg_epoch_mention_loss if configs.supervises_mention_scores else na_str}\n'
                f'avg_pos_loss:\t{avg_epoch_pos_loss if configs.predicts_pos_tags else na_str}\n'
                f'avg_ant_loss:\t{avg_epoch_ant_loss}\n'
                f'avg_epoch_pos_acc:\t{avg_epoch_pos_acc if configs.predicts_pos_tags else na_str}\n'
            )

            avg_conll_f1 = self.evaluate()

            if avg_conll_f1 > self.max_f1:
                self.max_f1 = avg_conll_f1
                # self.save_ckpt()

                max_f1_file = open(configs.max_f1_path)

                if avg_conll_f1 > float(max_f1_file.readline().strip()):
                    max_f1_file.close()
                    max_f1_file = open(configs.max_f1_path, 'w')
                    print(avg_conll_f1, file=max_f1_file)
                    self.save_ckpt()

                max_f1_file.close()


    def evaluate(self, name='test', saves_results=False):
        # from collections import Counter
        # span_len_cnts = Counter()

        with torch.no_grad():
            log('evaluating')
            evaluator = metrics.CorefEvaluator()

            self.model.eval()
            batch_num = 0
            next_logging_pct = 10.
            start_time = time.time()
            cluster_predictions = {}
            avg_pos_acc = 0.

            for pct, example_idx, input_tensors, pos_tags, cand_mention_labels in data_utils.gen_batches(name):
                batch_num += 1

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
                    # # [top_cand_num, 1 + pruned_ant_num]
                    # top_ant_scores_of_spans,
                    # 4 * [top_cand_num, 1 + pruned_ant_num]
                    list_of_top_ant_scores_of_spans,
                    # [top_span_num, pruned_ant_num]
                    top_ant_mask_of_spans,
                    # [doc_len, pos_tag_num]
                    pos_tag_logits,
                    # [top_span_num, 1 + top_span_num], [top_span_num, top_span_num]
                    full_fast_ant_scores_of_spans, full_ant_mask_of_spans
                ) = self.model(*input_tensors)

                (
                    top_start_idxes, top_end_idxes, predicted_ant_idxes,
                    predicted_clusters, span_to_predicted_cluster
                ) = Runner.predict(
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
                    # # [top_cand_num, 1 + pruned_ant_num]
                    # top_ant_scores_of_spans,
                    # 4 * [top_cand_num, 1 + pruned_ant_num]
                    list_of_top_ant_scores_of_spans,
                    # [top_span_num, pruned_ant_num]
                    top_ant_mask_of_spans
                )

                # span_len_cnts.update((top_end_idxes - top_start_idxes + 1).tolist())

                if configs.predicts_pos_tags:
                    avg_pos_acc += Runner.compute_accuracy(pos_tag_logits, pos_tags.cuda())

                gold_clusters = data_utils.get_gold_clusters(name, example_idx)
                gold_clusters = [
                    tuple(tuple(span) for span in cluster)
                    for cluster in gold_clusters
                ]
                span_to_gold_cluster = {
                    span: cluster
                    for cluster in gold_clusters
                    for span in cluster
                }

                evaluator.update(
                    predicted=predicted_clusters,
                    gold=gold_clusters,
                    mention_to_predicted=span_to_predicted_cluster,
                    mention_to_gold=span_to_gold_cluster
                )
                cluster_predictions[data_utils.get_doc_key(name, example_idx)] = predicted_clusters

                if pct >= next_logging_pct:
                    na_str = 'N/A'

                    log(
                        f'{int(pct)}%,\ttime:\t{time.time() - start_time}\n'
                        f'pos_acc:\t{avg_pos_acc / batch_num if configs.predicts_pos_tags else na_str}\n'
                        f'f1:\t{evaluator.get_f1()}\n'
                    )
                    next_logging_pct += 5.

            epoch_precision, epoch_recall, epoch_f1 = evaluator.get_prf()

            avg_pos_acc /= batch_num

            avg_conll_f1 = conll.compute_avg_conll_f1(
                f'{configs.data_dir}/{name}.english.v4_gold_conll', cluster_predictions, official_stdout=True
            )

            na_str = 'N/A'

            log(
                f'avg_valid_time:\t{time.time() - start_time}\n'
                f'pos_acc:\t{avg_pos_acc if configs.predicts_pos_tags else na_str}\n'
                f'precision:\t{epoch_precision}\n'
                f'recall:\t{epoch_recall}\n'
                f'f1:\t{epoch_f1}\n'
                f'conll_f1: {avg_conll_f1}'
            )

            # if saves_results:
            #     data_utils.save_predictions(name, cluster_predictions)

            # if name == 'test' and configs.training:
            #     if avg_conll_f1 > self.max_f1:
            #         self.max_f1 = avg_conll_f1
            #         # self.save_ckpt()
            #
            #         max_f1_file = open(configs.max_f1_path)
            #
            #         if epoch_f1 > float(max_f1_file.readline().strip()):
            #             max_f1_file.close()
            #             max_f1_file = open(configs.max_f1_path, 'w')
            #             print(epoch_f1, file=max_f1_file)
            #             self.save_ckpt()
            #
            #         max_f1_file.close()

                # self.lr_scheduler.step(epoch_f1)
                # self.lr_scheduler.step(-avg_epoch_loss)

            return avg_conll_f1

    def get_ckpt(self):
        return {
            'epoch_idx': self.epoch_idx,
            'max_f1': self.max_f1,
            'seed': configs.seed,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'embedder_optimizer': self.embedder_optimizer.state_dict() if not configs.freezes_embeddings else None,
            'lr_scheduler': self.lr_scheduler.state_dict()
        }

    def set_ckpt(self, ckpt_dict):
        if not configs.restarts:
            self.epoch_idx = ckpt_dict['epoch_idx'] + 1

        if not configs.resets_max_f1:
            self.max_f1 = ckpt_dict['max_f1']

        model_state_dict = self.model.state_dict()
        model_state_dict.update(
            {
                name: param
                for name, param in ckpt_dict['model'].items()
                if name in model_state_dict
            }
        )

        self.model.load_state_dict(model_state_dict)
        del model_state_dict

        if not (configs.uses_new_optimizer or configs.sets_new_lr):
            #     if ckpt_dict['embedder_optimizer'] and not configs.freezes_embeddings:
            #         self.embedder_optimizer.load_state_dict(ckpt_dict['embedder_optimizer'])
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            self.lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])
            print('loaded optimizer and lr_scheduler')

        del ckpt_dict

        torch.cuda.empty_cache()

    ckpt = property(get_ckpt, set_ckpt)

    def save_ckpt(self):
        ckpt_path = f'{configs.ckpts_dir}/{configs.timestamp}.{self.epoch_idx}.ckpt'
        log(f'saving checkpoint {ckpt_path}')
        torch.save(self.ckpt, f=ckpt_path)

    @staticmethod
    def to_timestamp_and_epoch_idx(ckpt_path_):
        date, time, epoch_idx = map(int, re.split(r'[-.]', ckpt_path_[:ckpt_path_.find('.ckpt')]))
        return date, time, epoch_idx

    def load_ckpt(self, ckpt_path=None):
        if not ckpt_path:
            if configs.ckpt_id:
                ckpt_path = f'{configs.ckpts_dir}/{configs.ckpt_id}.ckpt'
            elif configs.loads_best_ckpt:
                ckpt_path = configs.best_ckpt_path
            else:
                ckpt_paths = [path for path in os.listdir(f'{configs.ckpts_dir}/') if path.endswith('.ckpt')]
                ckpt_path = f'{configs.ckpts_dir}/{sorted(ckpt_paths, key=Runner.to_timestamp_and_epoch_idx)[-1]}'

        print(f'loading checkpoint {ckpt_path}')

        self.ckpt = torch.load(ckpt_path)


if __name__ == '__main__':
    runner = Runner()

    if configs.training or configs.debugging:
        runner.train()
        # try:
        #     runner.train()
        # except:
        #     traceback.print_stack()
        #     breakpoint()
    elif configs.validating:
        runner.evaluate()
        ...
        # trainer.test()
