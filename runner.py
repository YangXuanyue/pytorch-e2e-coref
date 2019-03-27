import configs
from model import Model
from modules import SelfAttendedDecoder
from model_utils import *
import os
from log import log
import time
import data_utils
import re
from itertools import chain
from allennlp.training.optimizers import DenseSparseAdam


class Runner:
    def __init__(self):
        self.model = Model()

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

        self.epoch_idx = 0
        self.max_acc = 0.

    @staticmethod
    def compute_prediction_batch(logits_batch):
        # [batch_size]
        return torch.argmax(logits_batch.detach(), dim=-1)

    @staticmethod
    def compute_accuracy(logits_batch, label_batch):
        prediction_batch = Runner.compute_prediction_batch(logits_batch)
        return (prediction_batch == label_batch).type(torch.cuda.FloatTensor).mean().item()

    @staticmethod
    def compute_loss(antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + torch.log(antecedent_labels.float())

        # gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]

        marginalized_gold_scores = F.log_softmax(gold_scores, dim=1)

        # marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = F.log_softmax(antecedent_scores, dim=1)
        # log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]

        return log_norm - marginalized_gold_scores  # [k]

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

            avg_epoch_loss = 0.
            avg_epoch_acc = 0.
            batch_num = 0
            next_logging_pct = .5
            start_time = time.time()

            for pct, input_tensors in data_utils.gen_batches('train'):
                batch_num += 1

                antecedent_scores, antecedent_labels = self.model(*input_tensors)

                self.optimizer.zero_grad()
                loss = self.compute_loss(antecedent_scores, antecedent_labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.get_trainable_params(), max_norm=configs.max_grad_norm)
                self.lr_scheduler.step()
                self.optimizer.step()

                # if not configs.freezes_embeddings and epoch_idx < configs.embedder_training_epoch_num:
                #     self.embedder_optimizer.step()

                avg_epoch_loss += loss.item()

                if pct >= next_logging_pct:
                    log(
                        f'{int(pct)}%, '
                        f'avg_train_loss: {avg_epoch_loss / batch_num}, '
                        f'time: {time.time() - start_time}'
                    )
                    next_logging_pct += 5.

                    self.validate()

            avg_epoch_loss /= batch_num
            avg_epoch_acc /= batch_num

            log(
                f'avg_train_loss: {avg_epoch_loss}\n'
                f'avg_train_time: {time.time() - start_time}'
            )

            self.validate()

    def validate(self, name='valid', saves_results=False):
        with torch.no_grad():
            log('validating')

            self.model.eval()
            batch_num = 0
            avg_epoch_loss = 0.
            epoch_acc = 0.
            next_logging_pct = 10.
            start_time = time.time()
            predictions = []
            labels = []

            for pct, input_tensors in data_utils.gen_batches(name):
                batch_num += 1

                antecedent_scores, antecedent_labels = self.model(*input_tensors)
                loss = self.compute_loss(antecedent_scores, antecedent_labels)
                avg_epoch_loss += loss.item()

                if pct >= next_logging_pct:
                    log(
                        f'{int(pct)}%, '
                        f'avg_train_loss: {avg_epoch_loss / batch_num}, '
                        f'time: {time.time() - start_time}'
                    )
                    next_logging_pct += 5.

            avg_epoch_loss /= batch_num
            log(
                f'avg_valid_loss: {avg_epoch_loss} '
                f'avg_valid_time: {time.time() - start_time}'
            )

            if saves_results:
                data_utils.save_predictions(name, predictions)

            if name == 'valid':
                if epoch_acc > self.max_acc:
                    self.max_acc = epoch_acc
                    # self.save_ckpt()

                    max_acc_file = open(configs.max_acc_path)

                    if epoch_acc > float(max_acc_file.readline().strip()):
                        max_acc_file.close()
                        max_acc_file = open(configs.max_acc_path, 'w')
                        print(epoch_acc, file=max_acc_file)
                        print(configs.seed, file=max_acc_file)
                        self.save_ckpt()

                    max_acc_file.close()

                # self.lr_scheduler.step(epoch_acc)
                self.lr_scheduler.step(-avg_epoch_loss)

    def test(self):
        with torch.no_grad():
            log('testing')

            self.model.eval()
            batch_num = 0
            next_logging_pct = 10.
            start_time = time.time()
            predictions = []

            for pct, (text_batch, _) in data_utils.gen_batches('test'):
                batch_num += 1
                # [batch_size, class_num]
                logits_batch = self.model(text_batch)
                predictions.extend(Runner.compute_prediction_batch(logits_batch))

                if pct >= next_logging_pct:
                    log(
                        f'{int(pct)}%, '
                        f'time: {time.time() - start_time}'
                    )
                    next_logging_pct += 10.

            data_utils.save_predictions('test', predictions)

    def get_ckpt(self):
        return {
            'epoch_idx': self.epoch_idx,
            'max_acc': self.max_acc,
            'seed': configs.seed,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'embedder_optimizer': self.embedder_optimizer.state_dict() if not configs.freezes_embeddings else None,
            'lr_scheduler': self.lr_scheduler.state_dict()
        }

    def set_ckpt(self, ckpt_dict):
        self.epoch_idx = ckpt_dict['epoch_idx'] + 1
        self.max_acc = ckpt_dict['max_acc']

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
    trainer = Runner()

    if configs.training:
        trainer.train()
    else:
        trainer.test()
