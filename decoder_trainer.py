import configs
from modules import SelfAttendedDecoder, LangModel, Decoder
from model_utils import *
import os
from log import log
import time
import data_utils
import re
from itertools import chain
from allennlp.training.optimizers import DenseSparseAdam


class DecoderTrainer:
    def __init__(self):
        self.decoder = SelfAttendedDecoder(vocab=data_utils.vocab, device_id=configs.decoder_device_id)
        # self.decoder.apply(init_weights)
        self.xe_loss = nn.CrossEntropyLoss(ignore_index=data_utils.vocab.padding_id)
        self.optimizer = DenseSparseAdam(self.decoder.parameters(), lr=configs.adam_lr) # , lr=configs.lr)
        # self.optimizer = optim.ASGD(self.model.parameters(), lr=configs.lr, weight_decay=configs.l2_weight_decay)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min',
            patience=configs.lr_scheduler_patience,
            factor=configs.lr_scheduler_factor, verbose=True
        )
        self.epoch_idx = 0
        self.min_ppl = 1000.
        self.ckpt_path = 'decoder.pretrained.params'

    def train(self):
        # self.load_ckpt()
        start_epoch_idx = self.epoch_idx

        for epoch_idx in range(start_epoch_idx, configs.epoch_num):
            self.epoch_idx = epoch_idx

            log(f'starting epoch {epoch_idx}')
            log('training')

            self.decoder.train()
            avg_epoch_loss = 0.
            batch_num = 0
            next_logging_pct = .5

            start_time = time.time()

            for pct, (report_batch, report_len_batch) in data_utils.gen_report_batches('train'):
                batch_num += 1
                self.optimizer.zero_grad()
                max_report_len, batch_size = report_batch.shape

                # [max_report_len - 1, batch_size, vocab_size]
                word_logits_seq_batch = self.decoder.run_lang_model(
                    # [max_report_len, batch_size], [batch_size]
                    report_batch, report_len_batch
                )
                loss = self.xe_loss(
                    # [(max_report_len - 1) * batch_size, vocab_size]
                    word_logits_seq_batch.view(-1, data_utils.vocab.size),
                    # [(max_report_len - 1) * batch_size]
                    report_batch[1:, :].contiguous().view(-1).to(torch.device(configs.decoder_device_id))
                )

                # print(torch.argmax(output_batch[:50, :5, :], dim=-1))

                loss.backward()
                self.optimizer.step()
                avg_epoch_loss += loss.item()

                if pct >= next_logging_pct:
                    log(
                        f'{int(pct)}%, avg_train_loss: {avg_epoch_loss / batch_num}, '
                        f'time: {time.time() - start_time}'
                    )
                    next_logging_pct += 10.

            avg_epoch_loss /= batch_num

            log(
                f'avg_train_loss: {avg_epoch_loss}\n'
                f'avg_train_time: {time.time() - start_time}'
            )

            with torch.no_grad():
                log('validating')

                self.decoder.eval()
                batch_num = 0
                avg_epoch_ppl = 0
                next_logging_pct = 10.

                start_time = time.time()

                for pct, (report_batch, report_len_batch) in data_utils.gen_report_batches('test'):
                    batch_num += 1

                    max_report_len, batch_size = report_batch.shape

                    # [max_report_len - 1, batch_size, vocab_size]
                    word_logits_seq_batch = self.decoder.run_lang_model(
                        # [max_report_len, batch_size], [batch_size]
                        report_batch, report_len_batch
                    )
                    loss = self.xe_loss(
                        # [(max_report_len - 1) * batch_size, vocab_size]
                        word_logits_seq_batch.view(-1, data_utils.vocab.size),
                        # [(max_report_len - 1) * batch_size]
                        report_batch[1:, :].contiguous().view(-1).to(torch.device(configs.decoder_device_id))
                    )

                    # print(torch.argmax(output_batch[:50, :5, :], dim=-1))

                    avg_epoch_ppl += math.exp(loss.item())

                    if pct >= next_logging_pct:
                        log(
                            f'{int(pct)}%, avg_dev_ppl: {avg_epoch_ppl / batch_num}, '
                            f'time: {time.time() - start_time}'
                        )
                        next_logging_pct += 10.

                avg_epoch_ppl /= batch_num
                self.lr_scheduler.step(avg_epoch_ppl)

                log(
                    f'avg_dev_time: {time.time() - start_time}\n'
                    f'avg_dev_ppl: {avg_epoch_ppl}'
                )

                if avg_epoch_ppl < self.min_ppl:
                    self.min_ppl = avg_epoch_ppl
                    self.save_ckpt()

    def get_ckpt(self):
        return {
            'epoch_idx': self.epoch_idx,
            'min_ppl': self.min_ppl,
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()
        }

    def set_ckpt(self, ckpt_dict):
        self.epoch_idx = ckpt_dict['epoch_idx'] + 1
        self.min_ppl = ckpt_dict['min_ppl']

        if configs.uses_gumbel_softmax:
            ckpt_dict['decoder']['embedder.weight'] = \
                ckpt_dict['decoder']['embedder.weight'].to(torch.device(configs.decoder_device_id))
        else:
            ckpt_dict['decoder']['embedder.weight'] = \
                ckpt_dict['decoder']['embedder.weight'].cpu()

        self.decoder.load_state_dict(ckpt_dict['decoder'])

        if configs.uses_new_optimizer:
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            self.lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])
        del ckpt_dict
        torch.cuda.empty_cache()

    ckpt = property(get_ckpt, set_ckpt)

    def save_ckpt(self):
        torch.save(self.ckpt, f=self.ckpt_path)
        print(f'saved checkpoint to {self.ckpt_path}')

    def load_ckpt(self):
        self.ckpt = torch.load(self.ckpt_path)
        print(f'loaded checkpoint from {self.ckpt_path}')


if __name__ == '__main__':
    decoder_trainer = DecoderTrainer()
    decoder_trainer.train()
