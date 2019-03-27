import configs
from model import Model
from model_utils import *
from vocab import Vocab
from collections import defaultdict
import json

vocab = Vocab()

id_to_class, class_to_id = [], {}

with open(configs.classes_path) as classes_file:
    for class_ in map(lambda s: s.strip(), classes_file.readlines()):
        class_to_id[class_] = len(id_to_class)
        id_to_class.append(class_)

configs.class_num = len(class_to_id)


class SingleClassDataset(tud.Dataset):
    def __init__(self, texts, class_id):
        self.texts = texts
        self.class_id = class_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.class_id


datasets = [
    SingleClassDataset([], class_id)
    for class_id in range(configs.class_num)
]

for text, label in zip(
        np.load(f'{configs.data_dir}/texts.id.train.npy'),
        np.load(f'{configs.data_dir}/labels.train.npy')
):
    datasets[label].texts.append(text)


def collate(batch):
    text_batch, label_batch = zip(*batch)

    # [batch_size]
    len_batch = torch.LongTensor(
        [len(text) for text in text_batch]
    )
    max_len = max(len_batch)
    # [batch_size, max_len]
    text_batch = torch.LongTensor(
        [
            np.concatenate((text, np.full((max_len - len(text)), vocab.padding_id)))
            if len(text) < max_len else text
            for text in text_batch
        ]
    )
    label_batch = torch.LongTensor(label_batch)

    return text_batch.cuda(), label_batch.cuda()


data_loaders = [
    tud.DataLoader(
        dataset=datasets[class_id],
        batch_size=16384,
        shuffle=False,
        # pin_memory=True,
        collate_fn=collate,
    )
    for class_id in range(configs.class_num)
]


def gen_batches(class_id):
    instance_num = 0

    for batch in data_loaders[class_id]:
        instance_num += len(batch[-1])
        pct = instance_num * 100. / len(datasets[class_id])
        yield pct, batch


model = Model()

'''
Model(
  (embedder): Embedding(35330, 300)
  (feature_extractors): ModuleList(
    (0): Sequential(
      (0): Conv1d(300, 100, kernel_size=(3,), stride=(1,))
      (1): ReLU()
      (2): AdaptiveMaxPool1d(output_size=1)
      (3): Reshaper()
    )
    (1): Sequential(
      (0): Conv1d(300, 100, kernel_size=(4,), stride=(1,))
      (1): ReLU()
      (2): AdaptiveMaxPool1d(output_size=1)
      (3): Reshaper()
    )
    (2): Sequential(
      (0): Conv1d(300, 100, kernel_size=(5,), stride=(1,))
      (1): ReLU()
      (2): AdaptiveMaxPool1d(output_size=1)
      (3): Reshaper()
    )
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5)
    (1): Linear(in_features=300, out_features=16, bias=True)
  )
)
'''

ckpt = torch.load('ckpts/0216-222720.5.ckpt')
model.load_state_dict(ckpt['model'])
print(f'loaded checkpoint 0216-222720.5.ckpt')
del ckpt
torch.cuda.empty_cache()

top_kernel_num = 20
feature_num = 300
kernel_widths = (3, 4, 5)
kernel_num_per_width = 100
top_ngram_num = 100

interpretation = []

with torch.no_grad():
    for class_id in range(configs.class_num):
        print(id_to_class[class_id])

        top_kernels = []
        interpretation.append(
            {
                'class_id': class_id,
                'class': id_to_class[class_id],
                'top_kernels': top_kernels
            }
        )

        kernel_scores = torch.zeros(feature_num)
        max_ngram_idxes_list = []
        max_conv_scores_list = []

        for pct, (text_batch, label_batch) in gen_batches(class_id):
            embeddings_batch = model.embedder(text_batch).transpose_(1, 2)
            feature_vec_batch = []
            max_ngram_idxes_batch = []

            for width_id in range(len(kernel_widths)):
                # [batch_size, kernel_num, len]
                conv_score_seqs_batch_of_width = F.relu(model.feature_extractors[width_id][0](embeddings_batch))
                # [batch_size, kernel_num, 1], [batch_size, kernel_num, 1]
                max_conv_scores_batch_of_width, max_ngram_idxes_batch_of_width = F.adaptive_max_pool1d(
                    conv_score_seqs_batch_of_width, output_size=1, return_indices=True
                )
                feature_vec_batch.append(max_conv_scores_batch_of_width.view(-1, kernel_num_per_width))
                max_ngram_idxes_batch.append(max_ngram_idxes_batch_of_width.view(-1, kernel_num_per_width))

            # [batch_size, feature_num]
            max_ngram_idxes_batch = torch.cat(max_ngram_idxes_batch, dim=-1)
            # batch_num * [batch_size, feature_num]
            max_ngram_idxes_list.append(max_ngram_idxes_batch.cpu())
            # [batch_size, feature_num]
            feature_vec_batch = torch.cat(feature_vec_batch, dim=-1)
            # batch_num * [batch_size, feature_num]
            max_conv_scores_list.append(feature_vec_batch.cpu())
            # [batch_size, feature_num]
            kernel_scores_batch = model.classifier[1].weight[class_id].view(1, feature_num) * feature_vec_batch
            kernel_scores += feature_vec_batch.sum(dim=0).cpu()

        kernel_scores /= len(datasets[class_id])

        top_kernel_scores, top_kernel_ids = torch.topk(kernel_scores, k=top_kernel_num)
        # [top_kernel_num]
        top_kernel_ids = top_kernel_ids.cpu().numpy()
        # [dataset_size, top_kernel_num]
        max_ngram_idxes_list = torch.cat(max_ngram_idxes_list, dim=0).numpy()[:, top_kernel_ids]
        # [dataset_size, top_kernel_num]
        max_conv_scores_list = torch.cat(max_conv_scores_list, dim=0).numpy()[:, top_kernel_ids]

        assert len(max_ngram_idxes_list) == len(max_conv_scores_list) == len(datasets[class_id])

        kernel_id_to_ngram_scores = defaultdict(lambda: defaultdict(float))

        for max_conv_scores, max_ngram_idxes, word_ids in zip(
                max_conv_scores_list, max_ngram_idxes_list, datasets[class_id].texts
        ):
            for kernel_id, max_conv_score, max_ngram_idx in zip(
                    top_kernel_ids, max_conv_scores, max_ngram_idxes
            ):
                ngram_scores = kernel_id_to_ngram_scores[kernel_id]
                width = None

                for i in range(len(kernel_widths)):
                    if kernel_id in range(i * kernel_num_per_width, (i + 1) * kernel_num_per_width):
                        width = kernel_widths[i]
                        break

                ngram = vocab.textify(word_ids[max_ngram_idx:(max_ngram_idx + width)])
                ngram_scores[ngram] = max(ngram_scores[ngram], float(max_conv_score))

        for kernel_id, kernel_score in zip(top_kernel_ids, top_kernel_scores):
            top_kernels.append(
                {
                    'kernel_id': int(kernel_id),
                    'kernel_score': float(kernel_score),
                    'top_ngram_scores': sorted(
                        kernel_id_to_ngram_scores[kernel_id].items(), key=(lambda g_s: g_s[1]), reverse=True
                    )[:top_ngram_num]
                }
            )

json.dump(interpretation, open('interpretation.json', 'w'), indent=4)

