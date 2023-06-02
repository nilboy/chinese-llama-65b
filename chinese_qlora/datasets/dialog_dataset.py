import os
import copy
import json
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer
import torch

class DialogDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 tokenizer_path: str,
                 max_token_len: int = 2048):
        self.data = []
        with open(file_path) as fin:
            for line in fin:
                record = json.loads(line)
                text = self.parse_record_to_text(record)
                self.data.append(text)
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.max_token_len = max_token_len

    @staticmethod
    def parse_record_to_text(record):
        messages = []
        if len(record) > 1:
            pattern = "<|im_start|>{role}\n{text}\n<|im_end|>"
            for item in record:
                role, text = item['role'], item['text']
                message = pattern.format(role=role,
                                         text=text)
                messages.append(message)
            return "\n".join(messages)
        else:
            return record[0]['text']

    @staticmethod
    def pad_to_max_length(ids, max_length, pad_value):
        pad_ids = ids + [pad_value] * (max_length - len(ids))
        return pad_ids[0:max_length]

    def __getitem__(self, index):
        encoding = self.tokenizer.encode_plus(self.data[index],
                                              add_special_tokens=True)
        input_ids = encoding['input_ids'] + [self.tokenizer.eos_token_id]
        labels = copy.deepcopy(input_ids)
        attention_mask = encoding['attention_mask'] + [1]
        return {
            'input_ids': torch.LongTensor(self.pad_to_max_length(input_ids, self.max_token_len, self.tokenizer.pad_token_id)),
            'attention_mask': torch.LongTensor(self.pad_to_max_length(attention_mask, self.max_token_len, 0)),
            'labels': torch.LongTensor(self.pad_to_max_length(labels, self.max_token_len, -100))
        }

    def __len__(self):
        return len(self.data)


class DialogDataModule(object):
    """Dialog Dataset"""
    def __init__(self,
                 data_dir: str,
                 tokenizer_path: str,
                 batch_size: int = 4,
                 max_token_len: int = 2048,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        """
        Args:
            data_dir: str. The data_dir directory contains three files: train.jsonl, val.jsonl, test.jsonl.
        """
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train = DialogDataset(os.path.join(data_dir, 'train.jsonl'),
                                        self.tokenizer_path,
                                        self.max_token_len)
        self.data_val = DialogDataset(os.path.join(data_dir, 'val.jsonl'),
                                      self.tokenizer_path,
                                      self.max_token_len)
        self.data_test = DialogDataset(os.path.join(data_dir, 'test.jsonl'),
                                       self.tokenizer_path,
                                       self.max_token_len)
    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)
