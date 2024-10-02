import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import zipfile
from torch.utils.data.distributed import DistributedSampler
import random
from tokenizers import Tokenizer

def read_tokenizer(
        tokenizer_src_path: str,
        tokenizer_tgt_path: str,
):
    tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
    tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)

    if not tokenizer_src or not tokenizer_tgt:
        ValueError("Tokenizer not found")
    
    return tokenizer_src, tokenizer_tgt

def read_csv_from_zip(zip_path, csv_filename):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(csv_filename) as csv_file:
            df = pd.read_csv(csv_file)
    return df

def get_file(file_path, file_name="zip_file.csv"):
    if zipfile.is_zipfile(file_path):
        return read_csv_from_zip(file_path, file_name)
    else:
        return pd.read_csv(file_path)

def shuffle_dataframe(train_ds, shuffle_index):
    # shuffle shuffle_index
    random.shuffle(shuffle_index)

    sub_datasets = []
    for i, j in shuffle_index:
        sub_dataset = train_ds.iloc[i:j].sample(frac=1).reset_index(drop=True)
        sub_datasets.append(sub_dataset)
    shuffle_dataset = pd.concat(sub_datasets).reset_index(drop=True)
    return shuffle_dataset

# read dataset
def read_ds(
        train_ds_path,
        val_ds_path,
        test_ds_path,
        max_num_val: int=10000,
        max_num_test: int=2000,
        max_num_train: int=140000,
        shuffle_range: list=None,
):

    train_ds, val_ds, test_ds = None, None, None
    
    if train_ds_path and os.path.exists(train_ds_path):
        train_ds = get_file(
            file_path=train_ds_path,
            file_name="train.csv",
        )
    else:
        ValueError("Train dataset not found")

    if val_ds_path and os.path.exists(val_ds_path):
        val_ds = get_file(
            file_path=val_ds_path,
            file_name="val.csv",
        )
        if max_num_val < len(val_ds):
            val_ds = val_ds[:max_num_val]
    else:
        num_train = len(train_ds)
        num_val = min(int(num_train * 0.1), max_num_val)
        val_ds = train_ds[:num_val]
        train_ds = train_ds[num_val:]
        train_ds.reset_index(drop=True, inplace=True)
    
    if test_ds_path and os.path.exists(test_ds_path):
        test_ds = get_file(
            file_path=test_ds_path,
            file_name="test.csv",
        )
        if max_num_test < len(test_ds):
            test_ds = test_ds[:max_num_test]
    else:
        num_train = len(train_ds)
        test_ds = train_ds[:max_num_test]
        train_ds = train_ds[max_num_test:]
        train_ds.reset_index(drop=True, inplace=True)

    if len(train_ds) > max_num_train:
        train_ds = train_ds[:max_num_train]

    if shuffle_range is None:
        shuffle_index = [(0, len(train_ds))]
    else:
        shuffle_index = []
        for i in range(1, len(shuffle_range)):
            shuffle_range[i] += shuffle_range[i - 1]
        for i in range(0, len(shuffle_range)):
            if i == 0:
                shuffle_index.append((0, shuffle_range[i]))
            else:
                shuffle_index.append((shuffle_range[i - 1], shuffle_range[i]))
    print("Shuffle index: ", shuffle_index)
    train_ds = shuffle_dataframe(train_ds, shuffle_index)

    print("Read dataset successfully")
    print("Length train dataset: ", len(train_ds))
    print("Length val dataset: ", len(val_ds))
    print("Length test dataset: ", len(test_ds))
    print("====================================")

    return train_ds, val_ds, test_ds

# custom dataset
class Seq2seqDataset(Dataset):

    def __init__(self, ds: pd.DataFrame, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.lang_src = lang_src
        self.lang_tgt = lang_tgt

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds.iloc[idx]
        src_text = src_target_pair[self.lang_src]
        tgt_text = src_target_pair[self.lang_tgt]       

        return {
            'src_text': src_text,
            'tgt_text': tgt_text,
        }

# collate function
# define collate function
def collate_fn(batch, tokenizer_src, tokenizer_tgt):
    pad_token_id = tokenizer_src.token_to_id("<pad>")
    
    inputs_trainning_batch, inputs_inference_batch, labels_batch, src_text_batch, tgt_text_batch = [], [], [], [], []
    sos_token = torch.tensor([tokenizer_tgt.token_to_id("<s>")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("</s>")], dtype=torch.int64)
    sep_token = torch.tensor([tokenizer_tgt.token_to_id("<sep>")], dtype=torch.int64)

    for item in batch:
        src_text = item["src_text"]
        tgt_text = item["tgt_text"]

        enc_input_tokens = tokenizer_src.encode(src_text).ids
        dec_input_tokens = tokenizer_tgt.encode(tgt_text).ids

        # model llama 3 for text summarization
        inputs_trainning = torch.cat(
            [
                sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                sep_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        
        labels = torch.cat(
            [
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                sep_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                eos_token,
            ]
        )

        inputs_trainning_batch.append(inputs_trainning)
        labels_batch.append(labels)
        src_text_batch.append(src_text)
        tgt_text_batch.append(tgt_text)
        
     
    inputs_trainning_batch = pad_sequence(inputs_trainning_batch, padding_value=pad_token_id, batch_first=True)
    labels_batch = pad_sequence(labels_batch, padding_value=pad_token_id, batch_first=True) 
    
    return {
        'inputs_training': inputs_trainning_batch,
        'labels': labels_batch,
        'src_text': src_text_batch,
        'tgt_text': tgt_text_batch,
    }

# get dataloader dataset
def get_dataloader(
        tokenizer_src,
        tokenizer_tgt,
        batch_train,
        batch_val,
        batch_test,
        lang_src,
        lang_tgt,
        train_ds_path: str=None,
        val_ds_path: str=None,
        test_ds_path: str=None,
        max_num_val: int=15000,
        max_num_test: int=15000,
        max_num_train: int=200000,
        shuffle_range: list=None,
):
    train_ds, val_ds, test_ds = read_ds(
        train_ds_path=train_ds_path,
        val_ds_path=val_ds_path,
        test_ds_path=test_ds_path,
        max_num_val=max_num_val,
        max_num_test=max_num_test,
        max_num_train=max_num_train,
        shuffle_range=shuffle_range,
    )

    train_dataset, val_dataset, test_dataset = None, None, None
    train_dataloader, val_dataloader, test_dataloader = None, None, None

    train_dataset = Seq2seqDataset(
        ds=train_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        lang_src=lang_src,
        lang_tgt=lang_tgt,
    )

    val_dataset = Seq2seqDataset(
        ds=val_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        lang_src=lang_src,
        lang_tgt=lang_tgt,
    )

    test_dataset = Seq2seqDataset(
        ds=test_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        lang_src=lang_src,
        lang_tgt=lang_tgt,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_train,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
        )
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_val,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
        )
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_test,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
        )
    )
    
    ValueError("Dataloader not found")
    print("Get dataloader successfully")
    return train_dataloader, val_dataloader, test_dataloader