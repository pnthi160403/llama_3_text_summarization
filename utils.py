import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import torch
import zipfile
import evaluate
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn

def read(file_path):
    if not os.path.exists(file_path):
        return []
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            val = float(line.strip())
            data.append(val)
    return data

def write(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for value in data:
                file.write(f"{value}\n")
    except Exception as e:
        print(e)

def read_json(file_path: str=None):
    if file_path is not None and not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def write_json(file_path: str=None, data=None):
    if file_path is not None and data is not None:
        with open(file_path, "w") as f:
            json.dump(data, f)

def join_base(base_dir: str, path: str):
    return f"{base_dir}{path}"

# create dirs
def create_dirs(dir_paths: list):
    created_dirs = []
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        created_dirs.append(dir_path)
    
    # message
    print("Created:")
    for name_dir in created_dirs:
        print(name_dir)
    print("====================================")

# file path
def get_weights_file_path(model_folder_name: str, model_base_name: str, step: int):
    model_name = f"{model_base_name}{step:010d}.pt"
    return f"{model_folder_name}/{model_name}"

def weights_file_path(model_folder_name: str, model_base_name: str):
    model_filename = f"{model_base_name}*"
    weights_files = list(Path(model_folder_name).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return weights_files

class LossFigure:
    def __init__(
        self,
        xlabel: str,
        ylabel: str,
        title: str,
        loss_value_path: str,
        loss_step_path: str,
    ):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self.loss_value_path = loss_value_path
        self.loss_step_path = loss_step_path
        self.loss_value = []
        self.loss_step = []
        if os.path.exists(loss_value_path) and os.path.exists(loss_step_path):
            self.loss_value = read(loss_value_path)
            self.loss_step = read(loss_step_path)

    def update(
        self,
        value: float,
        step: int,
    ):
        if len(self.loss_step) != 0 and step < self.loss_step[-1] and step >= 0:
            find_index = self.loss_step.index(step)
            self.loss_value[find_index] = value
        else:
            self.loss_value.append(value)
            self.loss_step.append(step)

    def save(self):
        write(self.loss_value_path, self.loss_value)
        write(self.loss_step_path, self.loss_step)

    def load(self):
        self.loss_value = read(self.loss_value_path)
        self.loss_step = read(self.loss_step_path)

# figures
def draw_graph(config, title, xlabel, ylabel, data, steps, log_scale=True):
    try:
        save_path = join_base(config['log_dir'], f"/{title}.png")
        plt.plot(steps, data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if log_scale:
            plt.yscale('log')
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
        plt.close()
    except Exception as e:
        print(e)

def draw_multi_graph(config, title, xlabel, ylabel, all_data, steps):
    try:
        save_path = join_base(config['log_dir'], f"/{title}.png")
        for data, info in all_data:
            plt.plot(steps, data, label=info)
            # add multiple legends
            plt.legend()

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
        plt.close()
    except Exception as e:
        print(e)

def figure_list_to_csv(config, column_names, data, name_csv):
    try:
        obj = {}
        for i in range(len(column_names)):
            if data[i] is not None:
                obj[str(column_names[i])] = data[i]

        data_frame = pd.DataFrame(obj, index=[0])
        save_path = join_base(config['log_dir'], f"/{name_csv}.csv")
        data_frame.to_csv(save_path, index=False)
        return data_frame
    except Exception as e:
        print(e)

def zip_directory(directory_path, output_zip_path):
    print(f"{ directory_path } -> { output_zip_path }")
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to zip, preserving the directory structure
                arcname = os.path.relpath(file_path, start=directory_path)
                zipf.write(file_path, arcname)

# save model
def save_model(model, global_step, global_val_step, optimizer, lr_scheduler, model_folder_name, model_base_name):
    model_filename = get_weights_file_path(
        model_folder_name=model_folder_name,
        model_base_name=model_base_name,    
        step=global_step
    )

    torch.save({
        "global_step": global_step,
        "global_val_step": global_val_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict()
    }, model_filename)
    
    print(f"Saved model at {model_filename}")

# save config
def save_config(config: dict, global_step: int):
    config_filename = f"{config['config_dir']}/config_{global_step:010d}.json"
    with open(config_filename, "w") as f:
        json.dump(config, f)
    print(f"Saved config at {config_filename}")
   
# metrics   
def compute_rouges(preds, refs):
    rouge = evaluate.load('rouge')
    res = rouge.compute(predictions=preds, references=refs)
    ans = {
        "rouge1_fmeasure": res['rouge1'],
        "rouge2_fmeasure": res['rouge2'],
        "rougeL_fmeasure": res['rougeL'],
    }
    return ans

# optimizers
def get_AdamW(
    model,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    
# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
# model
def show_layer_un_freeze(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def freeze_model(model, modules=[]):
    for module in modules:
        for name, param in module.named_parameters():
            param.requires_grad = False
    return model

def un_freeze_model(model, modules=[]):
    for module in modules:
        for name, param in module.named_parameters():
            param.requires_grad = True
    return model

# load model state dict
def load_model(checkpoint, model):
    if torch.cuda.is_available():
        state = torch.load(checkpoint)
    else:
        state = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(state["model_state_dict"])
    return model

def calc_consine_similarity(
    E: torch.Tensor,
    vocab_size: int,
    k: int,
    eos_token_id: int,
) -> torch.Tensor:
    # E (vocab_size, d_model)
    # k: number of top cosine similarity indices to return
    top_cosine_similarity_indices = None
    for i in tqdm(range(vocab_size)):
        # (vocab_size, d_model)
        embed_i = E[i].unsqueeze(0).repeat(vocab_size, 1)
        cosine_similarities = nn.functional.cosine_similarity(
            x1=E,
            x2=embed_i,
            dim=-1,
        )
        val, idx = torch.topk(
            input=cosine_similarities,
            k=k,
        )
        if top_cosine_similarity_indices is None:
            top_cosine_similarity_indices = idx.unsqueeze(0)
        else:
            top_cosine_similarity_indices = torch.cat([
                top_cosine_similarity_indices,
                idx.unsqueeze(0),
            ], dim=0)
    return top_cosine_similarity_indices

def get_cosine_similarity(
    path: str,
    vocab_size: int,
    k: int,
    decoder_embeds_matrix: torch.Tensor=None,
    eos_token_id: int=None,
):
    if path is not None and os.path.exists(path):
        top_cosine_similarity_indices = torch.load(path)
    else:
        top_cosine_similarity_indices = calc_consine_similarity(
            E=decoder_embeds_matrix,
            vocab_size=vocab_size,
            k=k,
            eos_token_id=eos_token_id,
        )
        if path is not None:
            torch.save(
                obj=top_cosine_similarity_indices,
                f=path,
            )
    return top_cosine_similarity_indices