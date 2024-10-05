import torch
from .generate import generate
from .utils import (
    write_json,
    read_json,
    zip_directory,
)
from .search_algos import (
    DIVERSE_BEAM_SEARCH,
    BEAM_SEARCH,
    NEURAL_EMBEDDING_TYPE_DIVERSITY,
)
from .utils import (
    compute_rouges,
)
from .data import read_tokenizer
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import sys

def validate(model, config, beam_size, val_dataloader, num_example: int=5):
    model.eval()
    device = config["device"]
    # get sub test id
    sub_test_id = config["sub_test_id"]
    
    # read tokenizer
    tokenizer_src, tokenizer_tgt = read_tokenizer(
        tokenizer_src_path=config["tokenizer_src_path"],
        tokenizer_tgt_path=config["tokenizer_tgt_path"],
    )

    vocab_size = tokenizer_tgt.get_vocab_size()
    pad_token_id = tokenizer_src.token_to_id("<pad>")
    sep_token_id = tokenizer_src.token_to_id("<sep>")

    with torch.no_grad():

        source_texts = []
        expected = []
        predicted = []

        count = 0

        labels = []
        preds = []

        rouge_preds = []
        rouge_targets = []

        bleus = None
        
        batch_iterator = tqdm(val_dataloader, desc=f"Testing model...")
        for batch in batch_iterator:
            src_text = batch["src_text"][0]
            tgt_text = batch["tgt_text"][0]

            preds_ids = generate(
                model=model,
                config=config,
                beam_size=beam_size,
                tokenizer_src=tokenizer_src,
                tokenizer_tgt=tokenizer_tgt,
                src=src_text,
            )
            for i in range(len(preds_ids)):
                index_step_token_id = 0
                for j in range(len(preds_ids[i].tgt)):
                    if preds_ids[i].tgt[j] == sep_token_id:   
                        index_step_token_id = j
                        break
                preds_ids[i].tgt = preds_ids[i].tgt[index_step_token_id:]
                
            if config["type_search"] in [BEAM_SEARCH, DIVERSE_BEAM_SEARCH]:
                pred_ids = preds_ids[0].tgt.squeeze()
                
            pred_text = tokenizer_tgt.decode(
                pred_ids.detach().cpu().numpy(),
                skip_special_tokens=True,
            )

            rouge_preds.append(pred_text)
            rouge_targets.append(tgt_text)  
            
            pred_ids = torch.tensor(tokenizer_tgt.encode(pred_text).ids, dtype=torch.int64).to(device)
            label_ids = torch.tensor(tokenizer_tgt.encode(tgt_text).ids, dtype=torch.int64).to(device)

            padding = pad_sequence([label_ids, pred_ids], padding_value=pad_token_id, batch_first=True)
            label_ids = padding[0]
            pred_ids = padding[1]
            
            labels.append(label_ids)
            preds.append(pred_ids)

            source_texts.append(tokenizer_src.encode(src_text).tokens)
            expected.append([tokenizer_tgt.encode(tgt_text).tokens])
            predicted.append(tokenizer_tgt.encode(pred_text).tokens)

            count += 1

            print_step = max(len(val_dataloader) // num_example, 1)
            
            if count % print_step == 0:
                print()
                print(f"{f'SOURCE: ':>12}{src_text}")
                print(f"{f'TARGET: ':>12}{tgt_text}")
                for i in range(len(preds_ids)):
                    text = tokenizer_tgt.decode(
                        preds_ids[i].tgt.squeeze().detach().cpu().numpy(),
                        skip_special_tokens=True,
                    )
                    print(f"{f'PREDICTED {i}: ':>12}{text}")
            
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)

        recall, precision, rouges = None, None, None
        
        write_json(
            file_path=config["generated_dir"] + f"/rouge_preds_{sub_test_id}.json",
            data=rouge_preds,
        )
        write_json(
            file_path=config["generated_dir"] + f"/rouge_targets_{sub_test_id}.json",
            data=rouge_targets,
        )

        rouges = compute_rouges(
            preds=rouge_preds,
            refs=rouge_targets,
        )
        zip_directory(
            directory_path=config["generated_dir"],
            output_zip_path=f"{config['generated_dir']}_{config['sub_test_id']}.zip",
        )
        
        res = {}
        if rouges is not None and config["use_rouge"]:
            for key, val in rouges.items():
                res[key] = val.item()
        return [res]