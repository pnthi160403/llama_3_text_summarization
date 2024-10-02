import torch
from .val import validate
from .utils import (
    set_seed,
    create_dirs,
    weights_file_path,
    figure_list_to_csv,
    zip_directory,
)
from .data import (
    read_tokenizer,
    get_dataloader,
)
from .model import Transformer, ModelArgs

def test(config):
    # create dirs
    create_dirs(dir_paths=[config["log_dir"], config["model_folder_name"], config["log_files"], config["config_dir"], config["generated_dir"]])
    # set seed
    set_seed(seed=config["seed"])
    # device
    device = config["device"]
    # read tokenizer
    tokenizer_src, tokenizer_tgt = read_tokenizer(
        tokenizer_src_path=config["tokenizer_src_path"],
        tokenizer_tgt_path=config["tokenizer_tgt_path"],
    )
    config["src_vocab_size"] = tokenizer_src.get_vocab_size()
    config["tgt_vocab_size"] = tokenizer_tgt.get_vocab_size()
    # beams
    beams = config["beams"]
    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        batch_train=config["batch_train"],
        batch_val=config["batch_val"],
        batch_test=config["batch_test"],
        lang_src=config["lang_src"],
        lang_tgt=config["lang_tgt"],
        train_ds_path=config["train_ds_path"],
        val_ds_path=config["val_ds_path"],
        test_ds_path=config["test_ds_path"],
        max_num_val=config["max_num_val"],
        max_num_test=config["max_num_test"],
        max_num_train=config["max_num_train"],
        shuffle_range=config["shuffle_range"],
    )
    # model config
    ModelArgs.dim = config["dim"]
    ModelArgs.n_layers = config["n_layers"]
    ModelArgs.n_heads = config["n_heads"]
    ModelArgs.n_kv_heads = config["n_kv_heads"]
    ModelArgs.vocab_size = config["vocab_size"]
    ModelArgs.multiple_of = config["multiple_of"]
    ModelArgs.ffn_dim_multiplier = config["ffn_dim_multiplier"]
    ModelArgs.norm_eps = config["norm_eps"]
    ModelArgs.rope_theta = config["rope_theta"]
    ModelArgs.max_batch_size = config["max_batch_size"]
    ModelArgs.max_seq_len = config["max_seq_len"]
    ModelArgs.label_smoothing = config["label_smoothing"]
    ModelArgs.device = config["device"]
    ModelArgs.pad_token_id = tokenizer_src.token_to_id("<pad>")
    # get model
    model = Transformer(
        ModelArgs,
    ).to(ModelArgs.device)
    print(f"Model: {model}")
        
    model_filenames = weights_file_path(
        model_folder_name=config["model_folder_name"],
        model_base_name=config["model_base_name"],
    )
    model_filename = model_filenames[-1]

    if model_filename:
        print(f"Preloading model from {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    else:
        print("No model to preload!")

    for beam_size in beams:
        ans = validate(
            model=model,
            config=config,
            beam_size=beam_size,
            val_dataloader=test_dataloader
        )
        for i in range(len(ans)):
            res = ans[i]
            column_names = []
            data = []
            for name, value in res.items():
                if value is None:
                    continue
                column_names.append(name)
                data.append(value)

            data_frame = figure_list_to_csv(
                config=config,
                column_names=column_names,
                data=data,
                name_csv=f"results_beam_{beam_size}_prediction_{i}"
            )

            zip_directory(
                directory_path=config["log_dir"],
                output_zip_path=config["log_dir_zip"]
            )

            print(f"Result test model in prediction {i} with beam size {beam_size}")
            print(data_frame)