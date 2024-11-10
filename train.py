import torch
from .data import get_dataloader, read_tokenizer
from .utils import (
    draw_graph,
    draw_multi_graph,
    zip_directory,
    create_dirs,
    get_weights_file_path,
    weights_file_path,
    set_seed,
    get_AdamW,
    LossFigure,
)
from .model import Transformer, ModelArgs
from .trainers import (
    BartTrainerSingleGPU,
)

# get optimizer lambda lr
def lambda_lr(global_step: int, config):
    global_step = max(global_step, 1)
    return (config["d_model"] ** -0.5) * min(global_step ** (-0.5), global_step * config["warmup_steps"] ** (-1.5))

def train(config):
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
    config["vocab_size"] = config["tgt_vocab_size"]
    # model config
    ModelArgs.dim = config["dim"]
    ModelArgs.n_layers = config["n_layers"]
    ModelArgs.n_heads = config["n_heads"]
    ModelArgs.n_kv_heads = config["n_kv_heads"]
    ModelArgs.multiple_of = config["multiple_of"]
    ModelArgs.ffn_dim_multiplier = config["ffn_dim_multiplier"]
    ModelArgs.norm_eps = config["norm_eps"]
    ModelArgs.rope_theta = config["rope_theta"]
    ModelArgs.max_batch_size = config["max_batch_size"]
    ModelArgs.max_seq_len = config["max_seq_len"]
    ModelArgs.label_smoothing = config["label_smoothing"]
    ModelArgs.layer_dropout = config["layer_dropout"]
    ModelArgs.device = config["device"]
    ModelArgs.pad_token_id = tokenizer_src.token_to_id("<pad>")
    ModelArgs.vocab_size = config["vocab_size"]
    # get model
    model = Transformer(
        ModelArgs,
    ).to(ModelArgs.device)
    print(f"Model: {model}")
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
    # optimizer
    optimizer = get_AdamW(
        model=model,
        lr=config["lr"],
        eps=config["eps"],
        weight_decay=config["weight_decay"],
        betas=config["betas"]
    )
    preload = config["preload"]
    # # load model
    model_folder_name=config["model_folder_name"]
    model_base_name=config["model_base_name"]
    weights_files = weights_file_path(
        model_folder_name=model_folder_name,
        model_base_name=model_base_name,
    )
    if weights_files is not None:
        model_filename = (str(weights_files[-1]) if preload == 'latest' else get_weights_file_path(
            model_folder_name=model_folder_name,
            model_base_name=model_base_name,
        )) if preload else None
    else:
        model_filename = None
    state = None
    if model_filename:
        state = torch.load(model_filename, map_location=device)
        print(f"Loaded model from {model_filename}")
    else:
        print("No model to preload, start training from scratch")
        
    # # loss figures
    # # train step figures
    loss_train_step_figure = LossFigure(
        xlabel="Step",
        ylabel="Loss value",
        title="Loss train",
        loss_value_path=config["step_loss_train_value_path"],
        loss_step_path=config["step_loss_train_step_path"],
    )
    # val step figures
    loss_val_step_figure = LossFigure(
        xlabel="Step",
        ylabel="Loss value",
        title="Loss val",
        loss_value_path=config["step_loss_val_value_path"],
        loss_step_path=config["step_loss_val_step_path"],
    )
    # train epoch figures
    loss_train_epoch_figure = LossFigure(
        xlabel="Epoch",
        ylabel="Loss value",
        title="Loss train",
        loss_value_path=config["epoch_loss_train_value_path"],
        loss_step_path=config["epoch_loss_train_step_path"],
    )
    # val epoch figures
    loss_val_epoch_figure = LossFigure(
        xlabel="Epoch",
        ylabel="Loss value",
        title="Loss val",
        loss_value_path=config["epoch_loss_val_value_path"],
        loss_step_path=config["epoch_loss_val_step_path"],
    )
    # rouge_1 epoch figures
    rouge_1_epoch_figure = LossFigure(
        xlabel="Epoch",
        ylabel="Rouge 1",
        title="Rouge 1",
        loss_value_path=config["epoch_rouge_1_value_path"],
        loss_step_path=config["epoch_rouge_1_step_path"],
    )
    # rouge_2 epoch figures
    rouge_2_epoch_figure = LossFigure(
        xlabel="Epoch",
        ylabel="Rouge 2",
        title="Rouge 2",
        loss_value_path=config["epoch_rouge_2_value_path"],
        loss_step_path=config["epoch_rouge_2_step_path"],
    )
    # rouge_l epoch figures
    rouge_l_epoch_figure = LossFigure(
        xlabel="Epoch",
        ylabel="Rouge L",
        title="Rouge L",
        loss_value_path=config["epoch_rouge_l_value_path"],
        loss_step_path=config["epoch_rouge_l_step_path"],
    )

    # # train model
    trainer = BartTrainerSingleGPU(
        config=config,
        model=model,
        optimizer=optimizer,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        loss_train_step_figure=loss_train_step_figure,
        loss_val_step_figure=loss_val_step_figure,
        loss_train_epoch_figure=loss_train_epoch_figure,
        loss_val_epoch_figure=loss_val_epoch_figure,
        rouge_1_epoch_figure=rouge_1_epoch_figure,
        rouge_2_epoch_figure=rouge_2_epoch_figure,
        rouge_l_epoch_figure=rouge_l_epoch_figure,
        model_folder_name=model_folder_name,
        model_base_name=model_base_name,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        state=state,
    )
    trainer.train_loop()

    # draw graph loss
    # train and val
    draw_multi_graph(
        config=config,
        xlabel="Step",
        ylabel="Loss value",
        title="Loss",
        all_data=[
            (trainer.loss_train_epoch_figure.loss_value, "Train"),
            (trainer.loss_val_epoch_figure.loss_value, "Val")
        ],
        steps=trainer.loss_train_epoch_figure.loss_step,
    )
    # train step
    draw_graph(
        config=config,
        title="Loss train",
        xlabel="Step",
        ylabel="Loss value",
        data=trainer.loss_train_step_figure.loss_value,
        steps=trainer.loss_train_step_figure.loss_step,
    )

    # val step
    draw_graph(
        config=config,
        title="Loss val",
        xlabel="Step",
        ylabel="Loss value",
        data=trainer.loss_val_step_figure.loss_value,
        steps=trainer.loss_val_step_figure.loss_step,
    )

    # rouge 1
    draw_graph(
        config=config,
        title="Rouge 1",
        xlabel="Epoch",
        ylabel="Rouge 1",
        data=trainer.rouge_1_epoch_figure.loss_value,
        steps=trainer.rouge_1_epoch_figure.loss_step,
        log_scale=False,
    )

    # rouge 2
    draw_graph(
        config=config,
        title="Rouge 2",
        xlabel="Epoch",
        ylabel="Rouge 2",
        data=trainer.rouge_2_epoch_figure.loss_value,
        steps=trainer.rouge_2_epoch_figure.loss_step,
        log_scale=False,
    )

    # rouge l
    draw_graph(
        config=config,
        title="Rouge L",
        xlabel="Epoch",
        ylabel="Rouge L",
        data=trainer.rouge_l_epoch_figure.loss_value,
        steps=trainer.rouge_l_epoch_figure.loss_step,
        log_scale=False,
    )

    # zip directory
    zip_directory(
        directory_path=config["log_dir"],
        output_zip_path=config["log_dir_zip"],
    )
    zip_directory(
        directory_path=config["config_dir"],
        output_zip_path=config["config_dir_zip"],
    )
    zip_directory(
        directory_path=config["model_folder_name"],
        output_zip_path=config["model_folder_name_zip"],
    )