import torch
from .utils import join_base

def get_config(base_dir: str=None):
    config = {}
    # tmp
    config["use_generate"] = True

    if not base_dir:
        config["base_dir"] = "./"
    else:
        config["base_dir"] = base_dir

    # Tokenizer
    config['tokenizer_tgt_path'] = None
    config['tokenizer_src_path'] = None
    config["use_tokenizer"] = "huggingface"
    config["special_tokens"] = [
      "<s>",
      "</s>",
      "<pad>",
      "<unk>",
      "<mask>",
      "<cls>",
      "<sep>",
    ]
    config["vocab_size"] = 30000
    config['min_frequency'] = 2

    # Directories
    config["model_folder_name"] = join_base(config["base_dir"], "/model")
    config["model_folder_name_zip"] = join_base(config["base_dir"], "/model.zip")
    config["model_base_name"] = "model_"
    config["model_out"] = "out_"
    
    config["preload"] = "latest"
    config["data"] = join_base(config["base_dir"], "/data")
    config["config_dir"] = join_base(config["base_dir"], "/config")
    config["config_dir_zip"] = join_base(config["base_dir"], "/config.zip")
    config["generated_dir"] = join_base(config["base_dir"], "/generated")
    config["log_dir"] = join_base(config["base_dir"], "/log")
    config["log_dir_zip"] = join_base(config["base_dir"], "/log.zip")
    config["log_files"] = join_base(config["log_dir"], "/log_files")
    config["step_loss_train_value_path"] = join_base(config["log_files"], "/step_loss_train_value.json")
    config["step_loss_train_step_path"] = join_base(config["log_files"], "/step_loss_train_step.json")
    config["step_loss_val_value_path"] = join_base(config["log_files"], "/step_loss_val_value.json")
    config["step_loss_val_step_path"] = join_base(config["log_files"], "/step_loss_val_step.json")
    config["epoch_loss_train_value_path"] = join_base(config["log_files"], "/epoch_loss_train_value.json")
    config["epoch_loss_train_step_path"] = join_base(config["log_files"], "/epoch_loss_train_step.json")
    config["epoch_loss_val_value_path"] = join_base(config["log_files"], "/epoch_loss_val_value.json")
    config["epoch_loss_val_step_path"] = join_base(config["log_files"], "/epoch_loss_val_step.json")
    config["epoch_rouge_1_value_path"] = join_base(config["log_files"], "/epoch_rouge_1_value.json")
    config["epoch_rouge_1_step_path"] = join_base(config["log_files"], "/epoch_rouge_1_step.json")
    config["epoch_rouge_2_value_path"] = join_base(config["log_files"], "/epoch_rouge_2_value.json")
    config["epoch_rouge_2_step_path"] = join_base(config["log_files"], "/epoch_rouge_2_step.json")
    config["epoch_rouge_l_value_path"] = join_base(config["log_files"], "/epoch_rouge_l_value.json")
    config["epoch_rouge_l_step_path"] = join_base(config["log_files"], "/epoch_rouge_l_step.json")

    # Dataset
    config["lang_src"] = "content"
    config["lang_tgt"] = "summary"
    config["train_ds_path"] = None
    config["val_ds_path"] = None
    config["test_ds_path"] = None
    config["max_num_test"] = 2000
    config["max_num_val"] = 10000
    config["max_num_train"] = 200000
    config["corpus"] = None
    config["max_len"] = 3072
    config["src_vocab_size_bart_encoder"] = 30000
    config["seed"] = 42
    config["shuffle_range"] = None

    # Train
    config["model_train"] = "bart"
    config["step_train"] = None
    config["preload"] = "latest"
    config["pretrain"] = False
    config["continue_step"] = False
    
    # Trainning loop
    config["batch_train"] = 32
    config["batch_val"] = 1
    config["batch_test"] = 1
    config["max_global_step"] = 1000000000
    config["max_epoch"] = 1
    config["step_accumulation"] = 1

    # Llama 3 config
    config["dim"] = 768
    config["n_layers"] = 6
    config["n_heads"] = 12
    config["n_kv_heads"] = 4
    config["vocab_size"] = 50265
    config["multiple_of"] = 256
    config["ffn_dim_multiplier"] = None
    config["norm_eps"] = 1e-5
    config["rope_theta"] = 10000.0
    config["max_batch_size"] = 4
    config["max_seq_len"] = 3072
    config["label_smoothing"] = 0.01
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GELU activation function
    config["approximate_gelu"] = 'none'

    # Search module
    config["type_search"] = "diverse_beam_search"
    config["beams"] = [2]
    config["num_groups_search"] = 1
    config["diversity_strength_search"] = 0.5
    config["diversity_discount_search"] = 0.5
    config["candidate_multiple_search"] = 1
    config["n_gram_search"] = 1
    config["type_diversity_function"] = "Hamming_Cumulative"
    config["cosine_similarity_path"] = None
    config["top_k_cosine_similarity"] = 4

    # Optimizer
    config["weight_decay"] = 0.0
    config["lr"] = 0.5
    config["eps"] = 1e-9
    config["betas"] = (0.9, 0.98)

    # Scheduler
    config["use_scheduler"] = True
    config["warmup_steps"] = 4000

    # Device
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Metric
    config["f_beta"] = 0.5
    config["use_pytorch_metric"] = False
    config["use_bleu"] = True
    config["use_recall"] = False
    config["use_precision"] = False
    config["use_rouge"] = False
    config["sub_test_id"] = 0

    return config