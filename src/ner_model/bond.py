import os
from datasets import DatasetDict
from .bert import BERTNERModelBase
from pathlib import Path
from hashlib import md5
from dataclasses import dataclass
import torch
import logging
from torch.nn import CrossEntropyLoss
from hydra.core.config_store import ConfigStore
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForTokenClassification,
    AutoTokenizer,
    pipeline,
)
from torch.nn import DataParallel
import datasets
from .bond_lib.train import train
import random
import numpy as np
from .bond_lib.modelling_bert import BertForTokenClassification
from .bond_lib.modeling_roberta import RobertaForTokenClassification
from .abstract_model import NERModelConfig
from omegaconf import MISSING

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "distilbert": (
        DistilBertConfig,
        DistilBertForTokenClassification,
        DistilBertTokenizer,
    ),
    "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    "xlmroberta": (
        XLMRobertaConfig,
        XLMRobertaForTokenClassification,
        XLMRobertaTokenizer,
    ),
}


@dataclass
class BONDArgs(NERModelConfig):
    model_type: str = MISSING
    model_name_or_path: str = MISSING
    ner_model_name: str = "BOND"
    # Other parameters
    config_name: str = ""
    tokenizer_name: str = ""
    cache_dir: str = ""
    # max_seq_length: int = 128
    max_length: int = 128
    do_train: bool = False
    do_eval: bool = False
    do_predict: bool = False
    evaluate_during_training: bool = False
    do_lower_case: bool = False
    per_gpu_train_batch_size: int = 8
    per_gpu_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    warmup_steps: int = 0
    logging_steps: int = 50
    save_steps: int = 50
    eval_all_checkpoints: bool = False
    no_cuda: bool = False
    overwrite_output_dir: bool = False
    overwrite_cache: bool = False
    seed: int = 42
    fp16: bool = False
    fp16_opt_level: str = "O1"
    local_rank: int = -1
    server_ip: str = ""
    server_port: str = ""
    # mean teacher
    mt: int = 0
    mt_updatefreq: int = 1
    mt_class: str = "kl"
    mt_lambda: float = 1
    mt_rampup: int = 300
    mt_alpha1: float = 0.99
    mt_alpha2: float = 0.995
    mt_beta: float = 10
    mt_avg: str = "exponential"
    mt_loss_type: str = "logits"

    # virtual adversarial training
    vat: int = 0
    vat_eps: float = 1e-3
    vat_lambda: float = 1
    vat_beta: float = 1
    vat_loss_type: str = "logits"

    # self-training
    self_training_reinit: int = 0
    self_training_begin_step: int = 900
    self_training_label_mode: str = "hard"
    self_training_period: int = 878
    self_training_hp_label: float = 0
    self_training_ensemble_label: int = 0

    pad_to_max_length: bool = False
    max_length: int = 512
    label_all_tokens: bool = False
    # For term-dropout
    term_dropout_ratio: float = 0.5
    saved_param_path: str = ""


def register_BOND_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="ner_model", name="base_BOND_model_config", node=BONDArgs)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class BONDNERModel(BERTNERModelBase):
    def __init__(
        self,
        ner_dataset: DatasetDict,
        bond_args: BONDArgs = BONDArgs(
            model_type="bert", model_name_or_path="bert-base-uncased"
        ),
    ) -> None:
        super().__init__()
        bond_args = BONDArgs(**bond_args)
        # Setup CUDA, GPU & distributed training
        if bond_args.local_rank == -1 or bond_args.no_cuda:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and not bond_args.no_cuda else "cpu"
            )
            bond_args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(bond_args.local_rank)
            device = torch.device("cuda", bond_args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            bond_args.n_gpu = 1
        bond_args.device = device

        self.conf["datasets"] = {
            key: split.__hash__() for key, split in ner_dataset.items()
        }
        self.conf["bond_args"] = bond_args
        output_dir = Path("data/output").joinpath(
            md5(str(self.conf).encode()).hexdigest()
        )
        self.output_dir = output_dir
        logger.info("output_dir is %s" % str(output_dir))
        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and bond_args.do_train
            and not bond_args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    output_dir
                )
            )

        # Create output directory if needed
        if not os.path.exists(output_dir) and bond_args.local_rank in [-1, 0]:
            os.makedirs(output_dir)

        # Setup CUDA, GPU & distributed training
        if bond_args.local_rank == -1 or bond_args.no_cuda:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and not bond_args.no_cuda else "cpu"
            )
            bond_args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(bond_args.local_rank)
            device = torch.device("cuda", bond_args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            bond_args.n_gpu = 1
        bond_args.device = device
        self.bond_args = bond_args
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if bond_args.local_rank in [-1, 0] else logging.WARN,
        )
        logging_fh = logging.FileHandler(os.path.join(output_dir, "log.txt"))
        logging_fh.setLevel(logging.DEBUG)
        logger.addHandler(logging_fh)
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            bond_args.local_rank,
            device,
            bond_args.n_gpu,
            bool(bond_args.local_rank != -1),
            bond_args.fp16,
        )

        # Set seed
        set_seed(bond_args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = CrossEntropyLoss().ignore_index
        self.pad_token_label_id = pad_token_label_id
        # Load pretrained model and tokenizer
        if bond_args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        bond_args.model_type = bond_args.model_type.lower()
        tokenizer = AutoTokenizer.from_pretrained(
            bond_args.tokenizer_name
            if bond_args.tokenizer_name
            else bond_args.model_name_or_path,
            # cache_dir=bond_args.cache_dir,
            use_fast=True,
            add_prefix_space=True,
        )

        column_names = ner_dataset["train"].column_names
        features = ner_dataset["train"].features
        text_column_name = "tokens" if "tokens" in column_names else column_names[0]
        padding = "max_length" if bond_args.pad_to_max_length else False

        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        label_column_name = "ner_tags"
        if isinstance(features[label_column_name].feature, datasets.ClassLabel):
            label_list = features[label_column_name].feature.names
            # No need to convert the labels since they are already ints.
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            label_list = get_label_list(ner_dataset["train"][label_column_name])
            label_to_id = {l: i for i, l in enumerate(label_list)}
        self.label_list = label_list

        num_labels = len(label_list)
        o_label_id = ner_dataset["train"].features["ner_tags"].feature.names.index("O")

        config_class, model_class, tokenizer_class = MODEL_CLASSES[bond_args.model_type]
        self.model_class = model_class
        config = config_class.from_pretrained(
            bond_args.config_name
            if bond_args.config_name
            else bond_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=bond_args.cache_dir if bond_args.cache_dir else None,
            term_dropout_ratio=bond_args.term_dropout_ratio,
        )
        self.config = config

        # Tokenize all texts and align the labels with them.
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
                max_length=bond_args.max_length,
            )
            labels = []
            for i, label in enumerate(examples[label_column_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label_to_id[label[word_idx]])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(
                            label_to_id[label[word_idx]]
                            if bond_args.label_all_tokens
                            else -100
                        )
                    previous_word_idx = word_idx

                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        ner_dataset = DatasetDict(
            {"train": ner_dataset["train"], "validation": ner_dataset["validation"]}
        )
        self.tokenized_datasets = ner_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            # num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not bond_args.overwrite_cache,
        )

        # tokenizer = tokenizer_class.from_pretrained(
        #     bond_args.tokenizer_name
        #     if bond_args.tokenizer_name
        #     else bond_args.model_name_or_path,
        #     do_lower_case=bond_args.do_lower_case,
        #     cache_dir=bond_args.cache_dir if bond_args.cache_dir else None,
        # )

        if bond_args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        logger.info("Training/evaluation parameters %s", bond_args)

        model = model_class.from_pretrained(
            bond_args.model_name_or_path,
            from_tf=bool(".ckpt" in bond_args.model_name_or_path),
            config=config,
            cache_dir=bond_args.cache_dir if bond_args.cache_dir else None,
        )
        model.to(bond_args.device)
        # Training
        if bond_args.saved_param_path != "":
            model.load_state_dict(torch.load(bond_args.saved_param_path))

        if isinstance(model, DataParallel):
            model_for_eval = model.module
        else:
            model_for_eval = model

        self.model = model
        self.tokenizer = tokenizer
        if isinstance(model.device.index, int):
            self.pipeline = pipeline(
                "ner",
                model=model_for_eval,
                tokenizer=tokenizer,
                device=model.device.index,
            )
        else:
            self.pipeline = pipeline(
                "ner",
                model=model_for_eval,
                tokenizer=tokenizer,
            )

    def train(self):
        if self.bond_args.do_train:
            # train_dataset = load_and_cache_examples(
            #     bond_args, tokenizer, labels, pad_token_label_id, mode="train"
            # )

            model, global_step, tr_loss, best_dev, best_test = train(
                self.bond_args,
                self.tokenized_datasets,
                self.model_class,
                self.config,
                self.tokenizer,
                self.label_list,
                self.pad_token_label_id,
                self.output_dir,
                self.model,
            )
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving last-practice: if you use defaults names for the model, you can reload it using from_pretrained()
        if self.bond_args.do_train and (
            self.bond_args.local_rank == -1 or torch.distributed.get_rank() == 0
        ):
            logger.info("Saving model checkpoint to %s", self.output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            torch.save(
                self.bond_args, os.path.join(self.output_dir, "training_args.bin")
            )
            if isinstance(model, DataParallel):
                torch.save(
                    model.module.state_dict(),
                    os.path.join(self.output_dir, "pytorch_model.bin"),
                )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(self.output_dir, "pytorch_model.bin"),
                )
