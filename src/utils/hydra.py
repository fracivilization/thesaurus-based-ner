from typing import Optional
from transformers import training_args
from transformers.training_args import TrainingArguments as OrigTrainingArguments
from dataclasses import dataclass, field
import dataclasses
from transformers.trainer_utils import (
    IntervalStrategy,
    SchedulerType,
)
import os
from hashlib import md5


@dataclass
class HydraAddaptedTrainingArguments(OrigTrainingArguments):
    eval_steps: Optional[int] = field(
        default=None, metadata={"help": "Run an evaluation every X steps."}
    )
    push_to_hub_organization: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the organization in with to which push the `Trainer`."
        },
    )
    push_to_hub_token: Optional[str] = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )
    evaluation_strategy: IntervalStrategy = field(
        default=IntervalStrategy.NO,
        metadata={"help": "The evaluation strategy to use."},
    )
    lr_scheduler_type: SchedulerType = field(
        default=SchedulerType.LINEAR,
        metadata={"help": "The scheduler type to use."},
    )
    logging_strategy: IntervalStrategy = field(
        default=IntervalStrategy.STEPS,
        metadata={"help": "The logging strategy to use."},
    )
    save_strategy: IntervalStrategy = field(
        default=IntervalStrategy.STEPS,
        metadata={"help": "The checkpoint save strategy to use."},
    )
    push_to_hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the repository to which push the `Trainer`."},
    )

    def __post_init__(self):
        """Adapt hydra because some parameters (e.g. debug) is changed in those type
        after init train args by __post_init__ of parent class
        Please pass this "raw" TrainingArguments into original TrainingArguments"""
        pass


def get_orig_transoformers_train_args_from_hydra_addapted_train_args(
    train_args: HydraAddaptedTrainingArguments,
):

    # NOTE: hydra経由で呼ぶ場合にはtrain_argsはdata_classではない
    if dataclasses.is_dataclass(train_args):
        train_args_dict = {
            k: v for k, v in dataclasses.asdict(train_args).items() if k != "_n_gpu"
        }
    else:
        train_args_dict = {k: v for k, v in train_args.items() if k != "_n_gpu"}

    train_args = OrigTrainingArguments(**train_args_dict)
    return train_args
