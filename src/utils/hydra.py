from typing import Optional
from transformers import training_args
from transformers.training_args import TrainingArguments as OrigTrainingArguments
from dataclasses import dataclass, field
from transformers.trainer_utils import (
    IntervalStrategy,
    SchedulerType,
)
from dataclasses import asdict, dataclass


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
    # NOTE: 元の型は debug: Union[str, List[DebugOption]] だが UnionにOmegaConfが対応してないので上書きする
    debug: str = field( 
        default="",
        metadata={
            "help": (
                "Whether or not to enable debug mode. Current options: "
                "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
                "`tpu_metrics_debug` (print debug metrics on TPU)."
            )
        },
    )
    # NOTE: 元の型は Optional[Union[List[ShardedDDPOption], str]] だが UnionにOmegaConfが対応してないので上書きする
    sharded_ddp: Optional[str] = field(
        default="",
        metadata={
            "help": (
                "Whether or not to use sharded DDP training (in distributed training only). The base option should be"
                " `simple`, `zero_dp_2` or `zero_dp_3` and you can add CPU-offload to `zero_dp_2` or `zero_dp_3` like"
                " this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or `zero_dp_3`"
                " with the same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`."
            ),
        },
    )
    # NOTE: 元の型は Optional[Union[List[FSDPOption], str]] だが UnionにOmegaConfが対応してないので上書きする
    fsdp: Optional[str] = field(
        default="",
        metadata={
            "help": (
                "Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in distributed training"
                " only). The base option should be `full_shard`, `shard_grad_op` or `no_shard` and you can add"
                " CPU-offload to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op"
                " offload`. You can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard"
                " auto_wrap` or `shard_grad_op auto_wrap`."
            ),
        },
    )

    def __post_init__(self):
        """Adapt hydra because some parameters (e.g. debug) is changed in those type
        after init train args by __post_init__ of parent class
        Please pass this "raw" TrainingArguments into original TrainingArguments"""
        pass


def get_orig_transoformers_train_args_from_hydra_addapted_train_args(
    train_args: HydraAddaptedTrainingArguments,
):
    if isinstance(train_args, HydraAddaptedTrainingArguments):
        train_args = asdict(train_args)

    train_args_dict = {k: v for k, v in train_args.items() if k != "_n_gpu"}
    train_args = OrigTrainingArguments(**train_args_dict)
    return train_args
