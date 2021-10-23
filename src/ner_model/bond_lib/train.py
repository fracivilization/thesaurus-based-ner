import argparse
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
from datasets import DatasetDict
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
)
from .model_utils import (
    multi_source_label_refine,
    soft_frequency,
    mt_update,
    get_mt_loss,
    opt_grad,
)
from torch.utils.tensorboard import SummaryWriter
from .data_utils import remove_unused_columns
import logging
from .eval import evaluate

logger = logging.getLogger(__name__)


def initialize(args, model_class, config, t_total, epoch, model):

    # model = model_class.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    # )

    # model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if epoch == 0:
        if os.path.isfile(
            os.path.join(args.model_name_or_path, "optimizer.pt")
        ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
            )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    model.zero_grad()
    return model, optimizer, scheduler


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(
    args,
    ner_dataset,
    model_class,
    config,
    tokenizer,
    labels,
    pad_token_label_id,
    output_dir,
    model,
):
    """Train the model"""
    train_dataset = ner_dataset["train"]
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(output_dir, "tfboard"))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    collate_fn = DataCollatorForTokenClassification(tokenizer)
    train_dataloader_num = len(train_dataset) // args.train_batch_size + 1
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps // (train_dataloader_num // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            train_dataloader_num
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    model, optimizer, scheduler = initialize(
        args, model_class, config, t_total, 0, model
    )
    remove_unused_columns(train_dataset, model)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (
            len(train_dataloader) // args.gradient_accumulation_steps
        )
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps
        )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch,
        )

    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproducibility
    best_dev, best_test = [0, 0, 0], [0, 0, 0]
    if args.mt:
        teacher_model = model
    self_training_teacher_model = model

    for epoch in train_iterator:

        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = {k: t.to(args.device) for k, t in batch.items()}

            # Update labels periodically after certain begin step
            if global_step >= args.self_training_begin_step:

                # Update a new teacher periodically
                delta = global_step - args.self_training_begin_step
                if delta % args.self_training_period == 0:
                    # logger.info("update a new teacher periodically")
                    self_training_teacher_model = copy.deepcopy(model)
                    self_training_teacher_model.eval()

                    # Re-initialize the student model once a new teacher is obtained
                    if args.self_training_reinit:
                        model, optimizer, scheduler = initialize(
                            args, model_class, config, t_total, epoch, model.module
                        )

                # Using current teacher to update the label
                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                }
                with torch.no_grad():
                    outputs = self_training_teacher_model(**inputs)

                label_mask = None
                hp_labels = None  # 使用用途が不明な引数だが、初期設定のハイパラでは考慮されていないのでとりあえず無視する
                combined_labels = None  # 上に同じ
                if args.self_training_label_mode == "hard":
                    pred_labels = torch.argmax(outputs.logits, axis=2)
                    pred_labels, label_mask = multi_source_label_refine(
                        args,
                        hp_labels,
                        combined_labels,
                        pred_labels,
                        pad_token_label_id,
                        pred_logits=outputs.logits,
                    )
                elif args.self_training_label_mode == "soft":
                    pred_labels = soft_frequency(logits=outputs.logits, power=2)
                    pred_labels, label_mask = multi_source_label_refine(
                        args,
                        hp_labels,
                        combined_labels,
                        pred_labels,
                        pad_token_label_id,
                    )

                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": pred_labels,
                    "label_mask": label_mask,
                    # 一旦入力に入らないので除外
                }
            else:
                inputs = batch
                # del inputs["labeled_flags"]

            outputs = model(output_hidden_states=True, **inputs)
            loss, logits, final_embeds = (
                outputs.loss,
                outputs.logits,
                outputs.hidden_states[-1],
            )  # model outputs are always tuple in pytorch-transformers (see doc)
            mt_loss, vat_loss = 0, 0

            # Mean teacher training scheme
            if args.mt and global_step % args.mt_updatefreq == 0:
                update_step = global_step // args.mt_updatefreq
                if update_step == 1:
                    teacher_model = copy.deepcopy(model)
                    teacher_model.train(True)
                elif update_step < args.mt_rampup:
                    alpha = args.mt_alpha1
                else:
                    alpha = args.mt_alpha2
                mt_update(
                    teacher_model.named_parameters(),
                    model.named_parameters(),
                    args.mt_avg,
                    alpha,
                    update_step,
                )

            if args.mt and update_step > 0:
                with torch.no_grad():
                    teacher_outputs = teacher_model(**inputs)
                    teacher_logits, teacher_final_embeds = (
                        teacher_outputs[1],
                        teacher_outputs[2],
                    )

                _lambda = args.mt_lambda
                if args.mt_class != "smart":
                    _lambda = args.mt_lambda * min(
                        1, math.exp(-5 * (1 - update_step / args.mt_rampup) ** 2)
                    )

                if args.mt_loss_type == "embeds":
                    mt_loss = get_mt_loss(
                        final_embeds,
                        teacher_final_embeds.detach(),
                        args.mt_class,
                        _lambda,
                    )
                else:
                    mt_loss = get_mt_loss(
                        logits, teacher_logits.detach(), args.mt_class, _lambda
                    )

            # Virtual adversarial training
            if args.vat:

                if args.model_type in ["roberta", "camembert", "xlmroberta"]:
                    word_embed = model.roberta.get_input_embeddings()
                elif args.model_type == "bert":
                    word_embed = model.bert.get_input_embeddings()
                elif args.model_type == "distilbert":
                    word_embed = model.distilbert.get_input_embeddings()

                if not word_embed:
                    logger.info(
                        "Model type not supported. Unable to retrieve word embeddings."
                    )
                else:
                    embeds = word_embed(batch[0])
                    vat_embeds = (
                        embeds.data.detach()
                        + embeds.data.new(embeds.size()).normal_(0, 1) * 1e-5
                    ).detach()
                    vat_embeds.requires_grad_()

                    vat_inputs = {
                        "inputs_embeds": vat_embeds,
                        "attention_mask": batch[1],
                        "labels": batch[3],
                    }
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in ["bert", "xlnet"] else None
                        )  # XLM and RoBERTa don"t use segment_ids

                    vat_outputs = model(**vat_inputs)
                    vat_logits, vat_final_embeds = vat_outputs[1], vat_outputs[2]

                    if args.vat_loss_type == "embeds":
                        vat_loss = get_mt_loss(
                            vat_final_embeds, final_embeds.detach(), args.mt_class, 1
                        )
                    else:
                        vat_loss = get_mt_loss(
                            vat_logits, logits.detach(), args.mt_class, 1
                        )

                    vat_embeds.grad = opt_grad(vat_loss, vat_embeds, optimizer)[0]
                    norm = vat_embeds.grad.norm()

                    if torch.isnan(norm) or torch.isinf(norm):
                        logger.info("Hit nan gradient in embed vat")
                    else:
                        adv_direct = vat_embeds.grad / (
                            vat_embeds.grad.abs().max(-1, keepdim=True)[0] + 1e-4
                        )
                        vat_embeds = vat_embeds + args.vat_eps * adv_direct
                        vat_embeds = vat_embeds.detach()

                        vat_inputs = {
                            "inputs_embeds": vat_embeds,
                            "attention_mask": batch[1],
                            "labels": batch[3],
                        }
                        if args.model_type != "distilbert":
                            inputs["token_type_ids"] = (
                                batch[2]
                                if args.model_type in ["bert", "xlnet"]
                                else None
                            )  # XLM and RoBERTa don"t use segment_ids

                        vat_outputs = model(**vat_inputs)
                        vat_logits, vat_final_embeds = vat_outputs[1], vat_outputs[2]
                        if args.vat_loss_type == "embeds":
                            vat_loss = get_mt_loss(
                                vat_final_embeds,
                                final_embeds.detach(),
                                args.mt_class,
                                args.vat_lambda,
                            ) + get_mt_loss(
                                final_embeds,
                                vat_final_embeds.detach(),
                                args.mt_class,
                                args.vat_lambda,
                            )
                        else:
                            vat_loss = get_mt_loss(
                                vat_logits,
                                logits.detach(),
                                args.mt_class,
                                args.vat_lambda,
                            ) + get_mt_loss(
                                logits,
                                vat_logits.detach(),
                                args.mt_class,
                                args.vat_lambda,
                            )

            loss = loss + args.mt_beta * mt_loss + args.vat_beta * vat_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if args.evaluate_during_training:

                        logger.info(
                            "***** Entropy loss: %.4f, mean teacher loss : %.4f; vat loss: %.4f *****",
                            loss - args.mt_beta * mt_loss - args.vat_beta * vat_loss,
                            args.mt_beta * mt_loss,
                            args.vat_beta * vat_loss,
                        )

                        results, _, best_dev, _ = evaluate(
                            args,
                            model,
                            tokenizer,
                            labels,
                            pad_token_label_id,
                            best_dev,
                            eval_dataset=ner_dataset["validation"],
                            mode="dev",
                            prefix="dev [Step {}/{} | Epoch {}/{}]".format(
                                global_step, t_total, epoch, args.num_train_epochs
                            ),
                            verbose=False,
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )

                        # results, _, best_test, is_updated = evaluate(
                        #     args,
                        #     model,
                        #     tokenizer,
                        #     labels,
                        #     pad_token_label_id,
                        #     best_test,
                        #     eval_dataset=ner_dataset["test"],
                        #     mode="test",
                        #     prefix="test [Step {}/{} | Epoch {}/{}]".format(
                        #         global_step, t_total, epoch, args.num_train_epochs
                        #     ),
                        #     verbose=False,
                        # )
                        # for key, value in results.items():
                        #     tb_writer.add_scalar(
                        #         "test_{}".format(key), value, global_step
                        #     )

                        output_dirs = []
                        # if args.local_rank in [-1, 0] and is_updated:
                        #     updated_self_training_teacher = True
                        #     output_dirs.append(
                        #         os.path.join(output_dir, "checkpoint-best")
                        #     )

                        if (
                            args.local_rank in [-1, 0]
                            and args.save_steps > 0
                            and global_step % args.save_steps == 0
                        ):
                            output_dirs.append(
                                os.path.join(
                                    output_dir, "checkpoint-{}".format(global_step)
                                )
                            )

                        if len(output_dirs) > 0:
                            for output_dir in output_dirs:
                                logger.info("Saving model checkpoint to %s", output_dir)
                                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                                # They can then be reloaded using `from_pretrained()`
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_pretrained(output_dir)
                                torch.save(
                                    args, os.path.join(output_dir, "training_args.bin")
                                )
                                torch.save(
                                    model.state_dict(),
                                    os.path.join(output_dir, "model.pt"),
                                )
                                torch.save(
                                    optimizer.state_dict(),
                                    os.path.join(output_dir, "optimizer.pt"),
                                )
                                torch.save(
                                    scheduler.state_dict(),
                                    os.path.join(output_dir, "scheduler.pt"),
                                )
                                logger.info(
                                    "Saving optimizer and scheduler states to %s",
                                    output_dir,
                                )

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return model, global_step, tr_loss / global_step, best_dev, best_test
