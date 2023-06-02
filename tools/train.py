import sys
import os
import math
import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from bitsandbytes.optim import AdamW
from transformers import get_constant_schedule_with_warmup
from chinese_qlora.models.qlora_model import get_qlora_model
from chinese_qlora.datasets.dialog_dataset import DialogDataModule

logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    trainable_params /= 2
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param}")

def main():
    config_file = sys.argv[1]
    config = None
    with open(config_file) as fin:
        config = yaml.safe_load(fin)
    set_seed(config['train']['seed'])
    accelerator_log_kwargs = {
        'log_with': config['train']['report_id'],
        'project_dir': config['train']['output_dir']
    }
    accelerator = Accelerator(
        **accelerator_log_kwargs
    )
    logger.info(accelerator.state, main_process_only=False)
    # model
    model = get_qlora_model(**config['model'])
    print_trainable_parameters(model)
    # dataloader
    data_module = DialogDataModule(**config['datamodule'])
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
    # optimizer
    optimizer = AdamW(model.parameters(),
                      lr=float(config['train']['lr']),
                      is_paged=True,
                      optim_bits=32)
    scheduler = get_constant_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=config['train']['warmup_steps'] * config['train']['gradient_accumulation_steps'])
    optimizer, model, train_dataloader, val_dataloader, test_dataloader, scheduler = \
        accelerator.prepare(
            optimizer,
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            scheduler
        )
    gradient_accumulation_steps = config['train']['gradient_accumulation_steps']
    max_train_steps = len(train_dataloader) * config['train']['num_epochs'] // gradient_accumulation_steps
    checkpointing_steps = config['train']['save_steps']
    experiment_config = {
        'lr_scheduler_type': 'constant_with_warmup'
    }
    experiment_config.update(config['train'])
    accelerator.init_trackers('accelerate_trainer', experiment_config)
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    for epoch in range(0, config['train']['num_epochs']):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            if (step+1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                logger.info(f'steps: {completed_steps}, train_loss: {loss}')
                accelerator.log({
                    'train_loss_step': loss.item()
                }, step=completed_steps)
                if completed_steps > 0 and completed_steps % checkpointing_steps == 0:
                    if accelerator.is_local_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            os.path.join(config['train']['output_dir'], f"checkpoint-{completed_steps}"),
                        )
                    model.eval()
                    losses = []
                    for step, batch in enumerate(val_dataloader):
                        with torch.no_grad():
                            outputs = model(**batch)
                        loss = outputs.loss
                        losses.append(accelerator.gather_for_metrics(loss.repeat(config['datamodule']['batch_size'])))
                    losses = torch.cat(losses)
                    try:
                        eval_loss = torch.mean(losses)
                        perplexity = math.exp(eval_loss)
                    except OverflowError:
                        perplexity = float('inf')
                    logger.info(f'steps: {completed_steps}, perplexity: {perplexity}, eval_loss: {eval_loss}')
                    accelerator.log({
                        "perplexity": perplexity,
                        "eval_loss": eval_loss.item(),
                        "train_loss": total_loss.item() / checkpointing_steps / gradient_accumulation_steps,
                    }, step=completed_steps)
                    total_loss = 0.0
                    model.train()
    accelerator.end_training()


if __name__ == '__main__':
    main()
