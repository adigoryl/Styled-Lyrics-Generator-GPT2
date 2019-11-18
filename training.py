import logging
import argparse
import torch
import os

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, WarmupLinearSchedule
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

import utils.utilities as U
# import utils.bleu as bleu

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    # Model hyperparams
    parser.add_argument("--model_size", type=str, default="gpt2", help="Model size (gpt2, gpt2-medium).")
    parser.add_argument("--store_in_folder", type=str, default="tuned_models", help="Master folder to store the model.")

    # Training hyperparams
    parser.add_argument("--train_model", action='store_true', help="Fine-tune gpt2 model.")
    parser.add_argument("--train_data_path", type=str, default="", help="Train dataset path.")
    parser.add_argument('--num_train_epochs', type=int, default=5, help="")
    parser.add_argument('--save_every_n_epoch', type=int, default=5, help="")
    parser.add_argument('--train_batch_size', type=int, default=1, help="")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="This is equivalent to batch size, if the GPU has limited memory can be used instead.")

    # Optimizer hyperparams
    parser.add_argument('--learning_rate', type=float, default=0.0000625, help="")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help="Epsilon for Adam optimizer.")
    parser.add_argument('--warmup_steps', type=int, default=0, help="")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")

    args = parser.parse_args()
    return args


def prepare_train_data(args, enc, device):
    raw_dataset = U.load_dataset(args.train_data_path)
    formated_dataset = U.format_n_tokenize_data(raw_dataset, enc)
    train_tensor_data = U.construct_input(formated_dataset, device, max_input_len=enc.max_len)

    # Load onto the Pytorch DataLoader
    # Note: the '*' extracts all elements from the list
    train_data = TensorDataset(*train_tensor_data)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    return train_data_loader


def main():
    args = init_args()
    device, n_gpu = U.get_device(logger)

    output_dir = U.create_save_path(args, __file__)

    run_details_file = os.path.join(output_dir, "run_details.txt")
    # tb_dir = os.path.join(output_dir, "all_scalars.json")
    tb_writer = SummaryWriter(output_dir)

    special_tokens_dict = {
        "additional_special_tokens": [
            '[s:genre]', '[s:artist]', '[s:year]', '[s:album]', '[s:song_name]', '[s:lyrics]',
            '[e:genre]', '[e:artist]', '[e:year]', '[e:album]', '[e:song_name]', '[e:lyrics]'
        ]
    }

    U.log_arguments(run_details_file, args, special_tokens_dict["additional_special_tokens"])

    # Initialise model & tokenizer
    enc = GPT2Tokenizer.from_pretrained(args.model_size)
    enc.add_special_tokens(special_tokens_dict)

    model = GPT2LMHeadModel.from_pretrained(args.model_size)
    model.resize_token_embeddings(len(enc))

    # Prepare training data
    train_data_loader = prepare_train_data(args, enc, device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimization_steps = ((len(train_data_loader) * args.num_train_epochs) // \
                         (args.train_batch_size * args.gradient_accumulation_steps)) + 1000

    # TODO: Could use NVIDIA Apex for lower precision calculations.
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=optimization_steps)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @                            FINE-TUNE GPT2
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if args.train_model:
        logger.info("\nFine-tuning GPT2")
        print("To visualise data using TensorBoardX -> type in console:\ntensorboard --logdir={}".format(output_dir))
        model.to(device)
        model.train()

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            past = None

            if epoch > 0:
                # Re-process dataset since the features dropout is random.
                train_data_loader = prepare_train_data(args, enc, device)

            for step, batch in enumerate(tqdm(train_data_loader, desc="Training")):
                tok_ids, tok_type_ids, pos_ids, att_mask, lm_labels = batch
                outputs = model(
                    input_ids=tok_ids, past=past, attention_mask=att_mask, token_type_ids=tok_type_ids,
                    position_ids=pos_ids, labels=lm_labels
                )

                loss = outputs[0]
                # predicted_scores = outputs[1]
                # past = outputs[2]

                # Log the loss to TensorBoardX
                global_step = (epoch * len(train_data_loader)) + (step + 1)
                tb_writer.add_scalar('loss', loss.item(), global_step)

                # Normalise the loss (Simulates average of a batch)
                loss = loss / args.gradient_accumulation_steps
                loss.backward(retain_graph=True)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if (epoch + 1) % args.save_every_n_epoch == 0:
                save_model_dir = U.make_dir(os.path.join(output_dir, "model_epoch_" + str(epoch + 1)))
                model.save_pretrained(save_model_dir)
                enc.save_pretrained(save_model_dir)

        tb_dir = os.path.join(output_dir, "all_scalars.json")
        tb_writer.export_scalars_to_json(tb_dir)
        tb_writer.close()

        # Save model and tokenizer to a directory
        save_model_dir = U.make_dir(os.path.join(output_dir, "model_epoch_" + str(epoch + 1)))
        model.save_pretrained(save_model_dir)
        enc.save_pretrained(save_model_dir)


if __name__ == '__main__':
    main()
