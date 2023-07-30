# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # This prevents tokenizers from taking all cpus
import sys
import time
import pickle
import shutil
import logging
import argparse
# 3rd-party Modules
import bs4
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import wandb
from wandb import AlertLevel
import ray
from ray import air, tune
from ray.air import session
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
#import nlpaug.augmenter.word as naw
from textaugment import EDA
# Pytorch Modules
import torch
torch.set_num_threads(2) # This prevents Pytorch from taking all cpus
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
# Huggingface Modules
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.classification.model import ClassificationModel
from model.classification.dataset import CustomDataset
from model.optimizer.optimizer import get_optimizer
from model.optimizer.scheduler import get_scheduler
from utils.utils import TqdmLoggingHandler, write_log, get_huggingface_model_name, get_wandb_exp_name, get_torch_device, check_path
# from .preprocessing import load_data
from .augment import load_preprocessed_data

"""
Text autoaugment for classification using ray tune
"""

Trial_Num = 0
log_df = pd.DataFrame({
    'Dataset': [],
    'Model': [],
    'Trial_Num': [],
    'ts': [],
    'Policy': [],
    'Best Epoch': [],
    'Best ACC': [],
})

def search(args):
    """
    Main function for searching augmentation policy & label smoothing eps
    """

    # pass args to global variable
    global args_
    args_ = args

    # Setup
    setup_dict = setup(args)

    """
    if args.use_wandb:
        wandb.init(project=args.proj_name,
            name=get_wandb_exp_name(args),
            config=args,
            notes=args.description,
            tags=["SEARCH",
                  f"Dataset: {args.task_dataset}",
                  f"Model: {args.model_type}"])
    """

    tuner = tune.Tuner(tune.with_resources(
                           tune.with_parameters(objective),
                           resources={"cpu":1, "gpu":0.5}
                       ),
                       tune_config=tune.TuneConfig(
                            num_samples=100,
                            max_concurrent_trials=2,
                            #metric=f'best_{args.optimize_objective}',
                            mode='max' if args.optimize_objective in ['accuracy', 'f1'] else 'min',
                            scheduler=AsyncHyperBandScheduler(),
                            search_alg=HyperOptSearch(),
                        ),
                    #    run_config=air.RunConfig(
                    #        callbacks=[WandbLoggerCallback(
                    #            project=args.proj_name,
                    #            log_config=True,
                    #            **{
                    #                 "name": f"SEARCH - {args.task_dataset.upper()} / {args.model_type.upper()} / {args.data_subsample_size.upper()}" if args.augmentation_type != 'ablation_no_labelsmoothing'
                    #                         else f"SEARCH - {args.task_dataset.upper()} / {args.model_type.upper()} / {args.data_subsample_size.upper()} / ablation_no_labelsmoothing",
                    #                 "config": args,
                    #                 "notes": args.description,
                    #                 "tags": ["SEARCH",
                    #                         f"Dataset: {args.task_dataset}",
                    #                         f"Model: {args.model_type}"],
                    #            }
                    #        )]
                    #     ),
                       param_space=policy_search_space)

    # Search
    result_grid = tuner.fit()
    best_result = result_grid.get_best_result()
    print(f"Best result: {best_result}")
    best_config = result_grid.get_best_result().config
    best_policy = {
        'aug_prob': best_config['aug_prob'],
        'aug_prob_SR': best_config['aug_prob_SR'],
        'aug_prob_RI': best_config['aug_prob_RI'],
        'aug_prob_RS': best_config['aug_prob_RS'],
        'aug_prob_RD': best_config['aug_prob_RD'],
        'aug_magn_SR': best_config['aug_magn_SR'],
        'aug_magn_RI': best_config['aug_magn_RI'],
        'aug_magn_RS': best_config['aug_magn_RS'],
        'aug_magn_RD': best_config['aug_magn_RD'],
        'aug_num': best_config['aug_num'],
        'ls_eps_ori': best_config['ls_eps_ori'],
        'ls_eps_aug': best_config['ls_eps_aug'],
    }

    print(f"Best policy: {best_policy}")

    # Save the augmented data with best policy to pickle file using ts
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    shutil.copy(os.path.join(preprocessed_path, f'train_search_ongoing_{best_config["ts"]}_{args.data_subsample_size}.pkl'),
                os.path.join(preprocessed_path, f'train_optimal_{args.data_subsample_size}.pkl'))
    print(f"Saved the augmented data with best policy to {os.path.join(preprocessed_path, f'train_optimal_{args.data_subsample_size}.pkl')}")

    checkpoint_path = os.path.join(args.checkpoint_path, args.task, args.task_dataset, args.model_type)
    if args.augmentation_type == 'ablation_no_labelsmoothing':
        checkpoint_save_name = f'checkpoint_ablation_search_ongoing_{best_config["ts"]}_{args.data_subsample_size}.pt'
    else:
        checkpoint_save_name = f'checkpoint_search_ongoing_{best_config["ts"]}_{args.data_subsample_size}.pt'

    model_save_path = os.path.join(args.model_path, args.task, args.task_dataset, args.model_type)
    check_path(model_save_path)
    if args.augmentation_type == 'ablation_no_labelsmoothing':
        model_save_name = f'final_model_ablation_nols_{args.data_subsample_size}.pt'
    else:
        model_save_name = f'final_model_softtaa_{args.data_subsample_size}.pt'
    shutil.copy(os.path.join(checkpoint_path, checkpoint_save_name),
                os.path.join(model_save_path, model_save_name))
    print(f"Saved the model with best policy to {os.path.join(model_save_path, model_save_name)}")

    # Delete the ongoing pickle file
    # Delete the file that contains "search_ongoing" in its name
    for file_name in os.listdir(preprocessed_path):
        if 'train_search_ongoing_' in file_name and args.data_subsample_size in file_name:
            os.remove(os.path.join(preprocessed_path, file_name))
    for file_name in os.listdir(checkpoint_path):
        if 'checkpoint_search_ongoing_' in file_name and args.data_subsample_size in file_name:
            os.remove(os.path.join(checkpoint_path, file_name))
    print("Deleted all ongoing data file and checkpoint file")

    # ts = augment_data(args, best_policy, searched=True)
    # augmented_train_data, num_classes = augment_data(args, best_config)
    # ts = preprocess_augmented_data(args, augmented_train_data, num_classes, best_config, searched=True)

    """
    if args.use_wandb:
        global log_df
        wandb_table = wandb.Table(dataframe=log_df)
        wandb.log({'search_log': wandb_table})
        wandb.finish()
    """


def setup(args):
    global device, logger, policy_search_config, policy_search_space
    device = get_torch_device(args.device)

    # Define logger
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    policy_search_config = {
        'aug_type': ['SR', 'RI', 'RS', 'RD'],
        'aug_type_prob': [0.0, 0.5],
        'aug_magnitude': [0.0, 0.5],
        'aug_prob': [0.0, 0.5],
        'aug_num': [1, 5],
        'label_smoothing_eps': [0.0, 0.5],
    }

    augmentation_magnitude_prob = [
        np.random.uniform(policy_search_config['aug_magnitude'][0], policy_search_config['aug_magnitude'][1]) for _ in range(len(policy_search_config['aug_type']))
    ]

    # Mapping augmentation type with each augmentation magnitude
    augmentation_magnitude_dict = {}
    for augmentation_type, augmentation_magnitude in zip(policy_search_config['aug_type'], augmentation_magnitude_prob):
        augmentation_magnitude_dict[augmentation_type] = augmentation_magnitude

    policy_search_space = {
        'aug_prob': tune.uniform(policy_search_config['aug_type_prob'][0], policy_search_config['aug_type_prob'][1]),
        'aug_prob_SR': tune.uniform(policy_search_config['aug_prob'][0], policy_search_config['aug_prob'][1]),
        'aug_prob_RI': tune.uniform(policy_search_config['aug_prob'][0], policy_search_config['aug_prob'][1]),
        'aug_prob_RS': tune.uniform(policy_search_config['aug_prob'][0], policy_search_config['aug_prob'][1]),
        'aug_prob_RD': tune.uniform(policy_search_config['aug_prob'][0], policy_search_config['aug_prob'][1]),
        'aug_magn_SR': tune.uniform(policy_search_config['aug_magnitude'][0], policy_search_config['aug_magnitude'][1]),
        'aug_magn_RI': tune.uniform(policy_search_config['aug_magnitude'][0], policy_search_config['aug_magnitude'][1]),
        'aug_magn_RS': tune.uniform(policy_search_config['aug_magnitude'][0], policy_search_config['aug_magnitude'][1]),
        'aug_magn_RD': tune.uniform(policy_search_config['aug_magnitude'][0], policy_search_config['aug_magnitude'][1]),
        'aug_num': tune.randint(policy_search_config['aug_num'][0], policy_search_config['aug_num'][1]),
        'ls_eps_ori': tune.uniform(policy_search_config['label_smoothing_eps'][0], policy_search_config['label_smoothing_eps'][1]) if args.augmentation_type != 'ablation_no_labelsmoothing' else 0.0,
        'ls_eps_aug': tune.uniform(policy_search_config['label_smoothing_eps'][0], policy_search_config['label_smoothing_eps'][1]) if args.augmentation_type != 'ablation_no_labelsmoothing' else 0.0,
        'args': args,
        'ts': None, # ts will be assigned later
    }

    setup_dict = {
        'device': device,
        'logger': logger,
        'policy_search_config': policy_search_config,
        'policy_search_space': policy_search_space,
    }

    return setup_dict

def augment_data(args, config, searched=False):
    """
    Augment data and construct augmented dataset from given policy
    """

    # Load data
    train_data = load_preprocessed_data(args)

    augmenter = EDA()

    # Construct augmented dataset
    augmented_data = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'labels': [],
        'soft_labels': [],
        'num_classes': train_data['num_classes'],
        'vocab_size': train_data['vocab_size'],
        'pad_token_id': train_data['pad_token_id'],
        'augmentation_policy': train_data['augmentation_policy'],
    }

    # Reconstruct sentence from input_ids using huggingface tokenizer
    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for idx in range(len(train_data['input_ids'])):
        decoded_sent = tokenizer.decode(train_data['input_ids'][idx], skip_special_tokens=True)

        # Augment data and append augmented data with label smoothing
        if np.random.uniform(0.0, 1.0) < config['aug_prob']: # Augment or not
            # Select augmentation type randomly
            augmentation_type_prob = [config['aug_prob_SR'], config['aug_prob_RI'], config['aug_prob_RS'], config['aug_prob_RD']]
            # Sum of augmentation_type_prob should be 1.0
            augmentation_type_prob = [prob / sum(augmentation_type_prob) for prob in augmentation_type_prob] # Normalize
            augmentation_type = np.random.choice(['SR', 'RI', 'RS', 'RD'], p=augmentation_type_prob)

            # Augment data using selected augmentation type
            len_words = len(decoded_sent.split())
            n_to_modify = max(1, int(config['aug_magn_' + augmentation_type] * len_words)) # Number of words to modify, minimum 1
            augmented_text = []
            if augmentation_type == 'SR':
                augmented_text = [augmenter.synonym_replacement(decoded_sent, n=n_to_modify) for _ in range(config['aug_num'])] # Augment n times
            elif augmentation_type == 'RI':
                augmented_text = [augmenter.random_insertion(decoded_sent, n=n_to_modify) for _ in range(config['aug_num'])]
            elif augmentation_type == 'RS':
                augmented_text = [augmenter.random_swap(decoded_sent, n=n_to_modify) for _ in range(config['aug_num'])]
            elif augmentation_type == 'RD':
                augmented_text = [augmenter.random_deletion(decoded_sent, p=config['aug_magn_RD']) for _ in range(config['aug_num'])]

            soft_labels = train_data['soft_labels'][idx] * (1 - config['ls_eps_aug']) + (config['ls_eps_aug'] / train_data['num_classes']) # Apply label smoothing

            # Append augmented data with label smoothing
            for each_augmented_text in augmented_text:
                # Encode augmented sentence using huggingface tokenizer
                tokenized_sent = tokenizer(each_augmented_text, padding='max_length', truncation=True,
                                           max_length=args.max_seq_len, return_tensors='pt')

                # Append augmented data
                augmented_data['input_ids'].append(tokenized_sent['input_ids'].squeeze())
                augmented_data['attention_mask'].append(tokenized_sent['attention_mask'].squeeze())
                if args.model_type in ['bert', 'albert', 'electra', 'deberta', 'debertav3']:
                    augmented_data['token_type_ids'].append(tokenized_sent['token_type_ids'].squeeze())
                else: # roberta does not use token_type_ids
                    augmented_data['token_type_ids'].append(torch.zeros(args.max_seq_len, dtype=torch.long))
                augmented_data['soft_labels'].append(soft_labels) # Apply label-smoothened soft labels
                augmented_data['labels'].append(train_data['labels'][idx]) # Keep the hard labels as they are

    # Apply label smoothing to original data
    for idx in range(len(train_data['input_ids'])):
        soft_labels = train_data['soft_labels'][idx] * (1 - config['ls_eps_ori']) + (config['ls_eps_ori'] / train_data['num_classes'])
        train_data['soft_labels'][idx] = soft_labels # Apply label-smoothened soft labels

    # Merge augmented data with original data
    total_dict = {
        'input_ids': train_data['input_ids'] + augmented_data['input_ids'],
        'attention_mask': train_data['attention_mask'] + augmented_data['attention_mask'],
        'token_type_ids': train_data['token_type_ids'] + augmented_data['token_type_ids'],
        'labels': train_data['labels'] + augmented_data['labels'],
        'soft_labels': train_data['soft_labels'] + augmented_data['soft_labels'],
        'num_classes': train_data['num_classes'],
        'vocab_size': train_data['vocab_size'],
        'pad_token_id': train_data['pad_token_id'],
        'augmentation_policy': train_data['augmentation_policy'],
    }

    # Save data as pickle file
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.localtime())
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    if searched: # Optimal policy searched through AutoAugment
        if args.augmentation_type == 'ablation_no_labelsmoothing':
            save_name = f'train_optimal_ablation_{args.data_subsample_size}.pkl'
        else:
            save_name = f'train_optimal_{args.data_subsample_size}.pkl'

        print(f"Saving augmented data with searched optimal policy to {os.path.join(preprocessed_path, save_name)}")
    else:
        save_name = f'train_search_ongoing_{ts}_{args.data_subsample_size}.pkl'
    with open(os.path.join(preprocessed_path, save_name), 'wb') as f:
        pickle.dump(total_dict, f)

    config['ts'] = ts
    return ts

def preprocess_augmented_data(args, augmented_train_data, num_classes, config, searched=False):
    # Define tokenizer & config
    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_config = AutoConfig.from_pretrained(model_name)

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    # Preprocessing - Define data_dict
    augmented_data = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'labels': [],
        'soft_labels': [],
        'num_classes': augmented_train_data['num_classes'],
        'vocab_size': augmented_train_data['vocab_size'],
        'pad_token_id': augmented_train_data['pad_token_id'],
        'augmentation_policy': augmented_train_data['augmentation_policy'], # None.
    }

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.localtime())

    for idx in range(len(augmented_train_data['text'])):
        # Get text and label
        text = augmented_train_data['text'][idx]
        label = augmented_train_data['label'][idx]
        soft_label = augmented_train_data['soft_label'][idx]

        # Remove html tags
        clean_text = bs4.BeautifulSoup(text, 'lxml').text
        # Remove special characters
        clean_text = clean_text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        # Remove multiple spaces
        clean_text = ' '.join(clean_text.split())

        # Tokenize
        tokenized = tokenizer(clean_text, padding='max_length', truncation=True,
                                max_length=args.max_seq_len, return_tensors='pt')

        # Add data to data_dict
        augmented_data['input_ids'].append(tokenized['input_ids'].squeeze())
        augmented_data['attention_mask'].append(tokenized['attention_mask'].squeeze())
        if args.model_type in ['bert', 'albert', 'electra', 'deberta', 'debertav3']:
            augmented_data['token_type_ids'].append(tokenized['token_type_ids'].squeeze())
        else: # roberta does not use token_type_ids
            augmented_data['token_type_ids'].append(torch.zeros(args.max_seq_len, dtype=torch.long))
        augmented_data['labels'].append(torch.tensor(label, dtype=torch.long)) # Cross Entropy Loss
        augmented_data['soft_labels'].append(torch.tensor(soft_label, dtype=torch.float)) # Label Smoothing Loss

    # Save data as pickle file
    if searched: # Optimal policy searched through AutoAugment
        if args.augmentation_type_list == 'ablation_no_labelsmoothing':
            save_name = f'train_optimal_ablation_{args.data_subsample_size}.pkl'
        else:
            save_name = f'train_optimal_{args.data_subsample_size}.pkl'

        print(f"Saving augmented data with searched optimal policy to {os.path.join(preprocessed_path, save_name)}")
    else:
        save_name = f'train_search_ongoing_{ts}_{args.data_subsample_size}.pkl'
    with open(os.path.join(preprocessed_path, save_name), 'wb') as f:
        pickle.dump(augmented_data, f)

    return ts

def evaluate_policy(args, policy, ts):
    """
    Train and validate the model with the dataset augmented by the policy
    """
    device = get_torch_device(args.device)
    # Load dataset and define dataloader

    dataset_dict, dataloader_dict = {}, {}
    dataset_dict['train'] = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'train_search_ongoing_{ts}_{args.data_subsample_size}.pkl'))
    dataset_dict['valid'] = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'valid_original_full.pkl'))

    dataloader_dict['train'] = DataLoader(dataset_dict['train'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=True, pin_memory=True, drop_last=True)
    dataloader_dict['valid'] = DataLoader(dataset_dict['valid'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=False, pin_memory=True, drop_last=False)
    args.vocab_size = dataset_dict['train'].vocab_size
    args.num_classes = dataset_dict['train'].num_classes
    args.pad_token_id = dataset_dict['train'].pad_token_id

    # Get model instance
    model = ClassificationModel(args).to(device)

    # Define optimizer and scheduler
    optimizer = get_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay, optim_type=args.optimizer)
    scheduler = get_scheduler(optimizer, len(dataloader_dict['train']), num_epochs=args.num_epochs,
                              early_stopping_patience=args.early_stopping_patience, learning_rate=args.learning_rate,
                              scheduler_type=args.scheduler)

    # Define loss function
    cls_loss = nn.CrossEntropyLoss()

    # Train/Valid - Start training
    best_epoch_idx = 0
    best_valid_objective_value = None
    early_stopping_counter = 0

    for epoch_idx in range(args.num_epochs):
        # Train - Set model to train mode
        model = model.train()
        train_loss_cls = 0
        train_acc_cls = 0
        train_f1_cls = 0

        # Train - Iterate one epoch
        for iter_idx, data_dicts in enumerate(dataloader_dict['train']):
            # Train - Get input data
            input_ids = data_dicts['input_ids'].to(device)
            attention_mask = data_dicts['attention_mask'].to(device)
            token_type_ids = data_dicts['token_type_ids'].to(device)
            labels = data_dicts['labels'].to(device) # For calculating accuracy
            soft_labels = data_dicts['soft_labels'].to(device) # For training

            # Train - Forward pass
            classification_logits = model(input_ids, attention_mask, token_type_ids)

            # Train - Calculate loss & accuracy/f1 score
            batch_loss_cls = cls_loss(classification_logits, soft_labels)
            batch_acc_cls = (classification_logits.argmax(dim=-1) == labels).float().mean()
            batch_f1_cls = f1_score(labels.cpu().numpy(), classification_logits.argmax(dim=-1).cpu().numpy(), average='macro')

            # Train - Backward pass
            optimizer.zero_grad()
            batch_loss_cls.backward()
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            if args.scheduler in ['StepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
                scheduler.step() # These schedulers require step() after every training iteration

            # Train - Logging
            train_loss_cls += batch_loss_cls.item()
            train_acc_cls += batch_acc_cls.item()
            train_f1_cls += batch_f1_cls

        # Valid - Set model to eval mode
        model = model.eval()
        valid_loss_cls = 0
        valid_acc_cls = 0
        valid_f1_cls = 0

        # Valid - Iterate one epoch
        for iter_idx, data_dicts in enumerate(dataloader_dict['valid']):
            # Valid - Get input data
            input_ids = data_dicts['input_ids'].to(device)
            attention_mask = data_dicts['attention_mask'].to(device)
            token_type_ids = data_dicts['token_type_ids'].to(device)
            labels = data_dicts['labels'].to(device) # For calculating accuracy
            soft_labels = data_dicts['soft_labels'].to(device)

            # Valid - Forward pass
            with torch.no_grad():
                classification_logits = model(input_ids, attention_mask, token_type_ids)

            # Valid - Calculate loss & accuracy/f1 score
            batch_loss_cls = cls_loss(classification_logits, soft_labels)
            batch_acc_cls = (classification_logits.argmax(dim=-1) == labels).float().mean()
            batch_f1_cls = f1_score(labels.cpu().numpy(), classification_logits.argmax(dim=-1).cpu().numpy(), average='macro')

            # Valid - Logging
            valid_loss_cls += batch_loss_cls.item()
            valid_acc_cls += batch_acc_cls.item()
            valid_f1_cls += batch_f1_cls

        # Valid - Call scheduler
        if args.scheduler == 'LambdaLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(valid_loss_cls)

        # Valid - Check loss & save model
        valid_loss_cls /= len(dataloader_dict['valid'])
        valid_acc_cls /= len(dataloader_dict['valid'])
        valid_f1_cls /= len(dataloader_dict['valid'])

        if args.optimize_objective == 'loss':
            valid_objective_value = valid_loss_cls
            valid_objective_value = -1 * valid_objective_value # Loss is minimized, but we want to maximize the objective value
        elif args.optimize_objective == 'accuracy':
            valid_objective_value = valid_acc_cls
        elif args.optimize_objective == 'f1':
            valid_objective_value = valid_f1_cls
        else:
            raise NotImplementedError

        if best_valid_objective_value is None or valid_objective_value > best_valid_objective_value:
            best_valid_objective_value = valid_objective_value
            best_epoch_idx = epoch_idx
            early_stopping_counter = 0 # Reset early stopping counter

            # Save model
            checkpoint_save_path = os.path.join(args.checkpoint_path, args.task, args.task_dataset, args.model_type)
            check_path(checkpoint_save_path)

            if args.augmentation_type == 'ablation_no_labelsmoothing':
                checkpoint_save_name = f'checkpoint_ablation_search_ongoing_{ts}_{args.data_subsample_size}.pt'
            else:
                checkpoint_save_name = f'checkpoint_search_ongoing_{ts}_{args.data_subsample_size}.pt'

            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None
            }, os.path.join(checkpoint_save_path, checkpoint_save_name))
        else:
            early_stopping_counter += 1

        # Valid - Early stopping
        if early_stopping_counter >= args.early_stopping_patience:
            break

    """
    # Valid - Logging
    if args.use_wandb:
        global log_df, Trial_Num
        Trial_Num += 1
        log_df = log_df.append({
            'Dataset': args.task_dataset,
            'Model': args.model_type,
            'Trial_Num': Trial_Num,
            'ts': ts,
            'Policy': policy,
            'Best Epoch': best_epoch_idx,
            'Best ACC': best_valid_objective_value,
        }, ignore_index=True)
    """

    return best_epoch_idx, best_valid_objective_value

def objective(config, checkpoint_dir=None):
    assert torch.cuda.is_available() == True

    ts = augment_data(config['args'], config)
    # augmented_train_data, soft_valid_data, soft_test_data, num_classes = augment_data(config['args'], config)
    # ts = preprocess_augmented_data(config['args'], augmented_train_data, soft_valid_data, soft_test_data, num_classes, config)
    _, best_valid_objective_value = evaluate_policy(config['args'], config, ts)

    session.report({f"best_{config['args'].optimize_objective}": best_valid_objective_value})

    return best_valid_objective_value
