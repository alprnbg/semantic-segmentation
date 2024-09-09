import os
import shutil
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
import optuna
import yaml
import wandb
from tqdm import tqdm

import archs
import losses
from utils import *
from metrics import iou_score
from dataset import get_dataloaders


def train(config, train_loader, model, criterion, optimizer, grad_scaler):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    model.train()
    if config["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        if config["channels_last"]: # todo not tested
            input = input.to(device="cuda", memory_format=torch.channels_last)
        else:
            input = input.cuda()
        target = target.cuda()
        with torch.cuda.amp.autocast():
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        avg_meters['loss'].update(loss.item(), input.size(0))
        if not np.isnan(iou): # target and output might be all zero, ignore in this case
            avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion, partition):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    model.eval()
    with torch.inference_mode():
        pbar = tqdm(total=len(val_loader))
        for batch_id, (input, target, _) in enumerate(val_loader):
            if not partition:
                avg_meters = normal_val_inference(config, model, input, target, criterion, 
                                                  batch_id, avg_meters)
            else:
                avg_meters = partition_val_inference(config, model, input, target, criterion, 
                                                     batch_id, avg_meters)
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def normal_val_inference(config, model, input, target, criterion, batch_id, avg_meters):
    input = input.cuda()
    target = target.cuda()
    output = model(input)
    loss = criterion(output, target)
    iou = iou_score(output, target)
    if config["draw_val"]:
        for cls_id in range(config["num_classes"]):
            os.makedirs(os.path.join("models", config["name"], "last-epoch-validation", \
                                        f"{cls_id}"), exist_ok=True)
            input_image_batch = (inv_normalize(image=input.permute(0,2,3,1).\
                                        detach().cpu().numpy())["image"]).astype("float32")
            out_img_batch = ((output[:,cls_id,:,:]>0).detach().cpu().numpy()).\
                                                                        astype("float32")
            target_img_batch = (target[:,cls_id,:,:].detach().cpu().numpy()).\
                                                                        astype("float32")
            for i in range(len(input_image_batch)):
                input_img = cv2.cvtColor(input_image_batch[i], cv2.COLOR_RGB2BGR)
                out_img = cv2.cvtColor(out_img_batch[i], cv2.COLOR_GRAY2BGR)
                target_img = cv2.cvtColor(target_img_batch[i], cv2.COLOR_GRAY2BGR)
                write_img = np.vstack([input_img, target_img, out_img])
                cv2.imwrite(os.path.join("models", config["name"], "last-epoch-validation",\
                                        f"{cls_id}", f"{batch_id}-{i}.png"), write_img*255)
    avg_meters['loss'].update(loss.item(), input.size(0))
    if not np.isnan(iou): # target and output might be all zero, ignore in this case
        avg_meters['iou'].update(iou, input.size(0))
    return avg_meters


def partition_val_inference(config, model, input, target, criterion, batch_id, avg_meters):
    org_h = input.shape[2]
    org_w = input.shape[3]
    assert org_h % config["input_size_h"] == 0 and org_w % config["input_size_w"] == 0
    input = input.cuda()
    target = target.cuda()
    ps_h = config["input_size_h"]
    ps_w = config["input_size_w"]
    output = torch.zeros_like(target)
    losses = []
    for i in range(org_h//ps_h):
        for j in range(org_w//ps_w):
            partition_input = input[:, :, i*ps_h:(i+1)*ps_h, j*ps_w:(j+1)*ps_w]
            partition_target = target[:, :, i*ps_h:(i+1)*ps_h, j*ps_w:(j+1)*ps_w]
            partition_output = model(partition_input)
            output[:, :, i*ps_h:(i+1)*ps_h, j*ps_w:(j+1)*ps_w] = partition_output
            losses.append(criterion(partition_output, partition_target).item())
    loss = np.mean(losses)
    iou = iou_score(output, target)
    if config["draw_val"]:
        for cls_id in range(config["num_classes"]):
            os.makedirs(os.path.join("models", config["name"], "last-epoch-validation", \
                                        f"{cls_id}"), exist_ok=True)
            input_image_batch = (inv_normalize(image=input.permute(0,2,3,1).\
                                        detach().cpu().numpy())["image"]).astype("float32")
            out_img_batch = ((output[:,cls_id,:,:]>0).detach().cpu().numpy()).\
                                                                        astype("float32")
            target_img_batch = (target[:,cls_id,:,:].detach().cpu().numpy()).\
                                                                        astype("float32")
            for i in range(len(input_image_batch)):
                input_img = cv2.cvtColor(input_image_batch[i], cv2.COLOR_RGB2BGR)
                out_img = cv2.cvtColor(out_img_batch[i], cv2.COLOR_GRAY2BGR)
                target_img = cv2.cvtColor(target_img_batch[i], cv2.COLOR_GRAY2BGR)
                write_img = np.vstack([input_img, target_img, out_img])
                cv2.imwrite(os.path.join("models", config["name"], "last-epoch-validation",\
                                        f"{cls_id}", f"{batch_id}-{i}.png"), write_img*255)
    avg_meters['loss'].update(loss.item(), input.size(0))
    if not np.isnan(iou): # target and output might be all zero, ignore in this case
        avg_meters['iou'].update(iou, input.size(0))
    return avg_meters


def run_experiment(config, trial=None):
    os.makedirs(os.path.join('models', config['name'], 'code'), exist_ok=True)
    is_testing_enabled = config["test_dataset_path"] is not None
    if is_testing_enabled:
        os.makedirs(os.path.join('models', config['name'], 'epoch_checkpoints'), exist_ok=True)
    record_code_files(os.path.join('models', config['name'], 'code'))

    print("##################### Config #####################")
    for key in config:
        print('%s: %s' % (key, config[key]))
    with open(os.path.join('models', config['name'], 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    print("--------------------------------------------------")

    print("##################### Model ######################")
    print("Arch: %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'])
    if config["weight"] is not None:
        model.load_state_dict(torch.load(config["weight"]))
        print("Checkpoint:", config["weight"], "is loaded")
    model = model.cuda()
    total_params = sum(p.numel() for p in model.parameters())
    params = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of Parameters:", total_params)
    print("--------------------------------------------------")

    criterion = losses.__dict__[config['loss']]().cuda()

    if config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Radam':
        optimizer = optim.RAdam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'],
                                                   patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e)
                                for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    elif config['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(config["epochs"] / 3),
                                                             eta_min=config["min_lr"])
    else:
        raise NotImplementedError

    train_loader, val_loader, test_loader = get_dataloaders(config, is_testing_enabled)

    log = OrderedDict([
        ('epoch', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('test_loss', []),
        ('test_iou', []),
        ('test_loss', []),
        ('test_iou', [])
    ])

    best_iou = 0
    trigger = 0
    grad_scaler = torch.cuda.amp.GradScaler()

    wandb.watch(model)

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        train_log = train(config, train_loader, model,
                          criterion, optimizer, grad_scaler)
        val_log = validate(config, val_loader, model, criterion, partition=config["val_partition"])

        # Test every n epochs and always at last epoch of training.
        is_on_test_epoch = epoch % config["test_n_epochs"] == 0 or \
                           epoch == (config["epochs"] - 1)
        is_on_test_epoch = is_on_test_epoch and is_testing_enabled

        if is_on_test_epoch:
            test_log = validate(config, test_loader, model, criterion, partition=config["val_partition"])
            print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - test_loss %.4f - test_iou %.4f'
                  % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], test_log['loss'],
                     test_log['iou']))
        else:
            print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
                  % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
            # If no test run was performed in this run, just set metrics to 0 to avoid confusion
            test_log = {"iou": 0, "loss": 0}

        if scheduler is not None:
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_log['loss'])
            else:
                scheduler.step()

        log['epoch'].append(epoch)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['test_loss'].append(test_log['loss'])
        log['test_iou'].append(test_log['iou'])

        pd.DataFrame(log).to_csv(os.path.join('models', config['name'], 'log.csv'), index=False)
        trigger += 1
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), os.path.join('models', config['name'], 'model.pth'))
            best_iou = val_log['iou']
            if config["draw_val"]:
                shutil.copytree(os.path.join('models', config['name'], "last-epoch-validation"),
                                os.path.join('models', config['name'], "best-validation"), 
                                dirs_exist_ok=True)
            print("=> saved best model")
            trigger = 0

        wandb.log({"loss": train_log['loss']})
        wandb.log({"iou": train_log['iou']})
        wandb.log({"val_loss": val_log['loss']})
        wandb.log({"val_iou": val_log['iou']})
        wandb.log({"test_loss": test_log['loss']})
        wandb.log({"test_iou": test_log['iou']})
        wandb.log({"best_val_iou": best_iou})

        if is_on_test_epoch:
            torch.save(model.state_dict(),
                       os.path.join('models', config['name'], "epoch_checkpoints",
                                    f"model_epoch_{epoch}.pth"))

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        # optuna prunning
        if trial is not None:
            trial.report(val_log['iou'], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        torch.cuda.empty_cache()
    return best_iou


def objective(trial):
    global optuna_config, config, config_name
    config["name"] = config_name + "_trial-"+str(trial._trial_id)
    config = suggest_config(config, optuna_config, trial)
    run = wandb.init(project=config["wandb_project_name"], name=config["name"],
                     entity=config["wandb_username"], config=config, reinit=True)
    with run:
        exp_res = run_experiment(config, trial)
    return exp_res


if __name__ == '__main__':
    seed_everything(42)
    cudnn.benchmark = True
    optuna_config = None
    config = vars(parse_args())
    if config['name'] is None:
        config_name = '%s_%s' % (config['dataset'], config['arch'])
    else:
        config_name = config["name"]
    if config["optuna_search_config"] is not None:
        print("###### OPTUNA SEARCH STARTING ######")
        assert config["test_dataset_path"] is None and config["early_stopping"] < 0
        with open(config["optuna_search_config"], 'r') as f:
            try:
                optuna_config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
                assert False
        study = optuna.create_study(direction="maximize",
                                    pruner=optuna.pruners.HyperbandPruner(),
                                    study_name=None)
        study.optimize(objective, n_trials=config["search_num_trials"],
                       timeout=config["search_timeout"])
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        config["name"] = config_name
        run = wandb.init(project=config["wandb_project_name"], name=config["name"],
                         entity=config["wandb_username"], config=config)
        with run:
            run_experiment(config)
