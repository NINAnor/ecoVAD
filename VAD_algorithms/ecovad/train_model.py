#!/usr/bin/env python3

import torch
import torch.nn as nn
import argparse
import datetime
import os
import random
import yaml

from yaml import FullLoader

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from VAD_algorithms.ecovad.VGG11 import VGG11
from VAD_algorithms.ecovad._utils.audiodataset import AudioDataset
from VAD_algorithms.ecovad._utils.earlystopping import EarlyStopping

class trainingApp():
    
    def __init__(self, data_path, save_path, save_es, batch_size, num_epoch, tb_prefix, comment,
                 lr, momentum, decay, num_workers, use_gpu=True):

        # script arguments
        self.data_path = data_path
        self.save_path = save_path
        self.save_es = save_es
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.tb_prefix = tb_prefix
        self.comment = comment
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.num_workers = num_workers
        
        # related to system
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        
        # related to the hardware
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print("Model training on {}".format(self.device))
        
        # related to the neural network
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
        self.criterion = nn.BCELoss(reduction='none')
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)
        
        # tensorboard
        self.trn_writer = None
        self.val_writer = None
        
        # early stopping
        self.patience = 10
        
        # related to metrics
        self.THRESHOLD = 0.5
        self.METRICS_SIZE = 3
        self.METRICS_LABELS_NDX = 0
        self.METRICS_PREDS_NDX = 1
        self.METRICS_LOSS_NDX = 2
        self.TRAINING_PROPORTION = 0.8
        
    def initModel(self):
        """Initialize the model, if GPU available computation done there"""       
        model = VGG11()
        model = model.float()
        
        if self.use_gpu:
            model = model.to(self.device)

        return model
    
    def initOptimizer(self):
        
        return SGD(self.model.parameters(), 
                   lr=self.lr,
                   momentum=self.momentum,
                   weight_decay=self.decay)
    
    def initDLs(self):
        # In this version of the script, we split the training / val
        # dataset here:

        # First we transform the raw files into tensors
        dataset = AudioDataset(self.data_path,
                                   n_fft=1024, 
                                   hop_length=376, 
                                   n_mels=128)

        # We set the proportion of training data
        train_number = int(len(dataset) * self.TRAINING_PROPORTION)

        # We subset the indexes of each images and separate into a train / val
        train_index = random.sample(range(0, len(dataset)), train_number)
        val_index = [item for item in range(0, len(dataset)) if item not in train_index]

        # Subset the images
        trainset = torch.utils.data.Subset(dataset, train_index)
        valset = torch.utils.data.Subset(dataset, val_index)

        # Set the batch size
        batch_size = self.batch_size

        # Make a trainLoader and a valLoader      
        trainLoader = DataLoader(trainset,
                                batch_size = batch_size, 
                                shuffle=True, 
                                num_workers=self.num_workers,
                                pin_memory=False)

        valLoader = DataLoader(valset,
                                batch_size = batch_size, 
                                shuffle=True, 
                                num_workers=self.num_workers,
                                pin_memory=False)

        
        return (trainLoader, valLoader)
    
    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.comment)
        
    def main(self):
        
        train_dl, val_dl = self.initDLs()
        
        # early stopping call
        early_stopping = EarlyStopping(patience=self.patience, path=self.save_es, verbose=True)

        # Repeat for each epoch
        for epoch_ndx in range(self.num_epoch):
            
            trn_metrics = self.do_training(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, trn_metrics, 'trn')
            
            val_metrics = self.do_val(epoch_ndx, val_dl)
            self.log_metrics(epoch_ndx, val_metrics, 'val')
            
            self.scheduler.step()

            # Add the mean loss of the val for the epoch
            early_stopping(val_metrics[self.METRICS_LOSS_NDX].mean(), self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('Finished Training')
        torch.save(self.model.state_dict(), self.save_path)
        
    def do_training(self, epoch_ndx, train_dl):
        
        trn_metrics = torch.zeros(self.METRICS_SIZE, len(train_dl.dataset), device=self.device)
        
        # Repeat for each batch in the training set
        for batch_ndx, batch_tup in enumerate(train_dl):

            self.optimizer.zero_grad()

            loss = self.ComputeBatchLoss(batch_ndx, 
                                        batch_tup, 
                                        trn_metrics)
            loss.backward()
            self.optimizer.step()
            
            return trn_metrics.to('cpu')
            
    def do_val(self, epoch_ndx, val_dl):
        
        val_metrics = torch.zeros(self.METRICS_SIZE, len(val_dl.dataset), device=self.device)
        
        with torch.no_grad():
            for batch_ndx, batch_tup in enumerate(val_dl):
                
                self.ComputeBatchLoss(batch_ndx, 
                                        batch_tup, 
                                        val_metrics)
                
                return val_metrics.to('cpu')
            
    def ComputeBatchLoss(self, batch_ndx, batch_tup, metrics_mat):
        
        inputs, labels = batch_tup[0].to(self.device), batch_tup[1].to(self.device)
        outputs = self.model(inputs)
        outputs = outputs.squeeze(1)
        loss = self.criterion(outputs, labels.float())
        
        start_ndx = batch_ndx * self.batch_size
        end_ndx = start_ndx + labels.size(0)
                
        metrics_mat[self.METRICS_LABELS_NDX, start_ndx:end_ndx] = labels.detach()
        metrics_mat[self.METRICS_PREDS_NDX, start_ndx:end_ndx] = outputs.detach()
        metrics_mat[self.METRICS_LOSS_NDX, start_ndx:end_ndx] = loss.detach()
        
        return loss.mean()
                       
    def log_metrics(self, epoch_ndx, metrics, mode_str):
        
        self.initTensorboardWriters()
        
        no_speech_preds = metrics[self.METRICS_PREDS_NDX] < self.THRESHOLD 
        no_speech_labels = metrics[self.METRICS_LABELS_NDX] < self.THRESHOLD 
        
        speech_preds = metrics[self.METRICS_PREDS_NDX] >= self.THRESHOLD 
        speech_labels = metrics[self.METRICS_LABELS_NDX] >= self.THRESHOLD
        
        no_speech_count = no_speech_labels.sum()
        speech_count = speech_labels.sum()
        
        no_speech_correct = (no_speech_preds & no_speech_labels).sum()
        speech_correct = (speech_preds & speech_labels).sum()
                
        avg_loss = metrics[self.METRICS_LOSS_NDX].mean()
        acc_no_speech = no_speech_correct / no_speech_count
        acc_speech = speech_correct / speech_count
        acc_all = (speech_correct + no_speech_correct) / (speech_count + no_speech_count)
        
        writer = getattr(self, mode_str + '_writer')
        
        writer.add_scalar("loss", avg_loss, epoch_ndx)
        writer.add_scalar("acc/speech", acc_speech, epoch_ndx)
        writer.add_scalar("acc/no_speech", acc_no_speech, epoch_ndx)
        writer.add_scalar("acc/all", acc_all, epoch_ndx)
        
        if mode_str == 'trn':        
            print(f'[TRAIN] Epoch: {epoch_ndx}, Loss: {avg_loss:.2f}, Accuracy/no speech: {acc_no_speech:.2f}, Accuracy/speech: {acc_speech:.2f}')
        
        else:  
            print(f'[VAL] Epoch: {epoch_ndx}, Loss: {avg_loss:.2f}, Accuracy/no speech: {acc_no_speech:.2f}, Accuracy/speech: {acc_speech:.2f}')

                     
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        help='Path to the config file',
                        default="./config_training.yaml",
                        required=False,
                        type=str,
                        )

    cli_args = parser.parse_args()

    # Open the config file
    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    trainingApp(cfg["TRAIN_VAL_PATH"],
            cfg["MODEL_SAVE_PATH"],
            cfg["CKPT_SAVE_PATH"],
            cfg["BATCH_SIZE"],
            cfg["NUM_EPOCH"],
            cfg["TB_PREFIX"],
            cfg["TB_COMMENTS"],
            cfg["LR"],
            cfg["MOMENTUM"],
            cfg["DECAY"],
            cfg["NUM_WORKERS"],
            cfg["USE_GPU"]
            ).main()
