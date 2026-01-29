import argparse
import os
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from .clssimp import clssimp
from .assistant import (Assistant, stats)
from . import utils as ut
from .visualize import plot_umap

class TrainEvalAutosort():
    def __init__(self, device, file_name, input_size, output_size):
        self.model = clssimp(input_size, output_size).to(device)
        self.device = device
        self.file_name = file_name
        self.loaded = False
        if os.path.exists(f"{self.file_name}.pt"):
            self.model.load_state_dict(torch.load(f"{self.file_name}.pt"))
            self.loaded = True
            print(f"Saved model \n{self.file_name}.pt loaded")
            print()
        else:
            print("New model trainning")
            print()
            
        self.loss = nn.BCEWithLogitsLoss(pos_weight=None)
        #self.loss = nn.MSELoss(reduction='sum')
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.0001
        )
        
        self.assistant = Assistant(
            net=self.model, 
            error=self.loss, 
            optimizer=self.optimizer
        )

    def train_eval_snn(self, epochs, save):  
        train_stats = stats()
        test_stats = stats()
        
        for epoch in range(1, epochs+1):
            # Train Model.
            self.model.train()
            ##########################################################################################
            train_data = ut.ExpDataset(is_train=True) 
            train_loader = DataLoader(train_data, batch_size=1024, num_workers=8, shuffle=True)
            ##########################################################################################
            for inp, lbl in tqdm(train_loader):
                inp, lbl = inp.to(self.device), lbl.to(self.device)
                output = self.assistant.train(inp, lbl,train_stats)

            # Evaluate Model.
            self.model.eval()
            ##########################################################################################
            test_data = ut.ExpDataset(is_train=False)
            test_loader = DataLoader(test_data, batch_size=1024, num_workers=8)
            ##########################################################################################

            for inp, lbl in tqdm(test_loader):
                inp, lbl = inp.to(self.device), lbl.to(self.device)
                output = self.assistant.test(inp, lbl,test_stats)

            # Print the Stats, Save the best test-accuracy model, and Update `stats`.
            print("Epoch: {0}, Train Stats: {1}, Test Stats: {2}".format(epoch, train_stats, test_stats))
            if test_stats.best_epoch:
                torch.save(self.model.state_dict(), f"{self.file_name}.pt")
                if save:
                    np.save(ut.test_data_path/'output_y.npy',test_stats.pred_label)
                    print("Result labels saved")
            print()
            train_stats.update()
            test_stats.update()
    
    def eval_snn(self, save):
        if not self.loaded: raise KeyError
        test_stats = stats()
        self.model.eval()
        ##########################################################################################
        test_data = ut.ExpDataset(is_train=False)
        test_loader = DataLoader(test_data, batch_size=1024, num_workers=8)
        ##########################################################################################
        for inp, lbl in tqdm(test_loader):
            inp, lbl = inp.to(self.device), lbl.to(self.device)
            output = self.assistant.test(inp, lbl,test_stats)
        print("Test Stats: {0}".format(test_stats))
        if save:
            np.save(ut.test_data_path/'output_y.npy',test_stats.pred_label)
            print("Result labels saved")
        print()
    
    
class AutosortTrainer():
    def __init__(self, dirs, params):
        dirs.trained_model_path.mkdir(exist_ok=True)
        self.file_name = dirs.trained_model_path/params.model_name
        self.input_size = params.input_size
        self.output_size = params.output_size
        self.train_path = dirs.train_data_path
        self.test_path = dirs.test_data_path
        self.device = params.device
        self.epochs = params.epochs
    
    def train_on_gpu(self, save=True):
        tes = TrainEvalAutosort(
            device=self.device, 
            file_name=self.file_name, 
            input_size = self.input_size, 
            output_size=self.output_size
        
        )
        ut.data_path(train_path=self.train_path, test_path=self.train_path)
        
        tes.train_eval_snn(epochs=self.epochs, save=save)
        
    def test_on_gpu(self, test_day=None, save=True):
        tes = TrainEvalAutosort(
            device=self.device, 
            file_name=self.file_name, 
            input_size = self.input_size, 
            output_size=self.output_size
        )
        
        if test_day is None:
            iter_dir = sorted(os.listdir(self.test_path))
        else:
            iter_dir = test_day
            
        if isinstance(iter_dir, str): iter_dir = [iter_dir]
        
        for day in sorted(iter_dir):
            print(f'Day: {day}    ', end='')
            ut.data_path(test_path=self.test_path/day)
            tes.eval_snn(save=save)

    def plot_umap(self, day_id, metric=0, s=1):
        test_path = self.test_path/day_id
        plot_umap(test_path=test_path, metric=metric, s=s)
        
    def get_accuracy(self, day, train=False):
        if train:
            target = np.load(self.train_path/'test_y.npy')
            output = np.load(self.train_path/'output_y.npy')
        else:
            target = np.load(self.test_path/day/'test_y.npy')
            output = np.load(self.test_path/day/'output_y.npy')
            
        result = self.cal_accuracy(target, output)
        
        print(f'\
Spike detection rate(GT): {result[0]:0.2f}%\n\
Spike detection rate:     {result[1]:0.2f}%\n\
Noise Ratio:              {result[2]:0.2f}%\n\
Neuron Loss:              {result[3]:0.2f}%\n\
Label clsfy acc:          {result[4]:0.2f}%\n\
Noise clsfy acc:          {result[5]:0.2f}%\n\
Total acc:                {result[6]:0.2f}%')
    
    @staticmethod
    def cal_accuracy(target,output):
        n_spike = target.shape[0]
        cm = confusion_matrix(target,output)
        assert n_spike == cm.sum()

        conf1 = np.array([[cm[:-1,:-1].sum(),cm[:-1,-1:].sum()],[cm[-1:,:-1].sum(),cm[-1:,-1:].sum()]])
        n_real_neuron = conf1[0,0]+conf1[1,0]
        target_detection = 100*n_real_neuron/n_spike
        n_detected_neuron = conf1[0,0]+conf1[0,1]
        output_detection = 100*n_detected_neuron/n_spike

        noisy = 100*(1 - (conf1[0,0] / (conf1[0,0]+conf1[1,0])))

        neuron_loss = 100*(1 - (conf1[0,0] / (conf1[0,0]+conf1[0,1])))

        label_clsfy_acc = 100*cm.diagonal()[:-1].sum()/conf1[0,0]

        noise_clsfy_acc = 100*conf1.diagonal().sum()/conf1.sum()
        
        total_acc = 100*cm.diagonal().sum()/cm.sum()
        
        return target_detection, output_detection, noisy, neuron_loss, label_clsfy_acc, noise_clsfy_acc, total_acc
        
