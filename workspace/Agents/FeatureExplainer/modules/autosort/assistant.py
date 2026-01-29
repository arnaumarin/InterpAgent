# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Assistant utility for automatically load network from network
description."""

import torch
import torch.nn.functional as F
import numpy as np

class Assistant:
    """Assistant that bundles training, validation and testing workflow.

    Parameters
    ----------
    net : torch.nn.Module
        network to train.
    error : object or lambda
        an error object or a lambda function that evaluates error.
        It is expected to take ``(output, target)`` | ``(output, label)``
        as it's argument and return a scalar value.
    optimizer : torch optimizer
        the learning optimizer.
    stats : slayer.utils.stats
        learning stats logger. If None, stats will not be logged.
        Defaults to None.
    classifier : slayer.classifier or lambda
        classifier object or lambda function that takes output and
        returns the network prediction. None means regression mode.
        Classification steps are bypassed.
        Defaults to None.
    count_log : bool
        flag to enable count log. Defaults to False.
    lam : float
        lagrangian to merge network layer based loss.
        None means no such additional loss.
        If not None, net is expected to return the accumulated loss as second
        argument. It is intended to be used with layer wise sparsity loss.
        Defaults to None.

    Attributes
    ----------
    net
    error
    optimizer
    stats
    classifier
    count_log
    lam
    device : torch.device or None
        the main device memory where network is placed. It is not at start and
        gets initialized on the first call.
    """
    def __init__(
        self,
        net, error=None, optimizer=None, device=None
    ):
        self.net = net
        self.error = error
        self.optimizer = optimizer
        self.device = device

    def reduce_lr(self, factor=10 / 3):
        """Reduces the learning rate of the optimizer by ``factor``.

        Parameters
        ----------
        factor : float
            learning rate reduction factor. Defaults to 10/3.

        Returns
        -------

        """
        for param_group in self.optimizer.param_groups:
            print('\nLearning rate reduction from', param_group['lr'])
            param_group['lr'] /= factor

    def train(self, input, target, stats=None):
        """Training assistant.

        Parameters
        ----------
        input : torch tensor
            input tensor.
        target : torch tensor
            ground truth or label.

        Returns
        -------
        output
            network's output.
        count : optional
            spike count if ``count_log`` is enabled

        """
        self.net.train()

        if self.device is None:
            for p in self.net.parameters():
                self.device = p.device
                break
        device = self.device

        input = input.to(device)
        target = target.to(device)
        output = self.net(input)
        one_hot = F.one_hot(target, num_classes=output.shape[-1])
        loss = self.error(output.float(), one_hot.float())

        if stats is not None:
            stats.num_samples += input.shape[0]
            stats.loss_sum += loss.cpu().data.item()* output.shape[0]
            stats.correct += torch.sum(torch.argmax(output,axis=1) == target).cpu().detach().item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        return output


    def test(self, input, target, stats=None):
        """Testing assistant.

        Parameters
        ----------
        input : torch tensor
            input tensor.
        target : torch tensor
            ground truth or label.

        Returns
        -------
        output
            network's output.
        count : optional
            spike count if ``count_log`` is enabled

        """
        self.net.eval()

        if self.device is None:
            for p in self.net.parameters():
                self.device = p.device
                break
        device = self.device

        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)
            output = self.net(input)
            one_hot = F.one_hot(target, num_classes=output.shape[-1])
            loss = self.error(output.float(), one_hot.float())

            if stats is not None:
                stats.num_samples += input.shape[0]
                stats.loss_sum += loss.cpu().data.item()* output.shape[0]
                stats.correct += (np.argmax(output,axis=1) == target).sum()
                stats.pred_label += list(torch.argmax(output,axis=1).cpu().detach())
        
            return output
        
    def real(self, input, stats=None):
        """Testing assistant.

        Parameters
        ----------
        input : torch tensor
            input tensor.
        target : torch tensor
            ground truth or label.

        Returns
        -------
        output
            network's output.
        count : optional
            spike count if ``count_log`` is enabled

        """
        self.net.eval()

        if self.device is None:
            for p in self.net.parameters():
                self.device = p.device
                break
        device = self.device

        with torch.no_grad():
            input = input.to(device)
            output = self.net(input)

            if stats is not None:
                stats.num_samples += input.shape[0]
                stats.pred_label += list(torch.argmax(output,axis=1).cpu().detach())
        
            return output
        

class stats():
    def __init__(self):
        self.best_acc = None
        self.idk = ''
        self.reset()
        
    def reset(self):
        self.num_samples = 0
        self.loss_sum = 0
        self.correct = 0
        self.pred_label = []
        
    def update(self):
        if self.best_acc is None or self.accuracy > self.best_acc:
            self.best_acc = self.accuracy
        self.reset()
        
    
    @property
    def loss(self):
        """Current loss."""
        if self.num_samples > 0:
            return self.loss_sum / self.num_samples
        else:
            return None

    @property
    def accuracy(self):
        """Current accuracy."""
        if self.num_samples > 0:
            return self.correct / self.num_samples
        else:
            return None
        
    @property
    def best_epoch(self):
        if self.best_acc is None or self.accuracy > self.best_acc:
            return True
        else:
            return False

        
    def __str__(self):
        """String method.
        """
        if self.best_acc is None or self.accuracy > self.best_acc:
            return (f'loss = {self.loss:11.5f} '
                    f'{self.idk}{" "*4}'
                    f'accuracy = {self.accuracy:7.5f} '
                    f'(max = {self.accuracy:7.5f}) '
                    f'{self.idk}'
                    )
        else:
            return (f'loss = {self.loss:11.5f} '
                    f'{self.idk}{" "*4}'
                    f'accuracy = {self.accuracy:7.5f} '
                    f'(max = {self.best_acc:7.5f}) '
                    f'{self.idk}')
