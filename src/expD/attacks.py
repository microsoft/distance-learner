import numpy as np

import torch
import torch.nn as nn


def norms(Z, ord=2):
    """Compute norms over all but the first dimension"""
    return Z.norm(p=ord, dim=1).reshape(-1, 1)


def pgd_l2_mfld_clf_attack(model_fn, x, y, eps_iter=0.01, nb_iter=40, norm=2):
    """
    On-manifold adversarial attack for standard classifier.
    As described in: 
    """
    alpha = eps_iter
    delta = torch.zeros_like(x, requires_grad=True)
    losses = list()
    for t in range(num_iter):
        # print(t, X.shape, delta.shape, (X+delta).shape)
        logits = model_fn(x + delta)
        y_pred = torch.max(logits, 1)[1]
        # if greedy and not (y_pred == y).all():
        #     # at least one error detected, so break
        #     break
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        losses.append(loss.cpu().detach().item())
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())

        delta.data = ((X + delta.detach()) * (norms(X) / norms(X + delta.detach()))) - X
        delta.grad.zero_()
        
    return delta.detach(), losses

def pgd_l2_mfld_dist_attack(model, X, y, alpha=0.01, num_iter=1000, greedy=True):
    """
    on-manifold PGD attack

    Note: if using `greedy == True` then use batch size as 1 in the dataloader so that X is a single sample
    for most accurate results
    """
    delta = torch.zeros_like(X, requires_grad=True)
    losses = list()
    for t in range(num_iter):
        # print(t, X.shape, delta.shape, (X+delta).shape)
        logits = model(X + delta)
        y_pred = torch.min(logits, 1)[1]
        true_labels = torch.min(y, 1)[1]
        if greedy and not (y_pred == true_labels).all():
            # first mis-classificaton detected, so break
            break
        loss = nn.MSELoss()(logits, y)
        loss.backward()
        losses.append(loss.cpu().detach().item())
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data = ((X + delta.detach()) * (norms(X) / norms(X + delta.detach()))) - X
        delta.grad.zero_()
        
    return delta.detach(), losses

def pgd_l2_mfld_eps_attack(model, X, y, loss_func=nn.CrossEntropyLoss(), alpha=0.01, num_iter=1000, greedy=True):
    """
    on-manifold attack that also restricts the perturbation within an epsilon ball of the original sample

    Note: if using `greedy == True` then use batch size as 1 in the dataloader so that X is a single sample
    for most accurate results
    """
    pass



def pgd_l2_cls(model, X, y, epsilon=0.1, alpha=0.1, num_iter=40):
    delta = torch.zeros_like(X, requires_grad=True)
    losses = list()
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        losses.append(loss.cpu().detach().item())
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        # delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()
        
    return delta.detach(), losses

def pgd_linf_rand(model, X, y, epsilon, alpha, num_iter, restarts):
    """ Construct PGD adversarial examples on the samples X, with random restarts"""
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_delta = torch.zeros_like(X)
    
    for i in range(restarts):
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
        
        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()
        
        all_loss = nn.CrossEntropyLoss(reduction='none')(model(X+delta),y)
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
        
    return max_delta

def pgd_dist(model, X, y, norm=2, epsilon=0.1, alpha=0.1, num_iter=40):
    delta = torch.zeros_like(X, requires_grad=True)
    losses = list()
    labels = torch.min(y, dim=1)[1]
    for t in range(num_iter):
        # loss = nn.MSELoss()(model(X + delta), y)
        logits = model(X + delta)
        loss = torch.mean(logits[:, labels] - logits[:, (labels + 1) % y.shape[1]])
        loss.backward()
        losses.append(loss.cpu().detach().item())
        # delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        if norm == 2:
            delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        elif norm == np.inf:
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
        
    return delta.detach(), losses

