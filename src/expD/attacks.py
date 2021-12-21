import numpy as np

import torch
import torch.nn as nn

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)



def norms(Z, ord=2):
    """Compute norms over all but the first dimension"""
    return Z.norm(p=ord, dim=1).reshape(-1, 1)


def pgd_l2_mfld_clf_attack(model_fn, x, y, eps_iter=0.01, nb_iter=40, norm=2, verbose=False):
    """
    On-manifold adversarial attack for standard classifier.
    As described in: https://arxiv.org/pdf/1801.02774.pdf
    """
    alpha = eps_iter
    delta = torch.zeros_like(x, requires_grad=True)
    losses = None
    pred_labels = None
    if verbose:
        losses = list()
        pred_labels = torch.zeros(nb_iter, x.shape[0])
    for t in range(nb_iter):
        # print(t, X.shape, delta.shape, (X+delta).shape)
        logits = model_fn(x + delta)
        y_pred = torch.max(logits, 1)[1]
        # if greedy and not (y_pred == y).all():
        #     # at least one error detected, so break
        #     break
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        if verbose:
            losses.append(loss.cpu().detach().item())
            pred_mfld = torch.max(logits, dim=1)[1]
            pred_labels[t] = pred_mfld.reshape(-1)
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach(), norm)

        delta.data = ((x + delta.detach()) * (norms(x, norm) / norms(x + delta.detach(), norm))) - x
        delta.grad.zero_()
    
    if verbose:
        return (x + delta).detach(), losses, pred_labels
    return (x + delta).detach()

def pgd_l2_mfld_dist_attack(model_fn, x, y, eps_iter=0.01, nb_iter=40, norm=2, verbose=False):
    """
    on-manifold PGD attack

    Note: if using `greedy == True` then use batch size as 1 in the dataloader so that X is a single sample
    for most accurate results
    """
    alpha = eps_iter
    delta = torch.zeros_like(x, requires_grad=True)
    losses = None
    pred_labels = None
    if verbose:
        losses = list()
        pred_labels = torch.zeros(nb_iter, x.shape[0])
    for t in range(nb_iter):
        # print(t, X.shape, delta.shape, (X+delta).shape)
        logits = model_fn(x + delta)
        y_pred = torch.min(logits, 1)[1]
        true_labels = torch.min(y, 1)[1]
        # if greedy and not (y_pred == true_labels).all():
        #     # first mis-classificaton detected, so break
        #     break
        # loss = nn.MSELoss()(logits, y)
        loss = torch.mean(logits[:, true_labels] - logits[:, (true_labels + 1) % y.shape[1]])
        loss.backward()

        if verbose:
            losses.append(loss.cpu().detach().item())
            pred_mfld = torch.min(logits, dim=1)[1]
            pred_labels[t] = pred_mfld.reshape(-1)
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data = ((x + delta.detach()) * (norms(x, norm) / norms(x + delta.detach(), norm))) - x
        delta.grad.zero_()
    if verbose:
        return (x + delta).detach(), losses, pred_labels
    return (x + delta).detach()

def pgd_l2_mfld_eps_attack(model_fn, x, y, loss_func=nn.CrossEntropyLoss(), eps_iter=0.01, nb_iter=1000):
    """
    on-manifold attack that also restricts the perturbation within an epsilon ball of the original sample

    Note: if using `greedy == True` then use batch size as 1 in the dataloader so that X is a single sample
    for most accurate results
    """
    pass

def pgd_cls(model_fn, x, y, eps=0.1, eps_iter=0.1, nb_iter=40, norm=2, restarts=1):

    if norm == 2:
        # note: std. l2 pgd attack does not have random restarts enabled right now
        adv_x = pgd_l2_cls(model_fn, x, y, eps, eps_iter, nb_iter, verbose=False)
    elif norm == np.inf:
        adv_x = pgd_linf_rand(model_fn, x, y, eps, eps_iter, nb_iter, restarts)
    
    return adv_x

def pgd_l2_cls(model_fn, x, y, eps=0.1, eps_iter=0.1, nb_iter=40, verbose=False):

    alpha = eps_iter
    delta = torch.zeros_like(x, requires_grad=True)
    if verbose:
        losses = list()
    for t in range(nb_iter):
        loss = nn.CrossEntropyLoss()(model_fn(x + delta), y)
        loss.backward()
        losses.append(loss.cpu().detach().item())
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach(), 2)
        # delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.data *= eps / norms(delta.detach()).clamp(min=eps)
        delta.grad.zero_()
    
    if verbose:
        return (x + delta).detach(), losses
    return (x + delta).detach()

def pgd_linf_rand(model_fn, x, y, eps=0.1, eps_iter=0.1, nb_iter=40, restarts=1):
    """ Construct PGD adversarial examples on the samples X, with random restarts"""
    
    epsilon = eps
    alpha = eps_iter
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_delta = torch.zeros_like(x)
    
    for i in range(restarts):
        delta = torch.rand_like(x, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
        
        for t in range(nb_iter):
            loss = nn.CrossEntropyLoss()(model_fn(x + delta), y)
            loss.backward()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()
        
        all_loss = nn.CrossEntropyLoss(reduction='none')(model_fn(x+delta),y)
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
        
    return (x + max_delta).detach()

def pgd_dist(model_fn, x, y, norm=2, eps=0.1, eps_iter=0.1, nb_iter=40, verbose=False):
    epsilon = eps
    alpha = eps_iter
    delta = torch.zeros_like(x, requires_grad=True)

    losses = None
    pred_labels = None
    if verbose:
        losses = list()
        pred_labels = torch.zeros((nb_iter, x.shape[0]))
    labels = torch.min(y, dim=1)[1]
    for t in range(nb_iter):
        # loss = nn.MSELoss()(model(X + delta), y)
        logits = model_fn(x + delta)
        
        loss = torch.mean(logits[:, labels] - logits[:, (labels + 1) % y.shape[1]])
        loss.backward()
        if verbose:
            losses.append(loss.cpu().detach().item())
            pred_mfld = torch.min(logits, dim=1)[1]
            pred_labels[t] = pred_mfld.reshape(-1)
        # delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        if norm == 2:
            delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        elif norm == np.inf:
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    
    if verbose:
        return (x + delta).detach(), losses, pred_mfld
    
    return (x + delta).detach()

# attacks = {
#     "pgd_l2_mfld_clf": pgd_l2_mfld_clf_attack,
#     "pgd_l2_mfld_dist": pgd_l2_mfld_dist_attack,
#     "pgd_l2_cls": pgd_l2_cls,
#     "pgd_linf_rand": pgd_linf_rand,
#     "pgd_dist": pgd_dist,
#     "chans_pgd": projected_gradient_descent,
#     "chans_fgsm": fast_gradient_method
# }

attacks = {
    "std_pgd" : {
        "clf": {
            "my": pgd_cls,
            "chans": projected_gradient_descent
        },
        "dist": {
            "my": pgd_dist
        }
    },
    "onmfld_pgd": {
        "clf": {
            "my": pgd_l2_mfld_clf_attack
        }
    }
}