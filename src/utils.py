import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
import deepspeed


def neftune_post_forward_hook(module, input, output, neftune_noise_alpha=5):
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for
    torch.nn.Embedding layers. This method is slightly adapted from the original source code
    that can be found here: https://github.com/neelsjain/NEFTune

    Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```

    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set
            `module.neftune_noise_alpha` to the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    # if module.training:
    #     input_mask = data['attention_mask'].to(embeds_init) # B x L
    #     input_lengths = torch.sum(input_mask, 1) # B

    #     noise_ = torch.zeros_like(embeds_init).uniform_(-1,1)
    #     delta = noise_ * input_mask.unsqueeze(2)
    #     dims = input_lengths * embeds_init.size(-1)
    #     mag = args.neftune_alpha / torch.sqrt(dims)
    #     delta = (delta * mag.view(-1, 1, 1)).detach()
    #     batch['inputs_embeds'] = delta + embeds_init

    if module.training:
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = neftune_noise_alpha / torch.sqrt(dims)
        noise = torch.zeros_like(output).uniform_(-mag_norm, mag_norm).detach()
        output = output + noise
    return output


def compute_kl_divergence_loss(output_logits, ref_logits, input_ids, labels, kl_penalty="full"):
    """
    This function computes the KL divergence loss between the output logits and the reference logits.
    It ignores the loss for tokens where the corresponding label is -100.
    Returns the KL divergence loss.
    """

    # compute logprobs for tokens
    if kl_penalty == "full":
        # if compute KL divergence loss for all output distributions
        active_logprobs = logprobs_from_logits(output_logits[:, :-1, :], None, gather=False)
        ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], None, gather=False)
    elif kl_penalty == "target_token":
        # if compute the KL divergence loss for the target token only
        active_logprobs = logprobs_from_logits(output_logits[:, :-1, :], input_ids[:, 1:])
        ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])
    else:
        raise NotImplementedError

    # Shift the labels to the right
    shift_labels = labels[:, 1:]

    # compute the token-wise KL divergence
    token_wise_kl = compute_kl_penalty(active_logprobs, ref_logprobs, kl_penalty)
    
    # Create a mask where labels are not equal to -100
    mask = (shift_labels != -100).float()

    # Apply the mask to the token-wise KL by multiplying. This zeros out the loss where labels are -100.
    # Ensure the dimensions match, might need to adjust depending on your logprob dimensions
    masked_kl = token_wise_kl * mask

    # Compute the mean of the masked KL, only considering non-zero (non-masked) elements
    kl_loss = masked_kl.sum() / mask.sum()
    
    return kl_loss.mean()


def compute_kl_divergence_loss_target_token(output_logits, ref_logprobs, input_ids, labels):
    active_logprobs = logprobs_from_logits(output_logits[:, :-1, :], input_ids[:, 1:])

    # Shift the labels to the right
    shift_labels = labels[:, 1:]

    # compute the token-wise KL divergence
    token_wise_kl = compute_kl_penalty(active_logprobs, ref_logprobs, kl_penalty="kl")
    
    # Create a mask where labels are not equal to -100
    mask = (shift_labels != -100).float()

    # Apply the mask to the token-wise KL by multiplying. This zeros out the loss where labels are -100.
    # Ensure the dimensions match, might need to adjust depending on your logprob dimensions
    masked_kl = token_wise_kl * mask

    # Compute the mean of the masked KL, only considering non-zero (non-masked) elements
    kl_loss = masked_kl.sum() / mask.sum()
    
    return kl_loss.mean()

def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True):
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def compute_kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty: str = "full"):
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty == "full":
        # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
        return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError