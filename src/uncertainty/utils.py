import logging
import math
import torch
from torch import Tensor


def check(
    scores: Tensor, max_value: float = math.inf, epsilon: float = 1e-6, score_type: str = ""
) -> Tensor:
    """
    Warn if any element of scores is negative, a nan or exceeds max_value.

    We set epilson = 1e-6 based on the fact that torch.finfo(torch.float).eps ~= 1e-7.
    """
    if not torch.all((scores + epsilon >= 0) & (scores - epsilon <= max_value)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()
        import pdb
        pdb.set_trace()

        logging.warning(f"Invalid {score_type} score (min = {min_score}, max = {max_score})")
    
    return scores

def margin_prob(probs: Tensor) -> Tensor:
    """
    See marginal_entropy_from_logprobs.

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    probs = torch.mean(probs, dim=1)  # [N, Cl]
    probs_sorted, idxs = probs.sort(descending=True)
    scores = probs_sorted[:, 0] - probs_sorted[:,1]
    return -scores  # [N,]

def margin_logprob(logprobs: Tensor) -> Tensor:
    return margin_prob(logprobs.exp())

# -------------------------------------------------

def least_conf_prob(probs: Tensor) -> Tensor:
    """
    See marginal_entropy_from_logprobs.

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    probs = torch.mean(probs, dim=1)  # [N, Cl]
    scores = probs.max(1)[0]
    return -scores  # [N,]

def least_conf_logprob(logprobs: Tensor) -> Tensor:
    return least_conf_prob(logprobs.exp())

#---------------------------------------------------

def meanstd_prob(probs: Tensor) -> Tensor:
    """
    See marginal_entropy_from_logprobs.

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    probs_std = torch.std(probs, dim=1)  # [N, Cl]
    scores = torch.mean(probs_std, dim = 1)
    return scores  # [N,]

def meanstd_logprob(logprobs: Tensor) -> Tensor:
    return meanstd_prob(logprobs.exp())

# ---------------------------------------------------

def marginal_entropy_probs(probs: Tensor) -> Tensor:
    """
    See marginal_entropy_from_logprobs.

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    probs = torch.mean(probs, dim=1)  # [N, Cl]
    logprobs = torch.log(probs + 1e-6)
    #scores = entropy_from_probs(probs)  # [N,]
    #scores = check(scores, math.log(probs.shape[-1]), score_type="ME")  # [N,]
    return -torch.sum(probs * logprobs, dim=1)

def marginal_entropy_logprobs(logprobs: Tensor) -> Tensor:
    return marginal_entropy_probs(logprobs.exp())