# Implementing similarity fucntions.
import numpy as np
from numpy.linalg import norm

import torch
import torch.nn as n

def compute_cosine_similarity(a, b):
    """Compute cosine similarity."""
    return np.dot(a,b) / (norm(a)*norm(b))


def compute_consine_similarity_from_matrix(a,b, eps=1e-8):
    """Computer consime similairy between two matrices.
    
    Args:
      a: Tensor, matrix with shape (num_query, dim).
      b: Tensor, matrix with shape (num_stories, dim).
      eps: float, added for numerical stability.
    Returns:
      sim_mt: Tensor, matrix with shape (num_query, num_data).
              The i-th row specifies query i over all sentences (stories).
    """
    a_n, b_n = a.norm(dim=1)[:,None], b.norm(dim=1)[:,None]
    # Avoide deviding 0
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mat = torch.mm(a_norm, b_norm.transpose(0,1))
    return sim_mat
