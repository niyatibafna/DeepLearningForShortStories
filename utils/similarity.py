import numpy as np
from numpy.linalg import norm

def compute_cosine_similarity(a, b):
    """Compute cosine similarity."""
    return np.dot(a,b) / (norm(a)*norm(b))

    
