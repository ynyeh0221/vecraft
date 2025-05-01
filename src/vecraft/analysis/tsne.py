
"""t‑SNE visualization helper."""
from typing import List, Optional
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def generate_tsne(vectors: np.ndarray,
                  labels: Optional[List[str]] = None,
                  outfile: str = "tsne.png",
                  perplexity: int = 30,
                  random_state: int = 42) -> str:
    """Run t‑SNE and save to *outfile* (PNG)."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2‑D")
    tsne = TSNE(n_components=2, perplexity=perplexity, init='random', random_state=random_state)
    emb = tsne.fit_transform(vectors)
    fig, ax = plt.subplots()
    ax.scatter(emb[:, 0], emb[:, 1], s=10)
    if labels is not None:
        for i, lab in enumerate(labels):
            ax.annotate(lab, (emb[i,0], emb[i,1]), fontsize=7, alpha=0.7)
    ax.set_title("t‑SNE")
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    return outfile