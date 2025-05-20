import logging
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from vecraft_exception_model.exception import TsnePlotGeneratingFailureException, NullOrZeroVectorException


class TSNEManager:
    """
    Manager for t-SNE (t-Distributed Stochastic Neighbor Embedding) plot generation.

    This class handles the generation of t-SNE plots for visualizing
    high-dimensional vector data in a collection. It supports visualizing
    all records in a collection or a specified subset.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use for logging messages. If None, a new logger will be created.

    Methods
    -------
    generate_tsne_plot(name, version, get_record_func, record_ids=None, perplexity=30,
                      random_state=42, outfile="tsne.png")
        Generate a t-SNE scatter plot for the specified records.
    """
    def __init__(self, logger=None):
        self._logger = logger or logging.getLogger(__name__)

    def generate_tsne_plot(
            self,
            name: str,
            version,
            get_record_func,
            record_ids: Optional[List[str]] = None,
            perplexity: int = 30,
            random_state: int = 42,
            outfile: str = "tsne.png"
    ) -> str:
        """
        Generate a t-SNE scatter plot for the given record IDs (or all records if None).

        Args:
            name: Collection name.
            version: Collection version to use.
            get_record_func: Function to retrieve a record by ID.
            record_ids: Optional list of record IDs to visualize.
            perplexity: t-SNE perplexity parameter.
            random_state: Random seed for reproducibility.
            outfile: Path to save the generated PNG image.

        Returns:
            Path to the saved t-SNE plot image.
        """
        try:
            self._logger.info(f"Generating t-SNE plot for collection {name}")
            start_time = time.time()

            # Determine which IDs to plot
            if record_ids is None:
                record_ids = list(version.storage.get_all_record_locations().keys())
                self._logger.debug(f"Using all {len(record_ids)} records in collection")
            else:
                self._logger.debug(f"Using {len(record_ids)} specified record IDs")

            vectors = []
            labels = []
            for rid in record_ids:
                rec = get_record_func(version, rid)
                if not rec:
                    self._logger.warning(f"Record {rid} not found, skipping for t-SNE")
                    continue
                vectors.append(rec.vector)
                labels.append(rid)

            if not vectors:
                err_msg = "No vectors available for t-SNE visualization"
                self._logger.error(err_msg)
                raise NullOrZeroVectorException(err_msg)

            # Stack into a 2D array
            data = np.vstack(vectors)
            self._logger.debug(f"Processing {len(vectors)} vectors of dimension {vectors[0].shape[0]}")

            # Log the parameters
            self._logger.debug(f"t-SNE parameters: perplexity={perplexity}, random_state={random_state}")
            self._logger.debug(f"Output file: {outfile}")

            # Call the helper to generate and save the plot
            plot = self._generate_tsne(
                vectors=data,
                labels=labels,
                outfile=outfile,
                perplexity=perplexity,
                random_state=random_state
            )

            elapsed = time.time() - start_time
            self._logger.info(f"T-SNE plot from {name} completed in {elapsed:.3f}s")

            return plot

        except Exception as e:
            error_message = f"Error generating t-SNE plot for collection {name}: {e}"
            self._logger.error(error_message)
            raise TsnePlotGeneratingFailureException(error_message, name, e)

    @staticmethod
    def _generate_tsne(vectors: np.ndarray,
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
                ax.annotate(lab, (emb[i, 0], emb[i, 1]), fontsize=7, alpha=0.7)
        ax.set_title("t‑SNE")
        fig.tight_layout()
        fig.savefig(outfile, dpi=300)
        plt.close(fig)
        return outfile