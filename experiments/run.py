import functools
import os

import pandas as pd

from bayesian_protein.cli import parse_args
from bayesian_protein.embedding import embedding_factory
from bayesian_protein.optimizer import Optimizer
from bayesian_protein.types import (
    EmbeddingModel,
)


@functools.lru_cache(maxsize=None)
def get_data_with_embeddings(embedding_model: EmbeddingModel, csvfile: str):
    """
    Calculate embeddings for molecules in the csv file. Molecules are expected to be in SMILES format and
    the file should contain a column named "smiles"
    """
    data = pd.read_csv(csvfile)
    embedder = embedding_factory(embedding_model, batch_size=256)
    embeddings = embedder(data["smiles"].tolist())
    data.loc[:, "embedding"] = embeddings.tolist()
    return data


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Huggingface tokenizers

    jobs = parse_args()
    for args in jobs:
        optimizer = Optimizer(
            args,
            get_data_with_embeddings(
                embedding_model=args.embedding, csvfile=str(args.data)
            ).copy(),
            job_id=0,
            database_path=str(args.out / "protein_ligand.sqlite"),
        )
        print(args.out)
        optimizer.run()
