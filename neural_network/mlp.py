import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from typing import Optional


def _named_sequential(*modules) -> nn.Sequential:
    return nn.Sequential(OrderedDict(modules))


class MLPWithEmbeddings(nn.Module):
    def __init__(
        self,
        d_num: int,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        dropout: float,
        skip_connection: bool,
        n_species: int,
        species_emb_dim: int,
        n_habitats: int,
        habitat_emb_dim: int,
    ) -> None:
        super().__init__()

        self.species_emb = nn.Embedding(n_species, species_emb_dim)
        self.habitat_emb = nn.Embedding(n_habitats, habitat_emb_dim)

        d_in_total = d_num + species_emb_dim + habitat_emb_dim

        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, however: {n_blocks=}")

        if skip_connection:
            assert n_blocks > 1

        blocks = []
        for i in range(n_blocks):
            if i == 0:
                blocks.append(
                    _named_sequential(
                        ("linear", nn.Linear(d_in_total, d_block)),
                        ("activation", nn.ReLU()),
                        ("dropout", nn.Dropout(dropout)),
                    )
                )
            elif skip_connection and (i == n_blocks // 2):
                blocks.append(
                    _named_sequential(
                        ("linear", nn.Linear(d_block + d_in_total, d_block)),
                        ("activation", nn.ReLU()),
                        ("dropout", nn.Dropout(dropout)),
                    )
                )
            else:
                blocks.append(
                    _named_sequential(
                        ("linear", nn.Linear(d_block, d_block)),
                        ("activation", nn.ReLU()),
                        ("dropout", nn.Dropout(dropout)),
                    )
                )

        self.blocks = nn.ModuleList(blocks)
        self.output = None if d_out is None else nn.Linear(d_block, d_out)
        self.skip_connection = skip_connection

    def forward(
        self,
        x_num: Tensor,
        species_idx: Tensor,
        habitat_freqs: Tensor,
    ) -> Tensor:
        species_vec = self.species_emb(species_idx).squeeze(1)
        habitat_weights = habitat_freqs / (habitat_freqs.sum(dim=1, keepdim=True) + 1e-8)
        habitat_vec = habitat_weights @ self.habitat_emb.weight

        x = torch.cat([x_num, species_vec, habitat_vec], dim=-1)

        out = x.clone()
        for i, block in enumerate(self.blocks):
            if self.skip_connection and (i == len(self.blocks) // 2):
                out = torch.cat([x, out], axis=-1)
            out = block(out)

        if self.output is not None:
            out = self.output(out)
        return out


class MLP(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_block: int,
        dropout: float = 0,
        skip_connection: bool = False,
    ) -> None:
        """
        Args:
            d_in: the input size.
            d_out: the output size.
            n_blocks: the number of blocks.
            d_block: the block width.
            dropout: the dropout rate.
            skip_connection: add skip connection.
        """
        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, however: {n_blocks=}")

        if skip_connection:
            assert n_blocks > 1

        super().__init__()
        blocks = []
        for i in range(n_blocks):
            if i == 0:
                blocks.append(
                    _named_sequential(
                        ("linear", nn.Linear(d_in, d_block)),
                        ("activation", nn.ReLU()),
                        ("dropout", nn.Dropout(dropout)),
                    )
                )
            elif skip_connection and (i == n_blocks // 2):
                blocks.append(
                    _named_sequential(
                        ("linear", nn.Linear(d_block + d_in, d_block)),
                        ("activation", nn.ReLU()),
                        ("dropout", nn.Dropout(dropout)),
                    )
                )
            else:
                blocks.append(
                    _named_sequential(
                        ("linear", nn.Linear(d_block, d_block)),
                        ("activation", nn.ReLU()),
                        ("dropout", nn.Dropout(dropout)),
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.output = None if d_out is None else nn.Linear(d_block, d_out)
        self.skip_connection = skip_connection

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        out = x.clone()
        for i, block in enumerate(self.blocks):
            if self.skip_connection and (i == len(self.blocks) // 2):
                out = torch.cat([x, out], axis=-1)
            out = block(out)
        if self.output is not None:
            out = self.output(out)
        return out

