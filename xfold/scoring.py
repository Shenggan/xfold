# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Library of scoring methods of the model outputs."""

from xfold import protein_data_processing

import torch


def pseudo_beta_fn(
    aatype: torch.Tensor,
    dense_atom_positions: torch.Tensor,
    dense_atom_masks: torch.Tensor,
    is_ligand: torch.Tensor | None = None,
    use_jax: torch.Tensor | None = True,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Create pseudo beta atom positions and optionally mask.

    Args:
      aatype: [num_res] amino acid types.
      dense_atom_positions: [num_res, NUM_DENSE, 3] vector of all atom positions.
      dense_atom_masks: [num_res, NUM_DENSE] mask.
      is_ligand: [num_res] flag if something is a ligand.
      use_jax: whether to use jax for the computations.

    Returns:
      Pseudo beta dense atom positions and the corresponding mask.
    """
    if is_ligand is None:
        is_ligand = torch.zeros_like(aatype, dtype=torch.bool)

    pseudobeta_index_polymer = torch.take_along_dim(
        protein_data_processing.RESTYPE_PSEUDOBETA_INDEX.to(device=aatype.device), aatype.to(dtype=torch.int64), axis=0
    ).to(dtype=torch.int32)

    pseudobeta_index = torch.where(
        is_ligand.to(dtype=torch.bool),
        torch.zeros_like(pseudobeta_index_polymer),
        pseudobeta_index_polymer,
    ).to(dtype=torch.int64)

    pseudo_beta = torch.take_along_dim(
        dense_atom_positions, pseudobeta_index[..., None, None], dim=-2
    )
    pseudo_beta = torch.squeeze(pseudo_beta, dim=-2)

    pseudo_beta_mask = torch.take_along_dim(
        dense_atom_masks, pseudobeta_index[..., None], dim=-1
    ).to(dtype=torch.float32)
    pseudo_beta_mask = torch.squeeze(pseudo_beta_mask, dim=-1)

    return pseudo_beta, pseudo_beta_mask
