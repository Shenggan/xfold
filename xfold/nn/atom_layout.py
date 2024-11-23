import dataclasses
from typing import Any

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class AtomLayout:
    """Atom layout in a fixed shape (usually 1-dim or 2-dim).

    Examples for atom layouts are atom37, atom14, and similar.
    All members are np.ndarrays with the same shape, e.g.
    - [num_atoms]
    - [num_residues, max_atoms_per_residue]
    - [num_fragments, max_fragments_per_residue]
    All string arrays should have dtype=object to avoid pitfalls with Numpy's
    fixed-size strings

    Attributes:
      atom_name: np.ndarray of str: atom names (e.g. 'CA', 'NE2'), padding
        elements have an empty string (''), None or any other value, that maps to
        False for .astype(bool). mmCIF field: _atom_site.label_atom_id.
      res_id: np.ndarray of int: residue index (usually starting from 1) padding
        elements can have an arbitrary value. mmCIF field:
        _atom_site.label_seq_id.
      chain_id: np.ndarray of str: chain names (e.g. 'A', 'B') padding elements
        can have an arbitrary value. mmCIF field: _atom_site.label_seq_id.
      atom_element: np.ndarray of str: atom elements (e.g. 'C', 'N', 'O'), padding
        elements have an empty string (''), None or any other value, that maps to
        False for .astype(bool). mmCIF field: _atom_site.type_symbol.
      res_name: np.ndarray of str: residue names (e.g. 'ARG', 'TRP') padding
        elements can have an arbitrary value. mmCIF field:
        _atom_site.label_comp_id.
      chain_type: np.ndarray of str: chain types (e.g. 'polypeptide(L)'). padding
        elements can have an arbitrary value. mmCIF field: _entity_poly.type OR
        _entity.type (for non-polymers).
      shape: shape of the layout (just returns atom_name.shape)
    """

    atom_name: np.ndarray
    res_id: np.ndarray
    chain_id: np.ndarray
    atom_element: np.ndarray | None = None
    res_name: np.ndarray | None = None
    chain_type: np.ndarray | None = None

    def __post_init__(self):
        """Assert all arrays have the same shape."""
        attribute_names = (
            'atom_name',
            'atom_element',
            'res_name',
            'res_id',
            'chain_id',
            'chain_type',
        )
        _assert_all_arrays_have_same_shape(
            obj=self,
            expected_shape=self.atom_name.shape,
            attribute_names=attribute_names,
        )
        # atom_name must have dtype object, such that we can convert it to bool to
        # obtain the mask
        if self.atom_name.dtype != object:
            raise ValueError(
                'atom_name must have dtype object, such that it can '
                'be converted converted to bool to obtain the mask'
            )

    def __getitem__(self, key: NumpyIndex) -> 'AtomLayout':
        return AtomLayout(
            atom_name=self.atom_name[key],
            res_id=self.res_id[key],
            chain_id=self.chain_id[key],
            atom_element=(
                self.atom_element[key] if self.atom_element is not None else None
            ),
            res_name=(self.res_name[key]
                      if self.res_name is not None else None),
            chain_type=(
                self.chain_type[key] if self.chain_type is not None else None
            ),
        )

    def __eq__(self, other: 'AtomLayout') -> bool:
        if not np.array_equal(self.atom_name, other.atom_name):
            return False

        mask = self.atom_name.astype(bool)
        # Check essential fields.
        for field in ('res_id', 'chain_id'):
            my_arr = getattr(self, field)
            other_arr = getattr(other, field)
            if not np.array_equal(my_arr[mask], other_arr[mask]):
                return False

        # Check optional fields.
        for field in ('atom_element', 'res_name', 'chain_type'):
            my_arr = getattr(self, field)
            other_arr = getattr(other, field)
            if (
                my_arr is not None
                and other_arr is not None
                and not np.array_equal(my_arr[mask], other_arr[mask])
            ):
                return False

        return True

    def copy_and_pad_to(self, shape: tuple[int, ...]) -> 'AtomLayout':
        """Copies and pads the layout to the requested shape.

        Args:
          shape: new shape for the atom layout

        Returns:
          a copy of the atom layout padded to the requested shape

        Raises:
          ValueError: incompatible shapes.
        """
        if len(shape) != len(self.atom_name.shape):
            raise ValueError(
                f'Incompatible shape {shape}. Current layout has shape {self.shape}.'
            )
        if any(new < old for old, new in zip(self.atom_name.shape, shape)):
            raise ValueError(
                "Can't pad to a smaller shape. Current layout has shape "
                f'{self.shape} and you requested shape {shape}.'
            )
        pad_width = [
            (0, new - old) for old, new in zip(self.atom_name.shape, shape)
        ]
        pad_val = np.array('', dtype=object)
        return AtomLayout(
            atom_name=np.pad(self.atom_name, pad_width,
                             constant_values=pad_val),
            res_id=np.pad(self.res_id, pad_width, constant_values=0),
            chain_id=np.pad(self.chain_id, pad_width, constant_values=pad_val),
            atom_element=(
                np.pad(self.atom_element, pad_width, constant_values=pad_val)
                if self.atom_element is not None
                else None
            ),
            res_name=(
                np.pad(self.res_name, pad_width, constant_values=pad_val)
                if self.res_name is not None
                else None
            ),
            chain_type=(
                np.pad(self.chain_type, pad_width, constant_values=pad_val)
                if self.chain_type is not None
                else None
            ),
        )

    def to_array(self) -> np.ndarray:
        """Stacks the fields to a numpy array with shape (6, <layout_shape>).

        Creates a pure numpy array of type `object` by stacking the 6 fields of the
        AtomLayout, i.e. (atom_name, atom_element, res_name, res_id, chain_id,
        chain_type). This method together with from_array() provides an easy way to
        apply pure numpy methods like np.concatenate() to `AtomLayout`s.

        Returns:
          np.ndarray of object with shape (6, <layout_shape>), e.g.
          array([['N', 'CA', 'C', ..., 'CB', 'CG', 'CD'],
           ['N', 'C', 'C', ..., 'C', 'C', 'C'],
           ['LEU', 'LEU', 'LEU', ..., 'PRO', 'PRO', 'PRO'],
           [1, 1, 1, ..., 403, 403, 403],
           ['A', 'A', 'A', ..., 'D', 'D', 'D'],
           ['polypeptide(L)', 'polypeptide(L)', ..., 'polypeptide(L)']],
          dtype=object)
        """
        if (
            self.atom_element is None
            or self.res_name is None
            or self.chain_type is None
        ):
            raise ValueError('All optional fields need to be present.')

        return np.stack(dataclasses.astuple(self), axis=0)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'AtomLayout':
        """Creates an AtomLayout object from a numpy array with shape (6, ...).

        see also to_array()
        Args:
          arr: np.ndarray of object with shape (6, <layout_shape>)

        Returns:
          AtomLayout object with shape (<layout_shape>)
        """
        if arr.shape[0] != 6:
            raise ValueError(
                'Given array must have shape (6, ...) to match the 6 fields of '
                'AtomLayout (atom_name, atom_element, res_name, res_id, chain_id, '
                f'chain_type). Your array has {arr.shape=}'
            )
        return cls(*arr)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.atom_name.shape


@dataclasses.dataclass(frozen=True)
class GatherInfo:
    """Gather indices to translate from one atom layout to another.

    All members are np or jnp ndarray (usually 1-dim or 2-dim) with the same
    shape, e.g.
    - [num_atoms]
    - [num_residues, max_atoms_per_residue]
    - [num_fragments, max_fragments_per_residue]

    Attributes:
      gather_idxs: np or jnp ndarray of int: gather indices into a flattened array
      gather_mask: np or jnp ndarray of bool: mask for resulting array
      input_shape: np or jnp ndarray of int: the shape of the unflattened input
        array
      shape: output shape. Just returns gather_idxs.shape
    """

    gather_idxs: torch.Tensor
    gather_mask: torch.Tensor
    input_shape: torch.Tensor

    def __post_init__(self):
        if self.gather_mask.shape != self.gather_idxs.shape:
            raise ValueError(
                'All arrays must have the same shape. Got\n'
                f'gather_idxs.shape = {self.gather_idxs.shape}\n'
                f'gather_mask.shape = {self.gather_mask.shape}\n'
            )

    def __getitem__(self, key: Any) -> 'GatherInfo':
        return GatherInfo(
            gather_idxs=self.gather_idxs[key],
            gather_mask=self.gather_mask[key],
            input_shape=self.input_shape,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.gather_idxs.shape

    def as_dict(
        self,
        key_prefix: str | None = None,
    ) -> dict[str, torch.Tensor]:
        prefix = f'{key_prefix}:' if key_prefix else ''
        return {
            prefix + 'gather_idxs': self.gather_idxs,
            prefix + 'gather_mask': self.gather_mask,
            prefix + 'input_shape': self.input_shape,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict[str, torch.Tensor],
        key_prefix: str | None = None,
    ) -> 'GatherInfo':
        """Creates GatherInfo from a given dictionary."""
        prefix = f'{key_prefix}:' if key_prefix else ''
        return cls(
            gather_idxs=d[prefix + 'gather_idxs'],
            gather_mask=d[prefix + 'gather_mask'],
            input_shape=d[prefix + 'input_shape'],
        )


def convert(
    gather_info: GatherInfo,
    arr: torch.Tensor,
    *,
    layout_axes: tuple[int, ...] = (0,),
) -> torch.Tensor:
    """Convert an array from one atom layout to another."""
    # Translate negative indices to the corresponding positives.
    layout_axes = tuple(i if i >= 0 else i + arr.ndim for i in layout_axes)

    # Ensure that layout_axes are continuous.
    layout_axes_begin = layout_axes[0]
    layout_axes_end = layout_axes[-1] + 1

    if layout_axes != tuple(range(layout_axes_begin, layout_axes_end)):
        raise ValueError(f'layout_axes must be continuous. Got {layout_axes}.')
    layout_shape = arr.shape[layout_axes_begin:layout_axes_end]
    gather_info_input_shape = gather_info.input_shape.numpy()

    # Ensure that the layout shape is compatible
    # with the gather_info. I.e. the first axis size must be equal or greater
    # than the gather_info.input_shape, and all subsequent axes sizes must match.
    if (len(layout_shape) != gather_info_input_shape.size) or (
        isinstance(gather_info_input_shape, np.ndarray)
        and (
            (layout_shape[0] < gather_info_input_shape[0])
            or (np.any(layout_shape[1:] != gather_info_input_shape[1:]))
        )
    ):
        raise ValueError(
            'Input array layout axes are incompatible. You specified layout '
            f'axes {layout_axes} with an input array of shape {arr.shape}, but '
            f'the gather info expects shape {gather_info.input_shape}. '
            'Your first axis size must be equal or greater than the '
            'gather_info.input_shape, and all subsequent axes sizes must '
            'match.'
        )

    # Compute the shape of the input array with flattened layout.
    batch_shape = arr.shape[:layout_axes_begin]
    features_shape = arr.shape[layout_axes_end:]
    arr_flattened_shape = batch_shape + \
        (np.prod(layout_shape),) + features_shape

    # Flatten input array and perform the gather.
    arr_flattened = arr.reshape(arr_flattened_shape)
    if layout_axes_begin == 0:
        out_arr = arr_flattened[gather_info.gather_idxs, ...]
    elif layout_axes_begin == 1:
        out_arr = arr_flattened[:, gather_info.gather_idxs, ...]
    elif layout_axes_begin == 2:
        out_arr = arr_flattened[:, :, gather_info.gather_idxs, ...]
    elif layout_axes_begin == 3:
        out_arr = arr_flattened[:, :, :, gather_info.gather_idxs, ...]
    elif layout_axes_begin == 4:
        out_arr = arr_flattened[:, :, :, :, gather_info.gather_idxs, ...]
    else:
        raise ValueError(
            'Only 4 batch axes supported. If you need more, the code '
            'is easy to extend.'
        )

    # Broadcast the mask and apply it.
    broadcasted_mask_shape = (
        (1,) * len(batch_shape)
        + gather_info.gather_mask.shape
        + (1,) * len(features_shape)
    )
    out_arr *= gather_info.gather_mask.reshape(broadcasted_mask_shape)
    return out_arr