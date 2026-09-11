"""Encode only the LP routing features selected by an immutable index plan.

This is equivalent to concatenating all seven fields, applying the existing
finite/sign/log1p/float32 encoding, then selecting columns. The common dtype
is determined from *all* fields, including unselected fields. LP inputs are
never modified. This module does not validate or certify an LP solution.

NumPy handles one LP; CuPy handles a leading batch dimension. CuPy is imported
only by the GPU method, and device index metadata is uploaded once per device.
No input values, features, or device indices are downloaded to the host.
"""

import math
import operator
from types import MappingProxyType

import numpy as np


FIELDS = ('rhs', 'lower', 'upper', 'c', 'delta', 'col_scale', 'row_scale')


class SelectedLPFeatures:
    """Cache the positions of selected entries in the unencoded LP fields.

    Parameters
    ----------
    feature_indices : one-dimensional host integer array
        Nonnegative indices into the C-order flattened field concatenation.
        Ordering and repeated indices are retained. Negative indexing is not
        accepted, and CuPy indices must be explicitly downloaded by the caller
        during setup if needed; this class never performs such a download.
    field_shapes : mapping
        Shape of each field for one LP, without the GPU batch dimension.

    Field shapes are checked on every call. A different batch size is allowed;
    a changed per-LP shape requires a new plan, even if its size is unchanged.
    Use ``from_inputs`` to construct the plan from example arrays. Call the
    GPU method once before CUDA graph capture to initialize index metadata.
    """

    def __init__(self, feature_indices, field_shapes):
        if hasattr(feature_indices, '__cuda_array_interface__'):
            raise TypeError('feature_indices must be a host integer array')
        indices = np.asarray(feature_indices)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu':
            raise ValueError('feature_indices must be a one-dimensional integer array')

        shapes = {}
        for field in FIELDS:
            if field not in field_shapes:
                raise ValueError(f'Missing LP field shape: {field}')
            try:
                raw_shape = tuple(field_shapes[field])
                shape = tuple(operator.index(size) for size in raw_shape)
            except (TypeError, ValueError) as error:
                raise ValueError(f'Invalid LP field shape: {field}') from error
            if any(isinstance(size, (bool, np.bool_)) for size in raw_shape) or any(size < 0 for size in shape):
                raise ValueError(f'Invalid LP field shape: {field}')
            shapes[field] = shape

        widths = tuple(math.prod(shapes[field]) for field in FIELDS)
        width = sum(widths)
        if width > np.iinfo(np.intp).max:
            raise ValueError('LP feature width exceeds supported index range')
        if np.any(indices < 0) or np.any(indices >= width):
            raise ValueError('LP feature index out of range')

        self.field_shapes = MappingProxyType(shapes)
        self.feature_width = width
        self.feature_indices = np.array(indices, dtype=np.int64, copy=True)
        self.feature_indices.setflags(write=False)
        self.selected_width = len(indices)
        self._groups = []
        start = 0
        for field, field_width in zip(FIELDS, widths):
            positions = np.flatnonzero((self.feature_indices >= start) & (self.feature_indices < start + field_width))
            if len(positions):
                local = self.feature_indices[positions] - start
                # Coordinates avoid an O(full LP width) ravel/reshape copy
                # for non-contiguous input views, on both CPU and GPU.
                coordinates = np.unravel_index(local, shapes[field], order='C') if shapes[field] else ()
                positions.setflags(write=False)
                for coordinate in coordinates:
                    coordinate.setflags(write=False)
                self._groups.append((field, positions, coordinates))
            start += field_width
        self._groups = tuple(self._groups)
        self._device_groups = {}

    @classmethod
    def from_inputs(cls, inputs, feature_indices, *, batched=False):
        """Infer per-LP shapes without downloading device data."""
        shapes = {}
        batch = None
        for field in FIELDS:
            if field not in inputs:
                raise ValueError(f'Missing LP field: {field}')
            value = inputs[field]
            shape = tuple(value.shape) if hasattr(value, 'shape') else np.asarray(value).shape
            if batched:
                if not shape:
                    raise ValueError(f'Batched LP field has no batch dimension: {field}')
                if batch is not None and batch != shape[0]:
                    raise ValueError('LP field batch sizes must match')
                batch = shape[0]
                shape = shape[1:]
            shapes[field] = shape
        return cls(feature_indices, shapes)

    def _validate_shape(self, field, shape, *, batched, batch=None):
        if batched:
            if not shape or tuple(shape[1:]) != self.field_shapes[field]:
                raise ValueError(f'Unexpected batched LP field shape: {field}')
            if batch is not None and shape[0] != batch:
                raise ValueError('LP field batch sizes must match')
        elif tuple(shape) != self.field_shapes[field]:
            raise ValueError(f'Unexpected LP field shape: {field}')

    def cpu_features(self, inputs):
        """Return ``[selected_width]`` float32 features using NumPy only."""
        arrays = {}
        for field in FIELDS:
            if field not in inputs:
                raise ValueError(f'Missing LP field: {field}')
            if hasattr(inputs[field], '__cuda_array_interface__'):
                raise TypeError('CPU LP features cannot read a device array')
            value = np.asarray(inputs[field])
            self._validate_shape(field, value.shape, batched=False)
            arrays[field] = value
        # np.concatenate promotes even fields from which no element was
        # selected. Preserve that promotion before applying any arithmetic.
        dtype = np.result_type(*(arrays[field].dtype for field in FIELDS))
        selected = np.empty(self.selected_width, dtype=dtype)
        for field, positions, coordinates in self._groups:
            selected[positions] = arrays[field][coordinates]
        selected = np.nan_to_num(selected, copy=False, nan=0., posinf=1e13, neginf=-1e13)
        return (np.sign(selected) * np.log1p(np.abs(selected))).astype(np.float32)

    def gpu_features(self, inputs):
        """Return ``[batch, selected_width]`` CuPy float32, without downloads."""
        import cupy as cp

        arrays = {}
        batch = device = None
        for field in FIELDS:
            if field not in inputs:
                raise ValueError(f'Missing LP field: {field}')
            value = inputs[field]
            if not isinstance(value, cp.ndarray):
                raise TypeError('GPU LP features require CuPy input arrays')
            self._validate_shape(field, value.shape, batched=True, batch=batch)
            if device is not None and value.device.id != device:
                raise ValueError('GPU LP fields must reside on one device')
            batch, device = value.shape[0], value.device.id
            arrays[field] = value

        with cp.cuda.Device(device):
            groups = self._device_groups.get(device)
            if groups is None:
                groups = tuple((field, cp.asarray(positions), tuple(cp.asarray(index) for index in coordinates))
                    for field, positions, coordinates in self._groups)
                self._device_groups[device] = groups
            dtype = cp.result_type(*(arrays[field].dtype for field in FIELDS))
            selected = cp.empty((batch, self.selected_width), dtype=dtype)
            for field, positions, coordinates in groups:
                values = arrays[field][(slice(None), *coordinates)] if coordinates else arrays[field][:, None]
                selected[:, positions] = values
            selected = cp.nan_to_num(selected, copy=False, nan=0., posinf=1e13, neginf=-1e13)
            return (cp.sign(selected) * cp.log1p(cp.abs(selected))).astype(cp.float32)

