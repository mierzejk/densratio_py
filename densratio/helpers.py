# -*- coding: utf-8 -*-
import numpy as np

from numpy import array, matrix, ndarray, result_type
from warnings import warn


np_float = result_type(float)
try:
    import numba as nb
except ModuleNotFoundError:
    guvectorize_compute = None
else:
    _nb_float = nb.from_dtype(np_float)

    def guvectorize_compute(target: str, *, cache: bool = True):
        return nb.guvectorize([nb.void(_nb_float[:, :], _nb_float[:], _nb_float, _nb_float[:])],
                              '(m, p),(p),()->(m)',
                              nopython=True,
                              target=target,
                              cache=cache)


def is_numeric(x):
    return isinstance(x, int) or isinstance(x, float)


def to_numpy_matrix(x):
    if isinstance(x, matrix):
        return x
    elif isinstance(x, ndarray):
        if len(x.shape) == 1:
            return matrix(x).T
        else:
            return matrix(x)
    elif str(type(x)) == "<class 'pandas.core.frame.DataFrame'>":
        return x.as_matrix()
    elif not x:
        raise ValueError("Cannot transform to numpy.matrix.")
    else:
        return to_numpy_matrix(array(x))


def alpha_normalize(values: array, alpha: float) -> array:
    """
    Normalizes values less than 1 so the minimum value to replace 0 is symmetrical to alpha^-1
    with respect to the natural logarithm.

    Arguments:
        values (numpy.array): A vector to normalize.
        alpha (float): The normalization term.

    Returns:
        Normalized numpy.array object that preserves the order and the number of unique input argument values.
    """
    if not alpha:
        return values

    a = 1. - alpha
    last_value = 1.
    inserted = last_value
    outcome = np.empty(values.shape, dtype=values.dtype)

    values_argsort = np.argsort(values)
    for i in np.flip(values_argsort):
        value = values[i]
        if value >= 1.:
            outcome[i] = value
            continue

        if value < last_value:
            new_value = inserted - a * (last_value - value)
            inserted = np.nextafter(inserted, 0) if new_value == inserted else new_value
            last_value = value
        else:
            assert value == last_value

        outcome[i] = inserted

    if not outcome.all():
        warn('Normalized vector contains some zero values.', RuntimeWarning)

    assert np.unique(values).size == np.unique(outcome).size
    assert (values_argsort == np.argsort(outcome)).all()
    return outcome


def semi_stratified_sample(data: ndarray, samples: int) -> ndarray:
    ndims = data.ndim
    if ndims > 2:
        raise ValueError('Only single and 2d arrays are supported.')
    if not samples:
        return np.empty(0)

    data_length = data.shape[0]
    result = np.arange(data_length, dtype=int)
    if samples == data_length:
        np.random.shuffle(result)
        return result
    if samples < 0:
        raise ValueError('Number of samples must be a non-negative integer number.')
    if samples > data_length:
        raise ValueError('Number of samples cannot exceed the shape of input data.')

    dims = data.shape[1] if 2 == ndims else 1
    indexed = np.column_stack((data, result))
    result = np.empty(0, dtype=indexed.dtype)

    samples_no = samples // dims
    if samples_no:
        percentiles = np.linspace(0., 100., num=samples_no, endpoint=False)[1:]

        for d in range(dims):
            column = indexed[..., d]
            quantiles = np.append(column.min(), np.percentile(column, percentiles))
            indices = []
            i, sample_size = 0, 1

            while i < samples_no:
                left = quantiles[i]
                i += 1
                right = np.Inf if i == samples_no else quantiles[i]
                try:
                    indices.extend(np.random.choice(
                        indexed[(left <= column) & (column < right), dims],
                        size=sample_size,
                        replace=False))
                except ValueError:
                    sample_size += 1
                    continue
                else:
                    sample_size = 1

            indexed = indexed[~np.isin(indexed[:, dims], indices)]
            result = np.append(result, indices)

    result = np.append(
        result,
        np.random.choice(indexed[..., dims], size=samples-result.size, replace=False)).astype(int)
    np.random.shuffle(result)
    return result
