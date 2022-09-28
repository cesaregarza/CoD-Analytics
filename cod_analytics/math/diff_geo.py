import numba as nb
import numpy as np
import numpy.typing as npt


@nb.njit(nb.float64(nb.float64[:], nb.float64[:]))
def wedge(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Computes the wedge product of two two-dimensional vectors.

    Args:
        a (npt.NDArray[np.float64]): First vector.
        b (npt.NDArray[np.float64]): Second vector.

    Returns:
        float: The wedge product of the two vectors.
    """
    return a[0] * b[1] - a[1] * b[0]


@nb.njit(nb.float64[:](nb.float64[:, :], nb.float64[:, :]))
def wedge_many(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Computes the wedge product of many two-dimensional vectors and returns
    the result as a one-dimensional array.

    Args:
        a (npt.NDArray[np.float64]): First list of vectors.
        b (npt.NDArray[np.float64]): Second list of vectors.

    Returns:
        npt.NDArray[np.float64]: A list of wedge products of the two vectors.
    """
    out = np.empty(len(a))
    for i in range(len(a)):
        out[i] = a[i, 0] * b[i, 1] - a[i, 1] * b[i, 0]
    return out
