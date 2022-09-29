import numba as nb
import numpy as np
import numpy.typing as npt


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.float64))
def cartesian_to_polar(
    data: npt.NDArray[np.float64], base: float = 2 * np.pi
) -> npt.NDArray[np.float64]:
    """Converts a set of cartesian coordinates to polar coordinates.

    Args:
        data (npt.NDArray[np.float64]): Array of cartesian coordinates.
        base (float): Base of the angular components. Defaults to 2 * np.pi.

    Returns:
        npt.NDArray[np.float64]: Array of polar coordinates.
    """
    base_to_rad = base / (2 * np.pi)
    out = np.empty_like(data)
    for i in range(len(data)):
        x, y = data[i]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) * base_to_rad
        out[i] = (r, theta)
    return out


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.float64))
def polar_to_cartesian(
    data: npt.NDArray[np.float64], base: float = 2 * np.pi
) -> npt.NDArray[np.float64]:
    """Converts a set of polar coordinates to cartesian coordinates.

    Args:
        data (npt.NDArray[np.float64]): Array of polar coordinates.
        base (float): Base of the angular components. Defaults to 2 * np.pi.

    Returns:
        npt.NDArray[np.float64]: Array of cartesian coordinates.
    """
    rad_to_base = (2 * np.pi) / base
    out = np.empty_like(data)
    for i in range(len(data)):
        r = data[i, 0]
        theta = data[i, 1] * rad_to_base
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        out[i] = (x, y)
    return out


@nb.njit(nb.float64[:, :](nb.float64[:], nb.float64[:]))
def project_to_unit_circle(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Projects a set of cartesian coordinates to the unit circle.

    Args:
        x (npt.NDArray[np.float64]): Array of x coordinates.
        y (npt.NDArray[np.float64]): Array of y coordinates.

    Returns:
        npt.NDArray[np.float64]: Array of projected coordinates.
    """
    n = len(x)
    out = np.empty((n, 2))
    for i in range(n):
        r: float = np.sqrt(x[i] ** 2 + y[i] ** 2)
        out[i] = (x[i] / r, y[i] / r)
    return out


@nb.njit(nb.float64[:](nb.float64[:, :]))
def cartesian_mean(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculates the mean of a set of cartesian coordinates.

    Args:
        data (npt.NDArray[np.float64]): Array of cartesian coordinates.

    Returns:
        npt.NDArray[np.float64]: Mean of the data.
    """
    out = np.empty(2)
    projected = project_to_unit_circle(data[:, 0], data[:, 1])
    out[0] = np.mean(projected[:, 0])
    out[1] = np.mean(projected[:, 1])
    return out


@nb.njit(nb.float64(nb.float64[:, :]))
def cartesian_variance(data: npt.NDArray[np.float64]) -> float:
    """Calculates the variance of a set of cartesian coordinates.

    Args:
        data (npt.NDArray[np.float64]): Array of cartesian coordinates.

    Returns:
        float: Variance of the data.
    """
    x_mean, y_mean = cartesian_mean(data)
    r: float = np.sqrt(x_mean**2 + y_mean**2)
    r = min(1, r)
    return 1 - r


@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:]))
def distance_between_angles(
    theta1: npt.NDArray[np.float64], theta2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Calculates the distance between two sets of angles.

    Args:
        theta1 (npt.NDArray[np.float64]): Array of angles.
        theta2 (npt.NDArray[np.float64]): Array of angles.

    Returns:
        npt.NDArray[np.float64]: Array of distances.
    """
    n = len(theta1)
    out = np.empty(n)
    for i in range(n):
        diff = np.abs(theta1[i] - theta2[i])
        if diff > np.pi:
            diff = 2 * np.pi - diff
        out[i] = diff
    return out


@nb.njit(nb.float64(nb.float64[:]))
def angular_mean(theta: npt.NDArray[np.float64]) -> float:
    """Calculates the mean of a set of angles.

    Args:
        theta (npt.NDArray[np.float64]): Array of angles.

    Returns:
        float: Mean of the data.
    """
    x = np.mean(np.cos(theta))
    y = np.mean(np.sin(theta))
    return np.arctan2(y, x)


@nb.njit(nb.complex128(nb.float64[:]))
def angular_mean_var(theta: npt.NDArray[np.float64]) -> complex:
    """Calculates the mean and variance of a set of angles.

    Args:
        theta (npt.NDArray[np.float64]): Array of angles.

    Returns:
        complex: Mean and variance of the data. Mean is the real part, variance
            is the imaginary part.
    """
    x = np.mean(np.cos(theta))
    y = np.mean(np.sin(theta))
    angle = np.arctan2(y, x)
    var = np.sqrt(np.square(x) + np.square(y))
    return angle + 1j * var


@nb.njit(nb.complex128(nb.float64[:]))
def angular_mean_cartesian(theta: npt.NDArray[np.float64]) -> complex:
    """Calculates the mean and variance of a set of angles, in cartesian

    Args:
        theta (npt.NDArray[np.float64]): Array of angles.

    Returns:
        complex: Cartesian mean of projected angles, x+yi. Mean is calculated by
            atan2(y, x). Variance is calculated by sqrt(x ** 2 + y ** 2).
    """
    x = np.mean(np.cos(theta))
    y = np.mean(np.sin(theta))
    return x + 1j * y
