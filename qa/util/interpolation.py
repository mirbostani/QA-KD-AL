from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

def interpolation(y, n, kind='linear', plot=False):
    """Interpolate y vector to n length.
    Args:
    y: a Python list
    n: length of the interpolated vector y
    kind: e.g. 'linear', 'cubic', 'quadratic', etc.
    """
    y_np = np.array(y, dtype=np.float32)
    x_np = np.linspace(0, len(y_np), num=len(y_np))

    fn = interpolate.interp1d(x_np, y_np, kind=kind)

    xn_np = np.linspace(0, len(y_np), num=n)
    yn_np = fn(xn_np)

    if plot:
        plt.plot(x_np, y_np, 'o', xn_np, yn_np, '-')
        plt.legend(['data', kind], loc='best')
        plt.show()

    return yn_np.tolist()