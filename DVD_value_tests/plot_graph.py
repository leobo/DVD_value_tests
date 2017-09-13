import numpy as np
import matplotlib.pyplot as plt


def ind_vector_gen(low, high, size):
    """
    Generate a basis with dimensions size.
    :param low: The lowest value in the basis
    :param high: The highest value in the basis
    :param size: The size of the basis
    :return: The generated basis
    """
    ind = False
    while not ind:
        basis = np.random.randint(low=low, high=high, size=size)
        if np.linalg.det(basis) != 0 and np.linalg.det(basis) < size[0] and np.linalg.det(basis) > -size[0]:
            ind = True
        if sum(basis[0]) == 0 or sum(basis[1]) == 0:
            ind = False
    return basis


def convolve(vec, basis):
    """
    Convolve every column vectors in basis with vector vec, concatenate all results and return.
    :param vector: The vector that need convolve
    :param basis: The basis
    :return: The concatenated convolving results.
    """
    size = np.shape(basis)
    length = np.shape(vec)

    res = []
    for v in basis:
        res = res + list(np.convolve(vec, v, 'valid'))
    return res



if __name__ == '__main__':
    length = 10
    b_size = (10,10)
    low = -2
    high = 2

    v1 = np.random.randint(low=0, high=10, size=length)
    v2 = np.copy(v1)
    v2[3] = v2[3] + 3
    # v2 = np.random.randint(low=0, high=10, size=length)
    basis = ind_vector_gen(low, high, b_size)

    # evenly sampled time at 200ms intervals
    t1 = np.arange(0, length, 1)
    t2 = np.arange(0, length, 1)

    r1 = convolve(v1, basis)
    r2 = convolve(v2, basis)

    r1 = convolve(r1, basis)
    r2 = convolve(r2, basis)

    r1 = convolve(r1, basis)
    r2 = convolve(r2, basis)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(t1, v1, 'ro', t1, v1, 'k', t1, v2, 'bs', t1, v2, 'k')

    # red dashes, blue squares and green triangles
    plt.subplot(212)
    plt.plot(t2, r1, 'ro', t2, r1, 'k', t2, r2, 'bs', t2, r2, 'k')
    plt.show()