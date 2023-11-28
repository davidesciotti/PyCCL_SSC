import bz2
import pdb
import sys
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml
from numba import njit
from scipy.interpolate import interp1d
import scipy
import pickle
import itertools
import os


###############################################################################


def get_zpairs(zbins):
    zpairs_auto = int((zbins * (zbins + 1)) / 2)  # = 55 for zbins = 10, cast it as int
    zpairs_cross = zbins ** 2
    zpairs_3x2pt = 2 * zpairs_auto + zpairs_cross
    return zpairs_auto, zpairs_cross, zpairs_3x2pt


def generate_ind(triu_tril_square, row_col_major, size):
    """
    Generates a list of indices for the upper triangular part of a matrix
    :param triu_tril_square: str. if 'triu', returns the indices for the upper triangular part of the matrix.
    If 'tril', returns the indices for the lower triangular part of the matrix
    If 'full_square', returns the indices for the whole matrix
    :param row_col_major: str. if True, the indices are returned in row-major order; otherwise, in column-major order
    :param size: int. size of the matrix to take the indices of
    :return: list of indices
    """
    assert row_col_major in ['row-major', 'col-major'], 'row_col_major must be either "row-major" or "col-major"'
    assert triu_tril_square in ['triu', 'tril', 'full_square'], 'triu_tril_square must be either "triu", "tril" or ' \
                                                                '"full_square"'

    if triu_tril_square == 'triu':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i, size)]
        elif 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i + 1)]
    elif triu_tril_square == 'tril':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(i + 1)]
        elif 'col-major':
            ind = [(j, i) for i in range(size) for j in range(i, size)]
    elif triu_tril_square == 'full_square':
        if row_col_major == 'row-major':
            ind = [(i, j) for i in range(size) for j in range(size)]
        elif 'col-major':
            ind = [(j, i) for i in range(size) for j in range(size)]

    return np.asarray(ind)


def build_full_ind(triu_tril, row_col_major, size):
    """
    Builds the good old ind file
    """

    assert triu_tril in ['triu', 'tril'], 'triu_tril must be either "triu" or "tril"'
    assert row_col_major in ['row-major', 'col-major'], 'row_col_major must be either "row-major" or "col-major"'

    zpairs_auto, zpairs_cross, zpairs_3x2pt = get_zpairs(size)

    LL_columns = np.zeros((zpairs_auto, 2))
    GL_columns = np.hstack((np.ones((zpairs_cross, 1)), np.zeros((zpairs_cross, 1))))
    GG_columns = np.ones((zpairs_auto, 2))

    LL_columns = np.hstack((LL_columns, generate_ind(triu_tril, row_col_major, size))).astype(int)
    GL_columns = np.hstack((GL_columns, generate_ind('full_square', row_col_major, size))).astype(int)
    GG_columns = np.hstack((GG_columns, generate_ind(triu_tril, row_col_major, size))).astype(int)

    ind = np.vstack((LL_columns, GL_columns, GG_columns))

    assert ind.shape[0] == zpairs_3x2pt, 'ind has the wrong number of rows'

    return ind


@njit
def cov_4D_to_2D(cov_4D, block_index='vincenzo', optimize=True):
    """ new (more elegant) version of cov_4D_to_2D. Also works for 3x2pt. The order
    of the for loops does not affect the result!

    Ordeting convention:
    - whether to use ij or ell as the outermost index (determined by the ordering of the for loops).
      This is going to be the index of the blocks in the 2D covariance matrix.
    Sylvain uses block_index == 'pair_wise', me and Vincenzo block_index == 'ell_wise':
    I add this distinction in the "if" to make it clearer.

    this function can also convert to 2D non-square blocks; this is needed to build the 3x2pt_2D according to CLOE's
    ordering (which is not actually Cloe's ordering...); it is sufficient to pass a zpairs_CD != zpairs_AB value
    (by default zpairs_CD == zpairs_AB). This is not necessary in the above function (unless you want to reshape the
    individual blocks) because also in the 3x2pt case I am reshaping a square matrix (of size [nbl*zpairs, nbl*zpairs])

    Note: this reshaping does not change the number of elements, we just go from [nbl, nbl, zpairs, zpairs] to
    [nbl*zpairs, nbl*zpairs]; hence no symmetrization or other methods to "fill in" the missing elements in the
    higher-dimensional array are needed.
    """

    assert block_index in ['ell', 'vincenzo', 'C-style'] + ['ij', 'sylvain', 'F-style'], \
        'block_index must be "ell", "vincenzo", "C-style" or "ij", "sylvain", "F-style"'

    assert cov_4D.ndim == 4, 'the input covariance must be 4-dimensional'
    assert cov_4D.shape[0] == cov_4D.shape[1], 'the first two axes of the input covariance must have the same size'
    # assert cov_4D.shape[2] == cov_4D.shape[3], 'the second two axes of the input covariance must have the same size'

    nbl = int(cov_4D.shape[0])
    zpairs_AB = int(cov_4D.shape[2])
    zpairs_CD = int(cov_4D.shape[3])

    cov_2D = np.zeros((nbl * zpairs_AB, nbl * zpairs_CD))

    if optimize:
        if block_index in ['ell', 'vincenzo', 'C-style']:
            cov_2D.reshape(nbl, zpairs_AB, nbl, zpairs_CD)[:, :, :, :] = cov_4D.transpose(0, 2, 1, 3)

        elif block_index in ['ij', 'sylvain', 'F-style']:
            cov_2D.reshape(zpairs_AB, nbl, zpairs_CD, nbl)[:, :, :, :] = cov_4D.transpose(2, 0, 3, 1)
        return cov_2D

    if block_index in ['ell', 'vincenzo', 'C-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[l1 * zpairs_AB + ipair, l2 * zpairs_CD + jpair] = cov_4D[l1, l2, ipair, jpair]

    elif block_index in ['ij', 'sylvain', 'F-style']:
        for l1 in range(nbl):
            for l2 in range(nbl):
                for ipair in range(zpairs_AB):
                    for jpair in range(zpairs_CD):
                        # block_index * block_size + running_index
                        cov_2D[ipair * nbl + l1, jpair * nbl + l2] = cov_4D[l1, l2, ipair, jpair]

    return cov_2D


def test_folder_content(output_path, benchmarks_path, extension, verbose=False, rtol=1e-10):
    """Test if the files in the output folder are equal to the benchmark files.

    Parameters:
    output_path (str): The path to the folder containing the output files.
    benchmarks_path (str): The path to the folder containing the benchmark files.
    extension (str): The extension of the files to be tested.

    Returns:
    None.
    """
    old_files = os.listdir(benchmarks_path)
    new_files = os.listdir(output_path)

    if 'benchmarks' in new_files:
        new_files.remove('benchmarks')

    # ignore hidden files
    old_files = [file for file in old_files if not file.startswith('.')]
    new_files = [file for file in new_files if not file.startswith('.')]

    assert old_files, f"No files found in the benchmarks path: {benchmarks_path} ❌"
    assert new_files, f"No files found in the output path: {output_path} ❌"
    assert set(old_files) == set(new_files), f"The number or name of files in the benchmark folder and output " \
                                             f"folder do not match ❌\n" \
                                             f"files which do not match: {set(old_files) ^ set(new_files)}"

    print(f'\n**** testing {extension} files in folder: *****\n{output_path}\n')

    for file_name in old_files:

        if extension in file_name:

            old_file_path = os.path.join(benchmarks_path, file_name)
            new_file_path = os.path.join(output_path, file_name)

            try:
                if extension == 'npz':
                    np.testing.assert_allclose(np.load(old_file_path)['arr_0'], np.load(new_file_path)['arr_0'],
                                               verbose=verbose, rtol=rtol, atol=0)
                elif extension == 'npy':
                    np.testing.assert_allclose(np.load(old_file_path), np.load(new_file_path), verbose=verbose,
                                               rtol=rtol,
                                               atol=0)
                elif extension == 'txt' or extension == 'dat':
                    np.testing.assert_allclose(np.genfromtxt(old_file_path), np.genfromtxt(new_file_path),
                                               verbose=verbose, rtol=rtol, atol=0)
                else:
                    raise ValueError(f"Unknown extension: {extension}")
            except AssertionError as exc:
                print(f'\nFile {file_name} ❌:', exc)
            else:
                print(f'\nThe file {file_name} is equal to the benchmarks ✅')

    return None

# ! ====================================================================================================================
