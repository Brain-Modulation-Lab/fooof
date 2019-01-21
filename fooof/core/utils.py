"""Internal utility functions for FOOOF."""

import numpy as np

###################################################################################################
###################################################################################################

def group_three(vec):
    """Takes array of inputs, groups by three.

    Parameters
    ----------
    vec : 1d array
        Array of items to sort by 3 - must be divisible by three.

    Returns
    -------
    list of list
        List of lists, each with three items.
    """

    if len(vec) % 3 != 0:
        raise ValueError('Wrong size array to group by three.')

    return [list(vec[i:i+3]) for i in range(0, len(vec), 3)]


def dict_array_to_lst(in_dict):
    """Convert any numpy arrays present in a dictionary to be lists.

    Parameters
    ----------
    in_dict : dict
        Input dictionary.

    Returns
    -------
    dict
        Output dictionary with all arrays converted to lists.
    """

    return {ke: va.tolist() if isinstance(va, np.ndarray) else va for ke, va in in_dict.items()}


def dict_lst_to_array(in_dict, mk_array):
    """Convert specified lists in a dictionary to be arrays.

    Parameters
    ----------
    in_dict : dict
        Input dictionary.
    mk_array : list of str
        Keys to convert to arrays in the dictionary.

    Returns
    -------
    dict
        Output dictionary with specified lists converted to arrays.
    """

    return {ke: np.array(va) if ke in mk_array else va for ke, va in in_dict.items()}


def dict_select_keys(in_dict, keep):
    """Restrict a dictionary to only keep specified keys.

    Parameters
    ----------
    in_dict : dict
        Input dictionary.
    keep : list or set
        Keys to retain in the dictionary.

    Returns
    -------
    dict
        Output dictionary containing only keys specified in keep.
    """

    return {ke:va for ke, va in in_dict.items() if ke in keep}


def check_array_dim(arr):
    """Check that parameter array has 2D shape, and reshape if not.

    Parameters
    ----------
    arr : np.array
        Array to check.

    Returns
    -------
    np.array
        Original array, if 2D, or 2D empty array.
    """

    return np.empty([0, 3]) if arr.ndim == 1 else arr


def get_obj_desc():
    """Get dictionary specifying FOOOF object names and kind of attributes.

    Returns
    -------
    attibutes : dict
        Mapping of FOOOF object attributes, and what kind of data they are.
    """

    attributes = {'results' : ['background_params_', 'peak_params_', 'error_',
                               'r_squared_', '_gaussian_params'],
                  'settings' : ['peak_width_limits', 'max_n_peaks', 'min_peak_amplitude',
                                'peak_threshold', 'background_mode'],
                  'data' : ['power_spectrum', 'freq_range', 'freq_res'],
                  'freq_info' : ['freq_range', 'freq_res'],
                  'arrays' : ['freqs', 'power_spectrum', 'background_params_',
                              'peak_params_', '_gaussian_params']}

    return attributes


def get_data_indices(background_mode):
    """Get a dictionary mapping the column labels to indices in FOOOF data (FOOOFResults).

    Parameters
    ----------
    background_mode : {'fixed', 'knee'}
        Which approach taken to fit the background.

    Returns
    -------
    indices : dict
        Mapping for data columns to the column indices in which they appear.
    """

    indices = {
        'CF'  : 0,
        'Amp' : 1,
        'BW'  : 2,
        'intercept' : 0,
        'knee'      : 1 if background_mode == 'knee' else None,
        'slope'     : 1 if background_mode == 'fixed' else 2
    }

    return indices
