"""FOOOF Object - base object which defines the model.

Private Attributes
==================
Private attributes of the FOOOF object are documented here.

Data Attributes
---------------
_spectrum_flat : 1d array
    Flattened power spectrum, with the aperiodic component removed.
_spectrum_peak_rm : 1d array
    Power spectrum, with peaks removed.

Model Component Attributes
--------------------------
_ap_fit : 1d array
    Values of the isolated aperiodic fit.
_peak_fit : 1d array
    Values of the isolated peak fit.

Internal Settings Attributes
----------------------------
_ap_percentile_thresh : float
    Percentile threshold for finding peaks above the aperiodic component.
_ap_guess : list of [float, float, float]
    Guess parameters for fitting the aperiodic component.
_ap_bounds : tuple of tuple of float
    Upper and lower bounds on fitting aperiodic component.
_cf_bound : float
    Parameter bounds for center frequency when fitting gaussians.
_gauss_overlap_thresh : float
    Degree of overlap (in units of standard deviation) between gaussian guesses to drop one.
_gauss_std_limits : list of [float, float]
    Peak width limits, converted to use for gaussian standard deviation parameter.
    This attribute is computed based on `peak_width_limits` and should not be updated directly.
_maxfev : int
    The maximum number of calls to the curve fitting function.
_error_metric : str
    The error metric to use for post-hoc measures of model fit error.

Run Modes
---------
_debug : bool
    Whether the object is set in debug mode.
    This should be controlled by using the `set_debug_mode` method.
_check_data : bool
    Whether to check added data for NaN or Inf values, and fail out if present.
    This should be controlled by using the `set_check_data_mode` method.

Code Notes
----------
Methods without defined docstrings import docs at runtime, from aliased external functions.
"""

import warnings
from copy import deepcopy

import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import curve_fit, minimize
import scipy.stats

from fooof.core.items import OBJ_DESC
from fooof.core.info import get_indices
from fooof.core.io import save_fm, load_json
from fooof.core.reports import save_report_fm
from fooof.core.modutils import copy_doc_func_to_method
from fooof.core.utils import group_three, check_array_dim
from fooof.core.funcs import gaussian_function, get_ap_func, lorentzian_function, lorentzian_noise_floor_function, fixed_noise_floor_function, infer_ap_func
from fooof.core.errors import (FitError, NoModelError, DataError,
                               NoDataError, InconsistentDataError)
from fooof.core.strings import (gen_settings_str, gen_results_fm_str,
                                gen_issue_str, gen_width_warning_str)

from fooof.plts.fm import plot_fm
from fooof.utils.data import trim_spectrum
from fooof.utils.params import compute_gauss_std
from fooof.data import FOOOFResults, FOOOFSettings, FOOOFMetaData
from fooof.data.conversions import model_to_dataframe
from fooof.sim.gen import gen_freqs, gen_aperiodic, gen_periodic, gen_model

###################################################################################################
###################################################################################################

def get_negative_AUC(params, fmin):
    params_reshape = params.reshape(int(len(params)/3), 3)
    AUC = 0
    num_peaks = params_reshape.shape[0]
    for i in range(num_peaks): 
        mu, height, sigma = params_reshape[i, 0], params_reshape[i, 1], params_reshape[i, 2]
        AUC += height * scipy.stats.norm(mu, sigma).cdf(fmin)

    return AUC

def calc_error(metric, power_spectrum,fooofed_spectrum_):
    """Calculate the overall error of the model fit, compared to the original data.

    Parameters
    ----------
    metric : {'MAE', 'MSE', 'RMSE'}, optional
        Which error measure to calculate:
        * 'MAE' : mean absolute error
        * 'MSE' : mean squared error
        * 'RMSE' : root mean squared error

    Raises
    ------
    ValueError
        If the requested error metric is not understood.

    Notes
    -----
    Which measure is applied is by default controlled by the `_error_metric` attribute.
    """


    if metric == 'MAE':
        error_ = np.abs(power_spectrum - fooofed_spectrum_).mean()

    elif metric == 'MSE':
        error_ = ((power_spectrum - fooofed_spectrum_) ** 2).mean()

    elif metric == 'RMSE':
        error_ = np.sqrt(((power_spectrum - fooofed_spectrum_) ** 2).mean())

    else:
        error_msg = "Error metric '{}' not understood or not implemented.".format(metric)
        raise ValueError(error_msg)
    return error_


def calc_r_squared(power_spectrum, fooofed_spectrum_):
    
    r_val = np.corrcoef(power_spectrum, fooofed_spectrum_)
    return r_val[0][1] ** 2

def calc_loglikelihood(fooofed_spectrum_, error):
    loglik = -len(fooofed_spectrum_)/2.*(1+np.log(error)+np.log(2*np.pi))
    return loglik

def calc_AIC(params,loglik):
    AIC = 2.*(len(params)-loglik)
    return AIC

def calc_BIC(params,fooofed_spectrum_, loglik):
    BIC = len(params)*np.log(len(fooofed_spectrum_))-2.*loglik
    return BIC

def calc_BF(BIC, BIC_H0):
    BF =  np.exp((BIC-BIC_H0)/2)
    return BF




class FOOOF():
    """Model a physiological power spectrum as a combination of aperiodic and periodic components.

    WARNING: FOOOF expects frequency and power values in linear space.

    Passing in logged frequencies and/or power spectra is not detected,
    and will silently produce incorrect results.

    Parameters
    ----------
    peak_width_limits : tuple of (float, float), optional, default: (0.5, 12.0)
        Limits on possible peak width, in Hz, as (lower_bound, upper_bound).
    max_n_peaks : int, optional, default: inf
        Maximum number of peaks to fit.
    min_peak_height : float, optional, default: 0
        Absolute threshold for detecting peaks.
        This threshold is defined in absolute units of the power spectrum (log power).
    peak_threshold : float, optional, default: 2.0
        Relative threshold for detecting peaks.
        This threshold is defined in relative units of the power spectrum (standard deviation).
    aperiodic_mode : {'fixed', 'knee', 'lorentzian', 'lorentzian-noise-floor', 'fixed-noise-floor'}
        Which approach to take for fitting the aperiodic component.,
    verbose : bool, optional, default: True
        Verbosity mode. If True, prints out warnings and general status updates.

    Attributes
    ----------
    freqs : 1d array
        Frequency values for the power spectrum.
    power_spectrum : 1d array
        Power values, stored internally in log10 scale.
    freq_range : list of [float, float]
        Frequency range of the power spectrum, as [lowest_freq, highest_freq].
    freq_res : float
        Frequency resolution of the power spectrum.
    fooofed_spectrum_ : 1d array
        The full model fit of the power spectrum, in log10 scale.
    aperiodic_params_ : 1d array
        Parameters that define the aperiodic fit. As [Offset, (Knee), Exponent].
        The knee parameter is only included if aperiodic component is fit with a knee.
    peak_params_ : 2d array
        Fitted parameter values for the peaks. Each row is a peak, as [CF, PW, BW].
    gaussian_params_ : 2d array
        Parameters that define the gaussian fit(s).
        Each row is a gaussian, as [mean, height, standard deviation].
    r_squared_ : float
        R-squared of the fit between the input power spectrum and the full model fit.
    error_ : float
        Error of the full model fit.
    n_peaks_ : int
        The number of peaks fit in the model.
    has_data : bool
        Whether data is loaded to the object.
    has_model : bool
        Whether model results are available in the object.

    Notes
    -----
    - Commonly used abbreviations used in this module include:
      CF: center frequency, PW: power, BW: Bandwidth, AP: aperiodic
    - Input power spectra must be provided in linear scale.
      Internally they are stored in log10 scale, as this is what the model operates upon.
    - Input power spectra should be smooth, as overly noisy power spectra may lead to bad fits.
      For example, raw FFT inputs are not appropriate. Where possible and appropriate, use
      longer time segments for power spectrum calculation to get smoother power spectra,
      as this will give better model fits.
    - The gaussian params are those that define the gaussian of the fit, where as the peak
      params are a modified version, in which the CF of the peak is the mean of the gaussian,
      the PW of the peak is the height of the gaussian over and above the aperiodic component,
      and the BW of the peak, is 2*std of the gaussian (as 'two sided' bandwidth).
    """
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, peak_width_limits=(0.5, 12.0), max_n_peaks=np.inf, min_peak_height=0.0,
                 peak_threshold=2.0, aperiodic_mode='fixed', regularization_weight = 0, bic_opt = False, verbose=True):
        """Initialize object with desired settings."""

        # Set input settings
        self.peak_width_limits = peak_width_limits
        self.max_n_peaks = max_n_peaks
        self.min_peak_height = min_peak_height
        self.peak_threshold = peak_threshold
        self.aperiodic_mode = aperiodic_mode
        self.verbose = verbose
        ## PRIVATE SETTINGS
        # Percentile threshold, to select points from a flat spectrum for an initial aperiodic fit
        #   Points are selected at a low percentile value to restrict to non-peak points
        self._ap_percentile_thresh = 0.025
        # Guess parameters for aperiodic fitting, [offset, knee, exponent]
        #   If offset guess is None, the first value of the power spectrum is used as offset guess
        #   If exponent guess is None, the abs(log-log slope) of first & last points is used
        if aperiodic_mode in ['fixed', 'knee']:
            self._ap_guess = (None, 0, None)
        elif aperiodic_mode == 'lorentzian':
            self._ap_guess = (None, 1, None)
        elif aperiodic_mode == 'lorentzian-noise-floor':
            self._ap_guess = (None, 1, None, None)
        elif aperiodic_mode == 'fixed-noise-floor':
            self._ap_guess = (None, None, None)
        # Bounds for aperiodic fitting, as: ((offset_low_bound, knee_low_bound, exp_low_bound),
        #                                    (offset_high_bound, knee_high_bound, exp_high_bound))
        # By default, aperiodic fitting is unbound, but can be restricted here, if desired
        #   Even if fitting without knee, leave bounds for knee (they are dropped later)
        if aperiodic_mode in ['fixed', 'knee']:
            self._ap_bounds = ((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf))
        #   Different format because these bounds will go to minimize rather than curvefit
        elif aperiodic_mode == 'lorentzian':
            self._ap_bounds =  ((None, None), (-1, 3), (None, None))
        elif aperiodic_mode == 'lorentzian-noise-floor':
            self._ap_bounds =  ((None, None), (-1, 3), (None, None), (None, None))
        elif aperiodic_mode == 'fixed-noise-floor':
            self._ap_bounds =  ((None, None), (None, None), (None, None))
        # Threshold for how far a peak has to be from edge to keep.
        #   This is defined in units of gaussian standard deviation
        self.bw_std_edge = 1.0
        # Degree of overlap between gaussians for one to be dropped
        #   This is defined in units of gaussian standard deviation
        self._gauss_overlap_thresh = 0.75
        # Parameter bounds for center frequency when fitting gaussians, in terms of +/- std dev
        self._cf_bound = 1.5
        # Lowest captured frequency in spectrum in Hz - Used as upper bound when integrating for negative AUC for regularization
        self.fmin = 1
        # The maximum number of calls to the curve fitting function
        self._maxfev = 5000
        # The error metric to calculate, post model fitting. See `_calc_error` for options
        #   Note: this is for checking error post fitting, not an objective function for fitting
        self._error_metric = 'MAE'
        # Regularization if Lorentzian Aperiodic Mode is used. Default - not implemented since regulatization = 0
        self.regularization_weight = regularization_weight

        ## RUN MODES
        # Set default debug mode - controls if an error is raised if model fitting is unsuccessful
        self._debug = False
        # Set default data checking modes - controls which checks get run on input data
        #   check_freqs: check the frequency values, and raises an error for uneven spacing
        self._check_freqs = False
        #   check_data: checks the power values and raises an error for any NaN / Inf values
        self._check_data = True
        # check whether optimize number of components in the module is enabled
        self._bic_opt = bic_opt
        if self._bic_opt and self.aperiodic_mode in ['fixed', 'knee']:
            error_msg = "BIC optimizatio not  implemented for '{}' model.".format(self.aperiodic_mode)
            raise NotImplementedError(error_msg)            

        # Set internal settings, based on inputs, and initialize data & results attributes
        self._reset_internal_settings()
        self._reset_data_results(True, True, True)


    @property
    def has_data(self):
        """Indicator for if the object contains data."""

        return True if np.any(self.power_spectrum) else False


    @property
    def has_model(self):
        """Indicator for if the object contains a model fit.

        Notes
        -----
        This check uses the aperiodic params, which are:

        - nan if no model has been fit
        - necessarily defined, as floats, if model has been fit
        """

        return True if not np.all(np.isnan(self.aperiodic_params_)) else False


    @property
    def n_peaks_(self):
        """How many peaks were fit in the model."""

        return self.peak_params_.shape[0] if self.has_model else None


    def _reset_internal_settings(self):
        """Set, or reset, internal settings, based on what is provided in init.

        Notes
        -----
        These settings are for internal use, based on what is provided to, or set in `__init__`.
        They should not be altered by the user.
        """

        # Only update these settings if other relevant settings are available
        if self.peak_width_limits:

            # Bandwidth limits are given in 2-sided peak bandwidth
            #   Convert to gaussian std parameter limits
            self._gauss_std_limits = tuple(bwl / 2 for bwl in self.peak_width_limits)

        # Otherwise, assume settings are unknown (have been cleared) and set to None
        else:
            self._gauss_std_limits = None


    def _reset_data_results(self, clear_freqs=False, clear_spectrum=False, clear_results=False):
        """Set, or reset, data & results attributes to empty.

        Parameters
        ----------
        clear_freqs : bool, optional, default: False
            Whether to clear frequency attributes.
        clear_spectrum : bool, optional, default: False
            Whether to clear power spectrum attribute.
        clear_results : bool, optional, default: False
            Whether to clear model results attributes.
        """

        if clear_freqs:
            self.freqs = None
            self.freq_range = None
            self.freq_res = None

        if clear_spectrum:
            self.power_spectrum = None

        if clear_results:

            if self.aperiodic_mode == 'fixed':
                self.aperiodic_params_ = np.array([np.nan] * 2)
            elif self.aperiodic_mode == 'lorentzian-noise-floor':
                self.aperiodic_params_ = np.array([np.nan] * 4)
            else:
                self.aperiodic_params_ = np.array([np.nan] * 3)
            
            self.gaussian_params_ = np.empty([0, 3])
            self.peak_params_ = np.empty([0, 3])
            self.r_squared_ = np.nan
            self.error_ = np.nan

            self.fooofed_spectrum_ = None

            self._spectrum_flat = None
            self._spectrum_peak_rm = None
            self._ap_fit = None
            self._peak_fit = None
    
    def negative_AUC(self):
        AUC = 0
        num_peaks = self.gaussian_params_.shape[0]
        for i in range(num_peaks): 
            mu, height, sigma = self.gaussian_params_[i, 0], self.gaussian_params_[i, 1], self.gaussian_params_[i, 2]
            AUC += height * scipy.stats.norm(mu, sigma).cdf(self.fmin)
        
        return AUC
    

  
    def model_regression(self, func, X, y, guess, bounds, reg_weight = 1):
        """
        args:
            func: aperiodic model
            X: frequencies vector (1D vector)
            y: Observed power spectra (1D vector)
            guess: initial guess of parameters (1d, length 3)
            reg_weight: weight of regularization term - negative AUC

        return: trained parameters
        """
        print(guess)
        print(bounds)
        if self.aperiodic_mode == 'lorentzian-noise-floor':
            def cost(params, reg_weight = 1):  # simply use globally defined x and y
                ap_params = params[:4]
                per_params = params[4:]
                offset, knee, exp, noise_floor = ap_params
                ap_model = func(X, offset, knee, exp, noise_floor)
                per_model = gaussian_function(X, *per_params)
                model = ap_model + per_model
                return np.mean((model - y)**2) 
            
        elif self.aperiodic_mode == 'fixed-noise-floor':
            def cost(params, reg_weight = 1):  # simply use globally defined x and y
                ap_params = params[:3]
                per_params = params[3:]
                offset, exp, noise_floor = ap_params
                ap_model = func(X, offset, exp, noise_floor)
                per_model = gaussian_function(X, *per_params)
                model = ap_model + per_model
                return np.mean((model - y)**2) 
        else:
            def cost(params, reg_weight = 1):  # simply use globally defined x and y
                ap_params = params[:3]
                per_params = params[3:]    
       
                offset, knee, exp = ap_params
                ap_model = func(X, offset, knee, exp)
                per_model = gaussian_function(X, *per_params)
                model = ap_model + per_model
            
                return np.mean((model - y)**2) 

        res = minimize(cost, guess, reg_weight, bounds = bounds, 
                                                options = {'maxiter': self._maxfev})                      

        return res.x
    
    def aperiodic_regression(self, func, X, y, guess, reg_weight = 1):
        """
        args:
            func: aperiodic model
            X: frequencies vector (1D vector)
            y: Observed power spectra (1D vector)
            guess: initial guess of parameters (1d, length 3)
            reg_weight: weight of regularization term - negative AUC

        return: trained parameters
        """
        if self.aperiodic_mode == 'lorentzian-noise-floor':
            def cost(ap_params, reg_weight = 1):  # simply use globally defined x and y
                offset, knee, exp, noise_floor = ap_params
                model = func(X, offset, knee, exp, noise_floor)
                return np.mean((model - y)**2) 
        elif self.aperiodic_mode == 'fixed-noise-floor':
            def cost(ap_params, reg_weight = 1):  # simply use globally defined x and y
                offset, exp, noise_floor = ap_params
                model = func(X, offset, exp, noise_floor)
                return np.mean((model - y)**2) 
        else:
            def cost(ap_params, reg_weight = 1):  # simply use globally defined x and y
                offset, knee, exp = ap_params
                model = func(X, offset, knee, exp)
                return np.mean((model - y)**2) 

        res = minimize(cost, guess, reg_weight, bounds = self._ap_bounds, 
                                                options = {'maxiter': self._maxfev})                      

        return res.x
    
    def gaussian_regression(self, X, y, params, reg_weight = 1, bounds = None):
        """
        args:
            X: frequencies vector (1D vector)
            y: Observed power spectra (1D vector)
            guess: initial guess of parameters (1d, of variable length, multiple of 3)
            reg_weight: weight of regularization term - negative AUC

        return: trained parameters
        """

        def cost(params, reg_weight = 1):  # simply use globally defined x and y
            model = gaussian_function(X, *params)
            return np.mean((model - y)**2) + reg_weight * get_negative_AUC(params, self.fmin) 
            #return np.mean((model - y)**2) + reg_weight * self.negative_AUC() 
        
        res = minimize(cost, x0 = params, args = reg_weight, bounds = bounds, 
                                                options = {'maxiter': self._maxfev})

        return res.x
    
        
        

    def add_data(self, freqs, power_spectrum, freq_range=None, clear_results=True):
        """Add data (frequencies, and power spectrum values) to the current object.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power spectrum, in linear space.
        power_spectrum : 1d array
            Power spectrum values, which must be input in linear space.
        freq_range : list of [float, float], optional
            Frequency range to restrict power spectrum to.
            If not provided, keeps the entire range.
        clear_results : bool, optional, default: True
            Whether to clear prior results, if any are present in the object.
            This should only be set to False if data for the current results are being re-added.

        Notes
        -----
        If called on an object with existing data and/or results
        they will be cleared by this method call.
        """

        # If any data is already present, then clear previous data
        # Also clear results, if present, unless indicated not to
        #   This is to ensure object consistency of all data & results
        self._reset_data_results(clear_freqs=self.has_data,
                                 clear_spectrum=self.has_data,
                                 clear_results=self.has_model and clear_results)

        self.freqs, self.power_spectrum, self.freq_range, self.freq_res = \
            self._prepare_data(freqs, power_spectrum, freq_range, 1)


    def add_settings(self, fooof_settings):
        """Add settings into object from a FOOOFSettings object.

        Parameters
        ----------
        fooof_settings : FOOOFSettings
            A data object containing the settings for a FOOOF model.
        """

        for setting in OBJ_DESC['settings']:
            setattr(self, setting, getattr(fooof_settings, setting))

        self._check_loaded_settings(fooof_settings._asdict())


    def add_meta_data(self, fooof_meta_data):
        """Add data information into object from a FOOOFMetaData object.

        Parameters
        ----------
        fooof_meta_data : FOOOFMetaData
            A meta data object containing meta data information.
        """

        for meta_dat in OBJ_DESC['meta_data']:
            setattr(self, meta_dat, getattr(fooof_meta_data, meta_dat))

        self._regenerate_freqs()


    def add_results(self, fooof_result):
        """Add results data into object from a FOOOFResults object.

        Parameters
        ----------
        fooof_result : FOOOFResults
            A data object containing the results from fitting a FOOOF model.
        """

        self.aperiodic_params_ = fooof_result.aperiodic_params
        self.gaussian_params_ = fooof_result.gaussian_params
        self.peak_params_ = fooof_result.peak_params
        self.r_squared_ = fooof_result.r_squared
        self.error_ = fooof_result.error

        self._check_loaded_results(fooof_result._asdict())


    def report(self, freqs=None, power_spectrum=None, freq_range=None,
               plt_log=False, **plot_kwargs):
        """Run model fit, and display a report, which includes a plot, and printed results.

        Parameters
        ----------
        freqs : 1d array, optional
            Frequency values for the power spectrum.
        power_spectrum : 1d array, optional
            Power values, which must be input in linear space.
        freq_range : list of [float, float], optional
            Desired frequency range to fit the model to.
            If not provided, fits across the entire given range.
        plt_log : bool, optional, default: False
            Whether or not to plot the frequency axis in log space.
        **plot_kwargs
            Keyword arguments to pass into the plot method.

        Notes
        -----
        Data is optional, if data has already been added to the object.
        """
        self.fit(freqs, power_spectrum, freq_range)
        self.plot(plt_log=plt_log, **plot_kwargs)
        self.print_results(concise=False)


    def fit(self, freqs=None, power_spectrum=None, freq_range=None):
        """Fit the full power spectrum as a combination of periodic and aperiodic components.

        Parameters
        ----------
        freqs : 1d array, optional
            Frequency values for the power spectrum, in linear space.
        power_spectrum : 1d array, optional
            Power values, which must be input in linear space.
        freq_range : list of [float, float], optional
            Frequency range to restrict power spectrum to. If not provided, keeps the entire range.

        Raises
        ------
        NoDataError
            If no data is available to fit.
        FitError
            If model fitting fails to fit. Only raised in debug mode.

        Notes
        -----
        Data is optional, if data has already been added to the object.
        """

        # If freqs & power_spectrum provided together, add data to object.
        if freqs is not None and power_spectrum is not None:
            self.add_data(freqs, power_spectrum, freq_range)
        # If power spectrum provided alone, add to object, and use existing frequency data
        #   Note: be careful passing in power_spectrum data like this:
        #     It assumes the power_spectrum is already logged, with correct freq_range
        elif isinstance(power_spectrum, np.ndarray):
            self.power_spectrum = power_spectrum

        # Check that data is available
        if not self.has_data:
            raise NoDataError("No data available to fit, can not proceed.")

        # Check and warn about width limits (if in verbose mode)
        if self.verbose:
            self._check_width_limits()

        # In rare cases, the model fails to fit, and so uses try / except
        try:

            # If not set to fail on NaN or Inf data at add time, check data here
            #   This serves as a catch all for curve_fits which will fail given NaN or Inf
            #   Because FitError's are by default caught, this allows fitting to continue
            if not self._check_data:
                if np.any(np.isinf(self.power_spectrum)) or np.any(np.isnan(self.power_spectrum)):
                    raise FitError("Model fitting was skipped because there are NaN or Inf "
                                   "values in the data, which preclude model fitting.")

            # Fit the aperiodic component
            self.aperiodic_params_ = self._robust_ap_fit(self.freqs, self.power_spectrum)
            self._ap_fit = gen_aperiodic(self.freqs, self.aperiodic_params_, self.aperiodic_mode)

            # Flatten the power spectrum using fit aperiodic fit
            self._spectrum_flat = self.power_spectrum - self._ap_fit
            
            if not self._bic_opt:
                # Find peaks, and fit them with gaussians
                self.gaussian_params_ = self._fit_peaks(np.copy(self._spectrum_flat))
                            # Calculate the peak fit
                #   Note: if no peaks are found, this creates a flat (all zero) peak fit
                self._peak_fit = gen_periodic(self.freqs, np.ndarray.flatten(self.gaussian_params_))

                # Create peak-removed (but not flattened) power spectrum
                self._spectrum_peak_rm = self.power_spectrum - self._peak_fit
                # Run final aperiodic fit on peak-removed power spectrum
                #   This overwrites previous aperiodic fit, and recomputes the flattened spectrum
                self.aperiodic_params_ = self._simple_ap_fit(self.freqs, self._spectrum_peak_rm)
                
                
                
                self._ap_fit = gen_aperiodic(self.freqs, self.aperiodic_params_, self.aperiodic_mode)
                self._spectrum_flat = self.power_spectrum - self._ap_fit

                # Create full power_spectrum model fit
                self.fooofed_spectrum_ = self._peak_fit + self._ap_fit

                # Convert gaussian definitions to peak parameters
                self.peak_params_ = self._create_peak_params(self.gaussian_params_)

                # Calculate R^2 and error of the model fit
                self._calc_r_squared()
                self._calc_error()
                
            else:
                self.models = list()
                # Find peaks, and fit them with gaussians using BIC optimization
                guess = self._est_peaks(np.copy(self._spectrum_flat))
                for k in range(len(guess) + 1):
                    print(f"Running model with {k} peaks..")

                    gaussian_params_ = self._est_fit(guess[:k])
                    # fit with current parameters
                    _peak_fit = gen_periodic(self.freqs, np.ndarray.flatten(self.gaussian_params_))
                    # Create peak-removed (but not flattened) power spectrum
                    _spectrum_peak_rm = self.power_spectrum - _peak_fit
                    # This overwrites previous aperiodic fit, and recomputes the flattened spectrum
                    aperiodic_params_ = self._simple_ap_fit(self.freqs, _spectrum_peak_rm)
                    # refit whole model sarting from self.gaussian_params_[:k] and self.aperiodic_params
                    aperiodic_params_, gaussian_params_ = self._fit_model(gaussian_params_[:k],aperiodic_params_)
 

                    # Create full power_spectrum model fit
                    _ap_fit = gen_aperiodic(self.freqs, aperiodic_params_, self.aperiodic_mode)
                    _spectrum_flat = self.power_spectrum - _ap_fit
                    _peak_fit = gen_periodic(self.freqs, np.ndarray.flatten(gaussian_params_))
                    fooofed_spectrum_ = _peak_fit + _ap_fit
                    
                    # general model spectrum [alternative]
                    #[1]
                    #ap_fit = gen_aperiodic(self.freqs, aperiodic_params_, self.aperiodic_mode)
                    #pe_fit = gen_periodic(self.freqs, np.ndarray.flatten(gaussian_params_))
                    #full_model = pe_fit + ap_fit
    
                    #[2]                
                    #fooofed_spectrum_2, _peak_fit, _ap_fit= gen_model(self.freqs, aperiodic_params_, self.aperiodic_mode,gaussian_params_ ,return_components=True)

                    # generate metrics for each choice of periodic parameters
                    error = calc_error(self._error_metric, self.power_spectrum,fooofed_spectrum_)
                    r_squared = calc_r_squared(self.power_spectrum,fooofed_spectrum_)
                    
                    loglik = calc_loglikelihood(fooofed_spectrum_,error)
                    AIC = calc_AIC(np.concatenate([aperiodic_params_,np.ravel(gaussian_params_)]), loglik)
                    BIC = calc_BIC(np.concatenate([aperiodic_params_,np.ravel(gaussian_params_)]), fooofed_spectrum_, loglik)
                    BF = 0 if k == 0 else calc_BF(BIC, self.models[0]['BIC'])
                    
                    print('error')
                    print(error)

                    
                    
                    self.models.append(
                        {
                        'peak_fit':_peak_fit,
                        'spectrum_peak_rm':_spectrum_peak_rm,
                        'spectrum_flat':_spectrum_flat,
                        'fooofed_spectrum':fooofed_spectrum_,
                        'aperiodic_params':aperiodic_params_,
                        'gauss_params':gaussian_params_,
                        'ap_fit': _ap_fit,
                        'error': error,
                        'r_squared': r_squared,
                        'loglik':loglik,
                        'AIC':AIC,
                        'BIC':BIC,   
                        'BF':BF                     
                        }   
                    )
                    
                print("Identify best model according to BIC:")
                self.get_best_model_by_BIC()
                #print(best_model)

        except FitError:

            # If in debug mode, re-raise the error
            if self._debug:
                raise

            # Clear any interim model results that may have run
            #   Partial model results shouldn't be interpreted in light of overall failure
            self._reset_data_results(clear_results=True)

            # Print out status
            if self.verbose:
                print("Model fitting was unsuccessful.")


    def get_best_model_by_BIC(self):
        """
        Prints the BIC of all models and returns the best model according to the Bayesian Information Criterion (BIC).

        Returns:
        --------
        dict
            The model from self.models with the lowest BIC value.
        """
        if not self.models:
            raise ValueError("No models available to select from.")
        
        # Print BIC for all models
        print("BIC values for all models:")
        for i, model in enumerate(self.models):
            print(f"Model {i+1}: BIC = {model['BIC']} [BF = {model['BF']}]")
        
        # Find the model with the minimum BIC
        best_model = min(self.models, key=lambda model: model['BIC'])
        print(f"\nBest model has BIC = {best_model['BIC']} [BF = {best_model['BF']}]")
        
        print(best_model)
        # populate best model
        self._peak_fit =best_model['peak_fit']
        self._spectrum_peak_rm = best_model['spectrum_peak_rm']
        self._spectrum_flat = best_model['spectrum_flat']
        self.fooofed_spectrum_ = best_model['fooofed_spectrum']
        self._ap_fit = best_model["ap_fit"]
        self.aperiodic_params_= best_model['aperiodic_params']        
        self.gaussian_params_ = best_model["gauss_params"]
        
        # Convert gaussian definitions to peak parameters
        self.peak_params_ = self._create_peak_params(self.gaussian_params_)
        self._best_model = best_model
        self.r_squared_ = best_model['r_squared']
        self.error_ = best_model['error']
        self.bic_ = best_model['BIC']
        self.aic_ = best_model['AIC']
        self.bf_ = best_model['BF']
        self._loglik = best_model['loglik']
        
        
        

    def print_settings(self, description=False, concise=False):
        """Print out the current settings.

        Parameters
        ----------
        description : bool, optional, default: False
            Whether to print out a description with current settings.
        concise : bool, optional, default: False
            Whether to print the report in a concise mode, or not.
        """

        print(gen_settings_str(self, description, concise))


    def print_results(self, concise=False):
        """Print out model fitting results.

        Parameters
        ----------
        concise : bool, optional, default: False
            Whether to print the report in a concise mode, or not.
        """

        print(gen_results_fm_str(self, concise))


    @staticmethod
    def print_report_issue(concise=False):
        """Prints instructions on how to report bugs and/or problematic fits.

        Parameters
        ----------
        concise : bool, optional, default: False
            Whether to print the report in a concise mode, or not.
        """

        print(gen_issue_str(concise))


    def get_settings(self):
        """Return user defined settings of the current object.

        Returns
        -------
        FOOOFSettings
            Object containing the settings from the current object.
        """

        return FOOOFSettings(**{key : getattr(self, key) \
                             for key in OBJ_DESC['settings']})


    def get_meta_data(self):
        """Return data information from the current object.

        Returns
        -------
        FOOOFMetaData
            Object containing meta data from the current object.
        """

        return FOOOFMetaData(**{key : getattr(self, key) \
                             for key in OBJ_DESC['meta_data']})


    def get_params(self, name, col=None):
        """Return model fit parameters for specified feature(s).

        Parameters
        ----------
        name : {'aperiodic_params', 'peak_params', 'gaussian_params', 'error', 'r_squared'}
            Name of the data field to extract.
        col : {'CF', 'PW', 'BW', 'offset', 'knee', 'exponent'} or int, optional
            Column name / index to extract from selected data, if requested.
            Only used for name of {'aperiodic_params', 'peak_params', 'gaussian_params'}.

        Returns
        -------
        out : float or 1d array
            Requested data.

        Raises
        ------
        NoModelError
            If there are no model fit parameters available to return.

        Notes
        -----
        If there are no fit peak (no peak parameters), this method will return NaN.
        """

        if not self.has_model:
            raise NoModelError("No model fit results are available to extract, can not proceed.")

        # If col specified as string, get mapping back to integer
        if isinstance(col, str):
            col = get_indices(self.aperiodic_mode)[col]

        # Allow for shortcut alias, without adding `_params`
        if name in ['aperiodic', 'peak', 'gaussian']:
            name = name + '_params'

        # Extract the request data field from object
        out = getattr(self, name + '_')

        # Periodic values can be empty arrays and if so, replace with NaN array
        if isinstance(out, np.ndarray) and out.size == 0:
            out = np.array([np.nan, np.nan, np.nan])

        # Select out a specific column, if requested
        if col is not None:

            # Extract column, & if result is a single value in an array, unpack from array
            out = out[col] if out.ndim == 1 else out[:, col]
            out = out[0] if isinstance(out, np.ndarray) and out.size == 1 else out

        return out


    def get_results(self):
        """Return model fit parameters and goodness of fit metrics.

        Returns
        -------
        FOOOFResults
            Object containing the model fit results from the current object.
        """

        return FOOOFResults(**{key.strip('_') : getattr(self, key) \
            for key in OBJ_DESC['results']})


    @copy_doc_func_to_method(plot_fm)
    def plot(self, plot_peaks=None, plot_aperiodic=True, plt_log=False,
             add_legend=True, save_fig=False, file_name=None, file_path=None,
             ax=None, data_kwargs=None, model_kwargs=None,
             aperiodic_kwargs=None, peak_kwargs=None, **plot_kwargs):

        plot_fm(self, plot_peaks=plot_peaks, plot_aperiodic=plot_aperiodic, plt_log=plt_log,
                add_legend=add_legend, save_fig=save_fig, file_name=file_name,
                file_path=file_path, ax=ax, data_kwargs=data_kwargs, model_kwargs=model_kwargs,
                aperiodic_kwargs=aperiodic_kwargs, peak_kwargs=peak_kwargs, **plot_kwargs)


        
        
    @copy_doc_func_to_method(save_report_fm)
    def save_report(self, file_name, file_path=None, plt_log=False,
                    add_settings=True, **plot_kwargs):

        save_report_fm(self, file_name, file_path, plt_log, add_settings, **plot_kwargs)


    @copy_doc_func_to_method(save_fm)
    def save(self, file_name, file_path=None, append=False,
             save_results=False, save_settings=False, save_data=False):

        save_fm(self, file_name, file_path, append, save_results, save_settings, save_data)


    def load(self, file_name, file_path=None, regenerate=True):
        """Load in a FOOOF formatted JSON file to the current object.

        Parameters
        ----------
        file_name : str or FileObject
            File to load data from.
        file_path : str or None, optional
            Path to directory to load from. If None, loads from current directory.
        regenerate : bool, optional, default: True
            Whether to regenerate the model fit from the loaded data, if data is available.
        """

        # Reset data in object, so old data can't interfere
        self._reset_data_results(True, True, True)

        # Load JSON file, add to self and check loaded data
        data = load_json(file_name, file_path)
        self._add_from_dict(data)
        self._check_loaded_settings(data)
        self._check_loaded_results(data)

        # Regenerate model components, based on what is available
        if regenerate:
            if self.freq_res:
                self._regenerate_freqs()
            if np.all(self.freqs) and np.all(self.aperiodic_params_):
                self._regenerate_model()


    def copy(self):
        """Return a copy of the current object."""

        return deepcopy(self)


    def set_debug_mode(self, debug):
        """Set debug mode, which controls if an error is raised if model fitting is unsuccessful.

        Parameters
        ----------
        debug : bool
            Whether to run in debug mode.
        """

        self._debug = debug


    def set_check_data_mode(self, check_data):
        """Set check data mode, which controls if an error is raised if NaN or Inf data are added.

        Parameters
        ----------
        check_data : bool
            Whether to run in check data mode.
        """

        self._check_data = check_data


    def to_df(self, peak_org):
        """Convert and extract the model results as a pandas object.

        Parameters
        ----------
        peak_org : int or Bands
            How to organize peaks.
            If int, extracts the first n peaks.
            If Bands, extracts peaks based on band definitions.

        Returns
        -------
        pd.Series
            Model results organized into a pandas object.
        """

        return model_to_dataframe(self.get_results(), peak_org)


    def _check_width_limits(self):
        """Check and warn about peak width limits / frequency resolution interaction."""

        # Check peak width limits against frequency resolution and warn if too close
        if 1.5 * self.freq_res >= self.peak_width_limits[0]:
            print(gen_width_warning_str(self.freq_res, self.peak_width_limits[0]))


    def _simple_ap_fit(self, freqs, power_spectrum):
        """Fit the aperiodic component of the power spectrum.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power_spectrum, in linear scale.
        power_spectrum : 1d array
            Power values, in log10 scale.

        Returns
        -------
        aperiodic_params : 1d array
            Parameter estimates for aperiodic fit.
        """

        # Get the guess parameters and/or calculate from the data, as needed
        #   Note that these are collected as lists, to concatenate with or without knee later
        off_guess = [power_spectrum[0] if not self._ap_guess[0] else self._ap_guess[0]]
        kne_guess = [self._ap_guess[1]] if self.aperiodic_mode in ['knee', 'lorentzian', 'lorentzian-noise-floor'] else []
        exp_guess = [np.abs((self.power_spectrum[-1] - self.power_spectrum[0]) /
                            (np.log10(self.freqs[-1]) - np.log10(self.freqs[0])))
                     if not self._ap_guess[2] else self._ap_guess[2]]
        nfl_guess = [np.min(power_spectrum)] if self.aperiodic_mode in ['lorentzian-noise-floor', 'fixed-noise-floor'] else []

        # Get bounds for aperiodic fitting, dropping knee bound if not set to fit knee
        if self.aperiodic_mode == 'fixed':
            ap_bounds = tuple(bound[0::2] for bound in self._ap_bounds)
        elif self.aperiodic_mode in ['knee', 'lorentzian', 'lorentzian-noise-floor', 'fixed-noise-floor']:
            ap_bounds = self._ap_bounds 

        # Collect together guess parameters
        guess = np.array(off_guess + kne_guess + exp_guess + nfl_guess)

        # Ignore warnings that are raised in curve_fit
        #   A runtime warning can occur while exploring parameters in curve fitting
        #     This doesn't effect outcome - it won't settle on an answer that does this
        #   It happens if / when b < 0 & |b| > x**2, as it leads to log of a negative number
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.aperiodic_mode in ['fixed', 'knee']:
                    aperiodic_params, _ = curve_fit(get_ap_func(self.aperiodic_mode),
                                                    freqs, power_spectrum, p0=guess,
                                                    maxfev=self._maxfev, bounds=ap_bounds)
                elif self.aperiodic_mode in ['lorentzian']:
                    aperiodic_params = self.aperiodic_regression(lorentzian_function,
                                                    freqs, power_spectrum, np.array(guess),
                                                    self.regularization_weight)
                elif self.aperiodic_mode in ['lorentzian-noise-floor']:
                    aperiodic_params = self.aperiodic_regression(lorentzian_noise_floor_function,
                                                    freqs, power_spectrum, np.array(guess),
                                                    self.regularization_weight)
                elif self.aperiodic_mode in ['fixed-noise-floor']:
                    aperiodic_params = self.aperiodic_regression(fixed_noise_floor_function,
                                                    freqs, power_spectrum, np.array(guess),
                                                    self.regularization_weight)
        except RuntimeError as excp:
            error_msg = ("Model fitting failed due to not finding parameters in "
                         "the simple aperiodic component fit.")
            raise FitError(error_msg) from excp

        return aperiodic_params


    def _robust_ap_fit(self, freqs, power_spectrum):
        """Fit the aperiodic component of the power spectrum robustly, ignoring outliers.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power spectrum, in linear scale.
        power_spectrum : 1d array
            Power values, in log10 scale.

        Returns
        -------
        aperiodic_params : 1d array
            Parameter estimates for aperiodic fit.

        Raises
        ------
        FitError
            If the fitting encounters an error.
        """

        # Do a quick, initial aperiodic fit
        popt = self._simple_ap_fit(freqs, power_spectrum)
        initial_fit = gen_aperiodic(freqs, popt, self.aperiodic_mode)

        # Flatten power_spectrum based on initial aperiodic fit
        flatspec = power_spectrum - initial_fit

        # Flatten outliers, defined as any points that drop below 0
        flatspec[flatspec < 0] = 0

        # Use percentile threshold, in terms of # of points, to extract and re-fit
        perc_thresh = np.percentile(flatspec, self._ap_percentile_thresh)
        perc_mask = flatspec <= perc_thresh
        freqs_ignore = freqs[perc_mask]
        spectrum_ignore = power_spectrum[perc_mask]

        # Get bounds for aperiodic fitting, dropping knee bound if not set to fit knee
        if self.aperiodic_mode == 'fixed':
            ap_bounds = tuple(bound[0::2] for bound in self._ap_bounds)
        elif self.aperiodic_mode in ['knee', 'lorentzian', 'lorentzian-noise-floor', 'fixed-noise-floor']:
            ap_bounds = self._ap_bounds 

        # Second aperiodic fit - using results of first fit as guess parameters
        #  See note in _simple_ap_fit about warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.aperiodic_mode in ['fixed', 'knee']:
                    aperiodic_params, _ = curve_fit(get_ap_func(self.aperiodic_mode),
                                                    freqs_ignore, spectrum_ignore, p0=popt,
                                                    maxfev=self._maxfev, bounds=ap_bounds)
                elif self.aperiodic_mode in ['lorentzian']:
                    aperiodic_params = self.aperiodic_regression(lorentzian_function,
                                                    freqs_ignore, spectrum_ignore, popt,
                                                    self.regularization_weight)
                elif self.aperiodic_mode in ['lorentzian-noise-floor']:
                    aperiodic_params = self.aperiodic_regression(lorentzian_noise_floor_function,
                                                    freqs_ignore, spectrum_ignore, popt,
                                                    self.regularization_weight)    
                elif self.aperiodic_mode in ['fixed-noise-floor']:
                    aperiodic_params = self.aperiodic_regression(fixed_noise_floor_function,
                                                    freqs_ignore, spectrum_ignore, popt,
                                                    self.regularization_weight)    

        except RuntimeError as excp:
            error_msg = ("Model fitting failed due to not finding "
                         "parameters in the robust aperiodic fit.")
            raise FitError(error_msg) from excp
        except TypeError as excp:
            error_msg = ("Model fitting failed due to sub-sampling "
                         "in the robust aperiodic fit.")
            raise FitError(error_msg) from excp

        return aperiodic_params


    def _est_peaks(self, flat_iter):
        """Iteratively fit peaks to flattened spectrum.

        Parameters
        ----------
        flat_iter : 1d array
            Flattened power spectrum values.

        Returns
        -------
        gaussian_params : 2d array
            Parameters that define the gaussian fit(s).
            Each row is a gaussian, as [mean, height, standard deviation].
        """

        # Initialize matrix of guess parameters for gaussian fitting
        guess = np.empty([0, 3])

        # Find peak: Loop through, finding a candidate peak, and fitting with a guess gaussian
        #   Stopping procedures: limit on # of peaks, or relative or absolute height thresholds
        while len(guess) < self.max_n_peaks:

            # Find candidate peak - the maximum point of the flattened spectrum
            max_ind = np.argmax(flat_iter)
            max_height = flat_iter[max_ind]

            # Stop searching for peaks once height drops below height threshold
            if max_height <= self.peak_threshold * np.std(flat_iter):
                break

            # Set the guess parameters for gaussian fitting, specifying the mean and height
            guess_freq = self.freqs[max_ind]
            guess_height = max_height

            # Halt fitting process if candidate peak drops below minimum height
            if not guess_height > self.min_peak_height:
                break

            # Data-driven first guess at standard deviation
            #   Find half height index on each side of the center frequency
            half_height = 0.5 * max_height
            le_ind = next((val for val in range(max_ind - 1, 0, -1)
                           if flat_iter[val] <= half_height), None)
            ri_ind = next((val for val in range(max_ind + 1, len(flat_iter), 1)
                           if flat_iter[val] <= half_height), None)

            # Guess bandwidth procedure: estimate the width of the peak
            try:
                # Get an estimated width from the shortest side of the peak
                #   We grab shortest to avoid estimating very large values from overlapping peaks
                # Grab the shortest side, ignoring a side if the half max was not found
                short_side = min([abs(ind - max_ind) \
                    for ind in [le_ind, ri_ind] if ind is not None])

                # Use the shortest side to estimate full-width, half max (converted to Hz)
                #   and use this to estimate that guess for gaussian standard deviation
                fwhm = short_side * 2 * self.freq_res
                guess_std = compute_gauss_std(fwhm)

            except ValueError:
                # This procedure can fail (very rarely), if both left & right inds end up as None
                #   In this case, default the guess to the average of the peak width limits
                guess_std = np.mean(self.peak_width_limits)

            # Check that guess value isn't outside preset limits - restrict if so
            #   Note: without this, curve_fitting fails if given guess > or < bounds
            if guess_std < self._gauss_std_limits[0]:
                guess_std = self._gauss_std_limits[0]
            if guess_std > self._gauss_std_limits[1]:
                guess_std = self._gauss_std_limits[1]

            # Collect guess parameters and subtract this guess gaussian from the data
            guess = np.vstack((guess, (guess_freq, guess_height, guess_std)))
            peak_gauss = gaussian_function(self.freqs, guess_freq, guess_height, guess_std)
            flat_iter = flat_iter - peak_gauss

        # Check peaks based on edges, and on overlap, dropping any that violate requirements
        guess = self._drop_peak_cf(guess)
        guess = self._drop_peak_overlap(guess)

        return guess
    
    
    
    def _est_fit(self,guess): # split this step from _fit_peaks
        # If there are peak guesses, fit the peaks, and sort results
        if len(guess) > 0:
            gaussian_params = self._fit_peak_guess(guess)
            gaussian_params = gaussian_params[gaussian_params[:, 0].argsort()]
        else:
            gaussian_params = np.empty([0, 3])
        return gaussian_params
    
    def _fit_peaks(self, flat_iter): # refactored the original _fit_peaks as two-steps function
        guess = self._est_peaks(flat_iter)
        gaussian_params = self._est_fit(guess)
        return gaussian_params
    
    
    
    def _fit_model(self, guess, aperiodic_params):
        """Fits a model to the data used for compare BIC models.

        Parameters
        ----------
        guess : 2d array, shape=[n_peaks, 3]
            Guess parameters for gaussian fits to peaks, as aperiodic and gaussian parameters.

        Returns
        -------
        gaussian_params : 2d array, shape=[n_peaks, 3]
            Parameters for gaussian fits to peaks, as gaussian parameters.
        """

        # Set the bounds for CF, enforce positive height value, and set bandwidth limits
        #   Note that 'guess' is in terms of gaussian std, so +/- BW is 2 * the guess_gauss_std
        #   This set of list comprehensions is a way to end up with bounds in the form:
        #     ((cf_low_peak1, height_low_peak1, bw_low_peak1, *repeated for n_peaks*),
        lo_bound = [[peak[0] - 2 * self._cf_bound * peak[2], 0, self._gauss_std_limits[0]]
                    for peak in guess]
        hi_bound = [[peak[0] + 2 * self._cf_bound * peak[2], np.inf, self._gauss_std_limits[1]]
                    for peak in guess]

        # Check that CF bounds are within frequency range
        #   If they are  not, update them to be restricted to frequency range
        lo_bound = [bound if bound[0] > self.freq_range[0] else \
            [self.freq_range[0], *bound[1:]] for bound in lo_bound]
        hi_bound = [bound if bound[0] < self.freq_range[1] else \
            [self.freq_range[1], *bound[1:]] for bound in hi_bound]

        # Unpacks the embedded lists into flat tuples
        #   This is what the fit function requires as input
        if self.aperiodic_mode in ['fixed', 'knee']:
            gaus_param_bounds = (tuple(item for sublist in lo_bound for item in sublist),
                                tuple(item for sublist in hi_bound for item in sublist))
        elif self.aperiodic_mode in ['lorentzian', 'lorentzian-noise-floor', 'fixed-noise-floor']:
            lo_bound_flatten = np.array(lo_bound).flatten()
            hi_bound_flatten = np.array(hi_bound).flatten()
            gaus_param_bounds = []
            for i in range(len(lo_bound_flatten)):
                lo_bound_oi = lo_bound_flatten[i]
                hi_bound_oi = hi_bound_flatten[i]
                if lo_bound_oi in [np.inf, -np.inf]:
                    lo_bound_oi = None
                if hi_bound_oi in [np.inf, -np.inf]:
                    hi_bound_oi = None
                gaus_param_bounds.append((lo_bound_oi, hi_bound_oi))

        # Flatten guess, for use with curve fit and incorporate aperiodic params
        print("checking parameters")

        
        guess_tot = np.concatenate([aperiodic_params, np.ravel(guess)])
        param_bounds = self._ap_bounds + tuple(gaus_param_bounds)
        
        print(list(guess_tot))
        print(tuple(param_bounds))
        

        # Fit the peaks
        try:
            if self.aperiodic_mode in ['lorentzian', 'lorentzian-noise-floor', 'fixed-noise-floor']:
             params = self.model_regression(get_ap_func(self.aperiodic_mode), self.freqs, self.power_spectrum, 
                                                        list(guess_tot), tuple(param_bounds), 
                                                        self.regularization_weight
                                                        ,)
        except RuntimeError as excp:
            error_msg = ("Model fitting failed due to not finding "
                         "parameters in the peak component fit.")
            raise FitError(error_msg) from excp
        except LinAlgError as excp:
            error_msg = ("Model fitting failed due to a LinAlgError during peak fitting. "
                         "This can happen with settings that are too liberal, leading, "
                         "to a large number of guess peaks that cannot be fit together.")
            raise FitError(error_msg) from excp


        if self.aperiodic_mode == 'lorentzian-noise-floor':
            ap_params = params[:4]
            gaussian_params = params[4:]
          
        elif self.aperiodic_mode == 'fixed-noise-floor':
            ap_params = params[:3]
            gaussian_params = params[3:]  
                          
        # Re-organize params into 2d matrix
        gaussian_params = np.array(group_three(gaussian_params))

        return ap_params, gaussian_params
           

    def _fit_peak_guess(self, guess):
        """Fits a group of peak guesses with a fit function.

        Parameters
        ----------
        guess : 2d array, shape=[n_peaks, 3]
            Guess parameters for gaussian fits to peaks, as gaussian parameters.

        Returns
        -------
        gaussian_params : 2d array, shape=[n_peaks, 3]
            Parameters for gaussian fits to peaks, as gaussian parameters.
        """

        # Set the bounds for CF, enforce positive height value, and set bandwidth limits
        #   Note that 'guess' is in terms of gaussian std, so +/- BW is 2 * the guess_gauss_std
        #   This set of list comprehensions is a way to end up with bounds in the form:
        #     ((cf_low_peak1, height_low_peak1, bw_low_peak1, *repeated for n_peaks*),
        #      (cf_high_peak1, height_high_peak1, bw_high_peak, *repeated for n_peaks*))
        #     ^where each value sets the bound on the specified parameter
        lo_bound = [[peak[0] - 2 * self._cf_bound * peak[2], 0, self._gauss_std_limits[0]]
                    for peak in guess]
        hi_bound = [[peak[0] + 2 * self._cf_bound * peak[2], np.inf, self._gauss_std_limits[1]]
                    for peak in guess]

        # Check that CF bounds are within frequency range
        #   If they are  not, update them to be restricted to frequency range
        lo_bound = [bound if bound[0] > self.freq_range[0] else \
            [self.freq_range[0], *bound[1:]] for bound in lo_bound]
        hi_bound = [bound if bound[0] < self.freq_range[1] else \
            [self.freq_range[1], *bound[1:]] for bound in hi_bound]

        # Unpacks the embedded lists into flat tuples
        #   This is what the fit function requires as input
        if self.aperiodic_mode in ['fixed', 'knee']:
            gaus_param_bounds = (tuple(item for sublist in lo_bound for item in sublist),
                                tuple(item for sublist in hi_bound for item in sublist))
        elif self.aperiodic_mode in ['lorentzian', 'lorentzian-noise-floor', 'fixed-noise-floor']:
            lo_bound_flatten = np.array(lo_bound).flatten()
            hi_bound_flatten = np.array(hi_bound).flatten()
            gaus_param_bounds = []
            for i in range(len(lo_bound_flatten)):
                lo_bound_oi = lo_bound_flatten[i]
                hi_bound_oi = hi_bound_flatten[i]
                if lo_bound_oi in [np.inf, -np.inf]:
                    lo_bound_oi = None
                if hi_bound_oi in [np.inf, -np.inf]:
                    hi_bound_oi = None
                gaus_param_bounds.append((lo_bound_oi, hi_bound_oi))

        # Flatten guess, for use with curve fit
        guess = np.ndarray.flatten(guess)

        # Fit the peaks
        try:
            if self.aperiodic_mode in ['fixed', 'knee']:
                gaussian_params, _ = curve_fit(gaussian_function, self.freqs, self._spectrum_flat,
                                            p0=guess, maxfev=self._maxfev, bounds=gaus_param_bounds)
            elif self.aperiodic_mode in ['lorentzian', 'lorentzian-noise-floor', 'fixed-noise-floor']:
                gaussian_params = self.gaussian_regression(self.freqs, self._spectrum_flat, 
                                                        list(guess), self.regularization_weight, 
                                                        tuple(gaus_param_bounds),)
        except RuntimeError as excp:
            error_msg = ("Model fitting failed due to not finding "
                         "parameters in the peak component fit.")
            raise FitError(error_msg) from excp
        except LinAlgError as excp:
            error_msg = ("Model fitting failed due to a LinAlgError during peak fitting. "
                         "This can happen with settings that are too liberal, leading, "
                         "to a large number of guess peaks that cannot be fit together.")
            raise FitError(error_msg) from excp

        # Re-organize params into 2d matrix
        gaussian_params = np.array(group_three(gaussian_params))

        return gaussian_params


    def _create_peak_params(self, gaus_params):
        """Copies over the gaussian params to peak outputs, updating as appropriate.

        Parameters
        ----------
        gaus_params : 2d array
            Parameters that define the gaussian fit(s), as gaussian parameters.

        Returns
        -------
        peak_params : 2d array
            Fitted parameter values for the peaks, with each row as [CF, PW, BW].

        Notes
        -----
        The gaussian center is unchanged as the peak center frequency.

        The gaussian height is updated to reflect the height of the peak above
        the aperiodic fit. This is returned instead of the gaussian height, as
        the gaussian height is harder to interpret, due to peak overlaps.

        The gaussian standard deviation is updated to be 'both-sided', to reflect the
        'bandwidth' of the peak, as opposed to the gaussian parameter, which is 1-sided.

        Performing this conversion requires that the model has been run,
        with `freqs`, `fooofed_spectrum_` and `_ap_fit` all required to be available.
        """

        peak_params = np.empty((len(gaus_params), 3))

        for ii, peak in enumerate(gaus_params):

            # Gets the index of the power_spectrum at the frequency closest to the CF of the peak
            ind = np.argmin(np.abs(self.freqs - peak[0]))

            # Collect peak parameter data
            peak_params[ii] = [peak[0], self.fooofed_spectrum_[ind] - self._ap_fit[ind],
                               peak[2] * 2]

        return peak_params


    def _drop_peak_cf(self, guess):
        """Check whether to drop peaks based on center's proximity to the edge of the spectrum.

        Parameters
        ----------
        guess : 2d array
            Guess parameters for gaussian peak fits. Shape: [n_peaks, 3].

        Returns
        -------
        guess : 2d array
            Guess parameters for gaussian peak fits. Shape: [n_peaks, 3].
        """

        cf_params = guess[:, 0]
        bw_params = guess[:, 2] * self.bw_std_edge

        # Check if peaks within drop threshold from the edge of the frequency range
        keep_peak = \
            (np.abs(np.subtract(cf_params, self.freq_range[0])) > bw_params) & \
            (np.abs(np.subtract(cf_params, self.freq_range[1])) > bw_params)

        # Drop peaks that fail the center frequency edge criterion
        guess = np.array([gu for (gu, keep) in zip(guess, keep_peak) if keep])

        return guess


    def _drop_peak_overlap(self, guess):
        """Checks whether to drop gaussians based on amount of overlap.

        Parameters
        ----------
        guess : 2d array
            Guess parameters for gaussian peak fits. Shape: [n_peaks, 3].

        Returns
        -------
        guess : 2d array
            Guess parameters for gaussian peak fits. Shape: [n_peaks, 3].

        Notes
        -----
        For any gaussians with an overlap that crosses the threshold,
        the lowest height guess Gaussian is dropped.
        """

        # Sort the peak guesses by increasing frequency
        #   This is so adjacent peaks can be compared from right to left
        guess = sorted(guess, key=lambda x: float(x[0]))

        # Calculate standard deviation bounds for checking amount of overlap
        #   The bounds are the gaussian frequency +/- gaussian standard deviation
        bounds = [[peak[0] - peak[2] * self._gauss_overlap_thresh,
                   peak[0] + peak[2] * self._gauss_overlap_thresh] for peak in guess]

        # Loop through peak bounds, comparing current bound to that of next peak
        #   If the left peak's upper bound extends pass the right peaks lower bound,
        #   then drop the Gaussian with the lower height
        drop_inds = []
        for ind, b_0 in enumerate(bounds[:-1]):
            b_1 = bounds[ind + 1]

            # Check if bound of current peak extends into next peak
            if b_0[1] > b_1[0]:

                # If so, get the index of the gaussian with the lowest height (to drop)
                drop_inds.append([ind, ind + 1][np.argmin([guess[ind][1], guess[ind + 1][1]])])

        # Drop any peaks guesses that overlap too much, based on threshold
        keep_peak = [not ind in drop_inds for ind in range(len(guess))]
        guess = np.array([gu for (gu, keep) in zip(guess, keep_peak) if keep])

        return guess


    def _calc_r_squared(self):
        """Calculate the r-squared goodness of fit of the model, compared to the original data."""
        self.r_squared_  = calc_r_squared(self.power_spectrum, self.fooofed_spectrum_)



    

    

    def _calc_error(self, metric=None):
        """Calculate the overall error of the model fit, compared to the original data.

        Parameters
        ----------
        metric : {'MAE', 'MSE', 'RMSE'}, optional
            Which error measure to calculate:
            * 'MAE' : mean absolute error
            * 'MSE' : mean squared error
            * 'RMSE' : root mean squared error

        Raises
        ------
        ValueError
            If the requested error metric is not understood.

        Notes
        -----
        Which measure is applied is by default controlled by the `_error_metric` attribute.
        """

        # If metric is not specified, use the default approach
        metric = self._error_metric if not metric else metric
        
        self.error_ = calc_error(metric, self.power_spectrum,self.fooofed_spectrum_)

       


    def _prepare_data(self, freqs, power_spectrum, freq_range, spectra_dim=1):
        """Prepare input data for adding to current object.

        Parameters
        ----------
        freqs : 1d array
            Frequency values for the power_spectrum, in linear space.
        power_spectrum : 1d or 2d array
            Power values, which must be input in linear space.
            1d vector, or 2d as [n_power_spectra, n_freqs].
        freq_range : list of [float, float]
            Frequency range to restrict power spectrum to. If None, keeps the entire range.
        spectra_dim : int, optional, default: 1
            Dimensionality that the power spectra should have.

        Returns
        -------
        freqs : 1d array
            Frequency values for the power_spectrum, in linear space.
        power_spectrum : 1d or 2d array
            Power spectrum values, in log10 scale.
            1d vector, or 2d as [n_power_specta, n_freqs].
        freq_range : list of [float, float]
            Minimum and maximum values of the frequency vector.
        freq_res : float
            Frequency resolution of the power spectrum.

        Raises
        ------
        DataError
            If there is an issue with the data.
        InconsistentDataError
            If the input data are inconsistent size.
        """

        # Check that data are the right types
        if not isinstance(freqs, np.ndarray) or not isinstance(power_spectrum, np.ndarray):
            raise DataError("Input data must be numpy arrays.")

        # Check that data have the right dimensionality
        if freqs.ndim != 1 or (power_spectrum.ndim != spectra_dim):
            raise DataError("Inputs are not the right dimensions.")

        # Check that data sizes are compatible
        if freqs.shape[-1] != power_spectrum.shape[-1]:
            raise InconsistentDataError("The input frequencies and power spectra "
                                        "are not consistent size.")

        # Check if power values are complex
        if np.iscomplexobj(power_spectrum):
            raise DataError("Input power spectra are complex values. "
                            "FOOOF does not currently support complex inputs.")

        # Force data to be dtype of float64
        #   If they end up as float32, or less, scipy curve_fit fails (sometimes implicitly)
        if freqs.dtype != 'float64':
            freqs = freqs.astype('float64')
        if power_spectrum.dtype != 'float64':
            power_spectrum = power_spectrum.astype('float64')

        # Check frequency range, trim the power_spectrum range if requested
        if freq_range:
            freqs, power_spectrum = trim_spectrum(freqs, power_spectrum, freq_range)

        # Check if freqs start at 0 and move up one value if so
        #   Aperiodic fit gets an inf if freq of 0 is included, which leads to an error
        if freqs[0] == 0.0:
            freqs, power_spectrum = trim_spectrum(freqs, power_spectrum, [freqs[1], freqs.max()])
            if self.verbose:
                print("\nFOOOF WARNING: Skipping frequency == 0, "
                      "as this causes a problem with fitting.")

        # Calculate frequency resolution, and actual frequency range of the data
        freq_range = [freqs.min(), freqs.max()]
        freq_res = freqs[1] - freqs[0]

        # Log power values
        power_spectrum = np.log10(power_spectrum)

        ## Data checks - run checks on inputs based on check modes

        if self._check_freqs:
            # Check if the frequency data is unevenly spaced, and raise an error if so
            freq_diffs = np.diff(freqs)
            if not np.all(np.isclose(freq_diffs, freq_res)):
                raise DataError("The input frequency values are not evenly spaced. "
                                "The model expects equidistant frequency values in linear space.")
        if self._check_data:
            # Check if there are any infs / nans, and raise an error if so
            if np.any(np.isinf(power_spectrum)) or np.any(np.isnan(power_spectrum)):
                error_msg = ("The input power spectra data, after logging, contains NaNs or Infs. "
                             "This will cause the fitting to fail. "
                             "One reason this can happen is if inputs are already logged. "
                             "Inputs data should be in linear spacing, not log.")
                raise DataError(error_msg)

        return freqs, power_spectrum, freq_range, freq_res


    def _add_from_dict(self, data):
        """Add data to object from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary of data to add to self.
        """

        # Reconstruct object from loaded data
        for key in data.keys():
            setattr(self, key, data[key])


    def _check_loaded_results(self, data):
        """Check if results have been added and check data.

        Parameters
        ----------
        data : dict
            A dictionary of data that has been added to the object.
        """

        # If results loaded, check dimensions of peak parameters
        #   This fixes an issue where they end up the wrong shape if they are empty (no peaks)
        if set(OBJ_DESC['results']).issubset(set(data.keys())):
            self.peak_params_ = check_array_dim(self.peak_params_)
            self.gaussian_params_ = check_array_dim(self.gaussian_params_)


    def _check_loaded_settings(self, data):
        """Check if settings added, and update the object as needed.

        Parameters
        ----------
        data : dict
            A dictionary of data that has been added to the object.
        """

        # If settings not loaded from file, clear from object, so that default
        # settings, which are potentially wrong for loaded data, aren't kept
        if not set(OBJ_DESC['settings']).issubset(set(data.keys())):

            # Reset all public settings to None
            for setting in OBJ_DESC['settings']:
                setattr(self, setting, None)

            # If aperiodic params available, infer whether knee fitting was used,
            if not np.all(np.isnan(self.aperiodic_params_)):
                self.aperiodic_mode = infer_ap_func(self.aperiodic_params_)

        # Reset internal settings so that they are consistent with what was loaded
        #   Note that this will set internal settings to None, if public settings unavailable
        self._reset_internal_settings()


    def _regenerate_freqs(self):
        """Regenerate the frequency vector, given the object metadata."""

        self.freqs = gen_freqs(self.freq_range, self.freq_res)


    def _regenerate_model(self):
        """Regenerate model fit from parameters."""

        self.fooofed_spectrum_, self._peak_fit, self._ap_fit = gen_model(
            self.freqs, self.aperiodic_params_, self.aperiodic_mode, self.gaussian_params_, return_components=True)
