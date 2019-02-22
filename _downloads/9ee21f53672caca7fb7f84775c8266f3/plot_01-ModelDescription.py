"""
01: Model Description
=====================

A theoretical / mathematical description of the FOOOF model.
"""

###################################################################################################
# Introduction
# ------------
#
# A neural power spectrum is fit as a combination of an aperiodic signal and periodic oscillations.
#
# The aperiodic component of the signal displays 1/f like properties.
#
# Putative oscillations (hereafter referred to as 'peaks'), are frequency regions
# in which there are 'bumps' of power over and above the aperiodic signal.
#
# This formulation roughly translates to fitting the power spectrum as:
#
# .. math::
#    P = L + \sum_{n=0}^{N} G_n
#
# Where `P` is the power spectrum, `L` is the aperiodic signal, and each :math:`G_n`
# is a Gaussian fit to a peak, for `N` total peaks extracted from the power spectrum.
#

###################################################################################################
# Aperiodic Fit
# -------------
#
# The aperiodic fit uses an exponential function, fit on the semilog power spectrum
# (linear frequencies and :math:`log_{10}` power values).
#
# The exponential is of the form:
#
# .. math::
#    L = 10^b * \frac{1}{(k + F^\chi)}
#
# Or, equivalently:
#
# .. math::
#    L = b - \log(k + F^\chi)
#
# In this formulation, the parameters `b`, `k`, and :math:`\chi`
# define the aperiodic signal, as:
#
# - `b` is the broadband 'offset'
# - `k` relates to the 'knee'
# - :math:`\chi` is the 'exponent' of the aperiodic fit
# - `F` is the vector of input frequencies
#
# Note that fitting the knee parameter is optional. If used, the knee defines a bend in the
# aperiodic `1/f` like component of the signal.
#
# By default the aperiodic signal is fit with the 'knee' parameter set to zero.
# This fits the aperiodic signal equivalently to fitting a linear fit in log-log space.
#
# Broader frequency ranges typically do not display a single 1/f like characteristic,
# and so for these cases fitting with the knee parameter allows for modelling bends
# in the aperiodic signal.
#

###################################################################################################
# Peaks
# -----
#
# Regions of power over above this aperiodic signal, as defined above, are considered
# to be putative oscillations and are fit in the model by a Gaussian.
#
# For each Gaussian, :math:`G_n`, with the form:
#
# .. math::
#    G_n = a * exp (\frac{- (F - c)^2}{2 * w^2})
#
# Each peak is defined in terms of parameters `a`, `c` and `w`, where:
#
# - `a` is the amplitude of the peak, over and above the aperiodic signal
# - `c` is the center frequency of the peak
# - `w` is the width of the peak
# - `F` is the vector of input frequencies
#
# The full power spectrum fit is therefore the combination of the aperiodic fit,
# `L` defined by the exponential fit, and `N` peaks, where each :math:`G_n` is
# formalized as a Gaussian process.
#
# Full method details are available in the paper:
# https://www.biorxiv.org/content/early/2018/04/11/299859
#

###################################################################################################
# This procedure is able to create a model of the neural power spectrum,
# that is fully described mathematical by the mathematical model from above.
#
