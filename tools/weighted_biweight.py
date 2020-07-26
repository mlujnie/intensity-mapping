import numpy as np
#from astropy.stats.funcs import median_absolute_deviation
import random
from astropy.stats import biweight_location, biweight_midvariance, median_absolute_deviation

def biweight_location_weights(data, weights, c=6.0, M=None, axis=None):
    r"""
    Compute the biweight location.

    The biweight location is a robust statistic for determining the
    central location of a distribution.  It is given by:

    .. math::

        \zeta_{biloc}= M + \frac{\Sigma_{|u_i|<1} \ (x_i - M) (1 - u_i^2)^2}
            {\Sigma_{|u_i|<1} \ (1 - u_i^2)^2}

    where :math:`x` is the input data, :math:`M` is the sample median
    (or the input initial location guess) and :math:`u_i` is given by:

    .. math::

        u_{i} = \frac{(x_i - M)}{c * MAD}

    where :math:`c` is the tuning constant and :math:`MAD` is the
    `median absolute deviation
    <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_.  The
    biweight location tuning constant ``c`` is typically 6.0 (the
    default).

    Parameters
    ----------
    data : array-like
        Input array or object that can be converted to an array.
    c : float, optional
        Tuning constant for the biweight estimator (default = 6.0).
    M : float or array-like, optional
        Initial guess for the location.  If ``M`` is a scalar value,
        then its value will be used for the entire array (or along each
        ``axis``, if specified).  If ``M`` is an array, then its must be
        an array containing the initial location estimate along each
        ``axis`` of the input array.  If `None` (default), then the
        median of the input array will be used (or along each ``axis``,
        if specified).
    axis : int, optional
        The axis along which the biweight locations are computed.  If
        `None` (default), then the biweight location of the flattened
        input array will be computed.

    Returns
    -------
    biweight_location : float or `~numpy.ndarray`
        The biweight location of the input data.  If ``axis`` is `None`
        then a scalar will be returned, otherwise a `~numpy.ndarray`
        will be returned.

    See Also
    --------
    biweight_scale, biweight_midvariance, biweight_midcovariance

    References
    ----------
    .. [1] Beers, Flynn, and Gebhardt (1990; AJ 100, 32) (http://adsabs.harvard.edu/abs/1990AJ....100...32B)

    .. [2] http://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/biwloc.htm

    Examples
    --------
    Generate random variates from a Gaussian distribution and return the
    biweight location of the distribution:

    >>> import numpy as np
    >>> from astropy.stats import biweight_location
    >>> rand = np.random.RandomState(12345)
    >>> biloc = biweight_location(rand.randn(1000))
    >>> print(biloc)    # doctest: +FLOAT_CMP
    -0.0175741540445
    """

    data = np.asanyarray(data).astype(np.float64)
    weights = np.asanyarray(weights).astype(np.float64)
    
    weights[~np.isfinite(data)] = np.nan
    
    if (data.shape!=weights.shape):
        raise ValueError("data.shape != weights.shape")

    if M is None:
        M = np.nanmedian(data, axis=axis)
    if axis is not None:
        M = np.expand_dims(M, axis=axis)

    # set up the differences
    d = data - M

    # set up the weighting
    mad = median_absolute_deviation(data, axis=axis, ignore_nan=True)
    #madweights = median_absolute_deviation(weights, axis=axis)

    if axis is None and mad == 0.:
        return M  # return median if data is a constant array
    
    #if axis is None and madweights == 0:
    #    madweights = 1.

    if axis is not None:
        mad = np.expand_dims(mad, axis=axis)
        const_mask = (mad == 0.)
        mad[const_mask] = 1.  # prevent divide by zero

    #if axis is not None:
    #    madweights = np.expand_dims(madweights, axis=axis)
    #    const_mask = (madweights == 0.)
    #    madweights[const_mask] = 1.  # prevent divide by zero

    #cmadsq = (c*mad)**2
    
    #factor = 0.5
    #weights  = weights*cmadsq/madweights*factor # this does not make sense...
    
    u = d / (c * mad)

    # now remove the outlier points
    mask = (np.abs(u) >= 1)
    #print("number of excluded points ", len(mask[mask]))
    
    u = (1 - u ** 2) ** 2
    
    weights = weights**2

    umad = median_absolute_deviation(u, axis=axis, ignore_nan=True)
    wmad = median_absolute_deviation(weights, axis=axis, ignore_nan=True)
    
    if axis is not None:
        umad = np.expand_dims(umad, axis=axis)
        const_mask = (umad == 0.)
        umad[const_mask] = 1.  # prevent divide by zero
        
        wmad = np.expand_dims(wmad, axis=axis)
        const_mask = (wmad == 0.)
        wmad[const_mask] = 1.  # prevent divide by zero
    
    weights = weights * 0.25 * umad / wmad
    
    weights[~np.isfinite(weights)] = 0
    u = u + weights   #**2
    u[weights==0] = 0
    d[weights==0] = 0
    u[mask] = 0

    # along the input axis if data is constant, d will be zero, thus
    # the median value will be returned along that axis
    return M.squeeze() + (d * u).sum(axis=axis) / u.sum(axis=axis)

def biweight_location_weights_karl(data, weights, c=6.0, M=None, axis=None): 
	# now ignores data with zero weight
    data = np.asanyarray(data).astype(np.float64)
    weights = np.asanyarray(weights).astype(np.float64)
   
    data[weights==0] = np.nan 
    weights[~np.isfinite(data)] = np.nan
    if len(weights[np.isfinite(weights)])==0:
        return np.nan
    
    if (data.shape!=weights.shape):
        raise ValueError("data.shape != weights.shape")

    if M is None:
        M = np.nanmedian(data, axis=axis)
    if axis is not None:
        M = np.expand_dims(M, axis=axis)

    # set up the differences
    d = data - M

    # set up the weighting
    mad = median_absolute_deviation(data, axis=axis, ignore_nan=True)
    #madweights = median_absolute_deviation(weights, axis=axis)

    if axis is None and mad == 0.:
        return M  # return median if data is a constant array
    
    #if axis is None and madweights == 0:
    #    madweights = 1.

    if axis is not None:
        mad = np.expand_dims(mad, axis=axis)
        const_mask = (mad == 0.)
        mad[const_mask] = 1.  # prevent divide by zero

    #if axis is not None:
    #    madweights = np.expand_dims(madweights, axis=axis)
    #    const_mask = (madweights == 0.)
    #    madweights[const_mask] = 1.  # prevent divide by zero

    cmadsq = (c*mad)**2
    
    factor = 0.5
    weights  = weights/np.nanmedian(weights)*factor 
    
    u = d / (c * mad)

    # now remove the outlier points
    mask = (np.abs(u) >= 1)
    #print("number of excluded points ", len(mask[mask]))
    
    u = (1 - u ** 2) ** 2
    
    weights[~np.isfinite(weights)] = 0
    
    u = u + weights**2
    u[weights==0] = 0
    d[weights==0] = 0
    u[mask] = 0

    # along the input axis if data is constant, d will be zero, thus
    # the median value will be returned along that axis
    return M.squeeze() + (d * u).sum(axis=axis) / u.sum(axis=axis)    


def biweight_location_weights_karl_old(data, weights, c=6.0, M=None, axis=None):

    data = np.asanyarray(data).astype(np.float64)
    weights = np.asanyarray(weights).astype(np.float64)
    
    weights[~np.isfinite(data)] = np.nan
    
    if (data.shape!=weights.shape):
        raise ValueError("data.shape != weights.shape")

    if M is None:
        M = np.nanmedian(data, axis=axis)
    if axis is not None:
        M = np.expand_dims(M, axis=axis)

    # set up the differences
    d = data - M

    # set up the weighting
    mad = median_absolute_deviation(data, axis=axis, ignore_nan=True)
    #madweights = median_absolute_deviation(weights, axis=axis)

    if axis is None and mad == 0.:
        return M  # return median if data is a constant array
    
    #if axis is None and madweights == 0:
    #    madweights = 1.

    if axis is not None:
        mad = np.expand_dims(mad, axis=axis)
        const_mask = (mad == 0.)
        mad[const_mask] = 1.  # prevent divide by zero

    #if axis is not None:
    #    madweights = np.expand_dims(madweights, axis=axis)
    #    const_mask = (madweights == 0.)
    #    madweights[const_mask] = 1.  # prevent divide by zero

    cmadsq = (c*mad)**2
    
    factor = 0.5
    weights  = weights/np.nanmedian(weights)*factor 
    
    u = d / (c * mad)

    # now remove the outlier points
    mask = (np.abs(u) >= 1)
    #print("number of excluded points ", len(mask[mask]))
    
    u = (1 - u ** 2) ** 2
    
    weights[~np.isfinite(weights)] = 0
    
    u = u + weights**2
    u[weights==0] = 0
    d[weights==0] = 0
    u[mask] = 0

    # along the input axis if data is constant, d will be zero, thus
    # the median value will be returned along that axis
    return M.squeeze() + (d * u).sum(axis=axis) / u.sum(axis=axis)



def biweight_scale(data, c=9.0, M=None, axis=None, modify_sample_size=False,
                   *, ignore_nan=False):
    r"""
    Compute the biweight scale.

    The biweight scale is a robust statistic for determining the
    standard deviation of a distribution.  It is the square root of the
    `biweight midvariance
    <https://en.wikipedia.org/wiki/Robust_measures_of_scale#The_biweight_midvariance>`_.
    It is given by:

    .. math::

        \zeta_{biscl} = \sqrt{n} \ \frac{\sqrt{\sum_{|u_i| < 1} \
            (x_i - M)^2 (1 - u_i^2)^4}} {|(\sum_{|u_i| < 1} \
            (1 - u_i^2) (1 - 5u_i^2))|}

    where :math:`x` is the input data, :math:`M` is the sample median
    (or the input location) and :math:`u_i` is given by:

    .. math::

        u_{i} = \frac{(x_i - M)}{c * MAD}

    where :math:`c` is the tuning constant and :math:`MAD` is the
    `median absolute deviation
    <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_.  The
    biweight midvariance tuning constant ``c`` is typically 9.0 (the
    default).

    For the standard definition of biweight scale, :math:`n` is the
    total number of points in the array (or along the input ``axis``, if
    specified).  That definition is used if ``modify_sample_size`` is
    `False`, which is the default.

    However, if ``modify_sample_size = True``, then :math:`n` is the
    number of points for which :math:`|u_i| < 1` (i.e. the total number
    of non-rejected values), i.e.

    .. math::

        n = \sum_{|u_i| < 1} \ 1

    which results in a value closer to the true standard deviation for
    small sample sizes or for a large number of rejected values.

    Parameters
    ----------
    data : array_like
        Input array or object that can be converted to an array.
        ``data`` can be a `~numpy.ma.MaskedArray`.
    c : float, optional
        Tuning constant for the biweight estimator (default = 9.0).
    M : float or array_like, optional
        The location estimate.  If ``M`` is a scalar value, then its
        value will be used for the entire array (or along each ``axis``,
        if specified).  If ``M`` is an array, then its must be an array
        containing the location estimate along each ``axis`` of the
        input array.  If `None` (default), then the median of the input
        array will be used (or along each ``axis``, if specified).
    axis : `None`, int, or tuple of ints, optional
        The axis or axes along which the biweight scales are computed.
        If `None` (default), then the biweight scale of the flattened
        input array will be computed.
    modify_sample_size : bool, optional
        If `False` (default), then the sample size used is the total
        number of elements in the array (or along the input ``axis``, if
        specified), which follows the standard definition of biweight
        scale.  If `True`, then the sample size is reduced to correct
        for any rejected values (i.e. the sample size used includes only
        the non-rejected values), which results in a value closer to the
        true standard deviation for small sample sizes or for a large
        number of rejected values.
    ignore_nan : bool, optional
        Whether to ignore NaN values in the input ``data``.

    Returns
    -------
    biweight_scale : float or `~numpy.ndarray`
        The biweight scale of the input data.  If ``axis`` is `None`
        then a scalar will be returned, otherwise a `~numpy.ndarray`
        will be returned.

    See Also
    --------
    biweight_midvariance, biweight_midcovariance, biweight_location, astropy.stats.mad_std, astropy.stats.median_absolute_deviation

    References
    ----------
    .. [1] Beers, Flynn, and Gebhardt (1990; AJ 100, 32) (http://adsabs.harvard.edu/abs/1990AJ....100...32B)

    .. [2] https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/biwscale.htm

    Examples
    --------
    Generate random variates from a Gaussian distribution and return the
    biweight scale of the distribution:

    >>> import numpy as np
    >>> from astropy.stats import biweight_scale
    >>> rand = np.random.RandomState(12345)
    >>> biscl = biweight_scale(rand.randn(1000))
    >>> print(biscl)    # doctest: +FLOAT_CMP
    0.986726249291
    """

    return np.sqrt(
        biweight_midvariance(data, c=c, M=M, axis=axis,
                             modify_sample_size=modify_sample_size,
                             ignore_nan=ignore_nan))

def biweight_midvariance(data, c=9.0, M=None, axis=None,
                         modify_sample_size=False, *, ignore_nan=False):
    r"""
    Compute the biweight midvariance.

    The biweight midvariance is a robust statistic for determining the
    variance of a distribution.  Its square root is a robust estimator
    of scale (i.e. standard deviation).  It is given by:

    .. math::

        \zeta_{bivar} = n \ \frac{\sum_{|u_i| < 1} \
            (x_i - M)^2 (1 - u_i^2)^4} {(\sum_{|u_i| < 1} \
            (1 - u_i^2) (1 - 5u_i^2))^2}

    where :math:`x` is the input data, :math:`M` is the sample median
    (or the input location) and :math:`u_i` is given by:

    .. math::

        u_{i} = \frac{(x_i - M)}{c * MAD}

    where :math:`c` is the tuning constant and :math:`MAD` is the
    `median absolute deviation
    <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_.  The
    biweight midvariance tuning constant ``c`` is typically 9.0 (the
    default).

    For the standard definition of `biweight midvariance
    <https://en.wikipedia.org/wiki/Robust_measures_of_scale#The_biweight_midvariance>`_,
    :math:`n` is the total number of points in the array (or along the
    input ``axis``, if specified).  That definition is used if
    ``modify_sample_size`` is `False`, which is the default.

    However, if ``modify_sample_size = True``, then :math:`n` is the
    number of points for which :math:`|u_i| < 1` (i.e. the total number
    of non-rejected values), i.e.

    .. math::

        n = \sum_{|u_i| < 1} \ 1

    which results in a value closer to the true variance for small
    sample sizes or for a large number of rejected values.

    Parameters
    ----------
    data : array_like
        Input array or object that can be converted to an array.
        ``data`` can be a `~numpy.ma.MaskedArray`.
    c : float, optional
        Tuning constant for the biweight estimator (default = 9.0).
    M : float or array_like, optional
        The location estimate.  If ``M`` is a scalar value, then its
        value will be used for the entire array (or along each ``axis``,
        if specified).  If ``M`` is an array, then its must be an array
        containing the location estimate along each ``axis`` of the
        input array.  If `None` (default), then the median of the input
        array will be used (or along each ``axis``, if specified).
    axis : `None`, int, or tuple of ints, optional
        The axis or axes along which the biweight midvariances are
        computed.  If `None` (default), then the biweight midvariance of
        the flattened input array will be computed.
    modify_sample_size : bool, optional
        If `False` (default), then the sample size used is the total
        number of elements in the array (or along the input ``axis``, if
        specified), which follows the standard definition of biweight
        midvariance.  If `True`, then the sample size is reduced to
        correct for any rejected values (i.e. the sample size used
        includes only the non-rejected values), which results in a value
        closer to the true variance for small sample sizes or for a
        large number of rejected values.
    ignore_nan : bool, optional
        Whether to ignore NaN values in the input ``data``.

    Returns
    -------
    biweight_midvariance : float or `~numpy.ndarray`
        The biweight midvariance of the input data.  If ``axis`` is
        `None` then a scalar will be returned, otherwise a
        `~numpy.ndarray` will be returned.

    See Also
    --------
    biweight_midcovariance, biweight_midcorrelation, astropy.stats.mad_std, astropy.stats.median_absolute_deviation

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Robust_measures_of_scale#The_biweight_midvariance

    .. [2] Beers, Flynn, and Gebhardt (1990; AJ 100, 32) (http://adsabs.harvard.edu/abs/1990AJ....100...32B)

    Examples
    --------
    Generate random variates from a Gaussian distribution and return the
    biweight midvariance of the distribution:

    >>> import numpy as np
    >>> from astropy.stats import biweight_midvariance
    >>> rand = np.random.RandomState(12345)
    >>> bivar = biweight_midvariance(rand.randn(1000))
    >>> print(bivar)    # doctest: +FLOAT_CMP
    0.97362869104
    """

    median_func, sum_func = _stat_functions(data, ignore_nan=ignore_nan)

    if isinstance(data, np.ma.MaskedArray) and ignore_nan:
        data = np.ma.masked_where(np.isnan(data), data, copy=True)

    data = np.asanyarray(data).astype(np.float64)

    if M is None:
        M = median_func(data, axis=axis)
    if axis is not None:
        M = _expand_dims(M, axis=axis)  # NUMPY_LT_1_18

    # set up the differences
    d = data - M

    # set up the weighting
    mad = median_absolute_deviation(data, axis=axis, ignore_nan=ignore_nan)

    if axis is None and mad == 0.:
        return 0.  # return zero if data is a constant array

    if axis is not None:
        mad = _expand_dims(mad, axis=axis)  # NUMPY_LT_1_18
        mad[mad == 0] = 1.  # prevent divide by zero

    u = d / (c * mad)

    # now remove the outlier points
    # ignore RuntimeWarnings for comparisons with NaN data values
    with np.errstate(invalid='ignore'):
        mask = np.abs(u) < 1
    if isinstance(mask, np.ma.MaskedArray):
        mask = mask.filled(fill_value=False)  # exclude masked data values

    u = u ** 2

    if modify_sample_size:
        n = sum_func(mask, axis=axis)
    else:
        # set good values to 1, bad values to 0
        include_mask = np.ones(data.shape)
        if isinstance(data, np.ma.MaskedArray):
            include_mask[data.mask] = 0
        if ignore_nan:
            include_mask[np.isnan(data)] = 0
        n = np.sum(include_mask, axis=axis)

    f1 = d * d * (1. - u)**4
    f1[~mask] = 0.
    f1 = sum_func(f1, axis=axis)
    f2 = (1. - u) * (1. - 5.*u)
    f2[~mask] = 0.
    f2 = np.abs(np.sum(f2, axis=axis))**2

    with np.errstate(divide='ignore', invalid='ignore'):
        return n * f1 / f2
