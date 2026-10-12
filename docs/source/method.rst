How it works
============

Detection is a hypothesis test on the visibilities (Vio & Andreani 2021). For a
trial position :math:`\boldsymbol{\ell}`, rotate the visibilities there and keep
the real part, :math:`x_{jk} = \mathrm{Re}[V_{jk}\,e^{-2\pi i\,\mathbf{u}_{jk}\cdot\boldsymbol{\ell}}]`
for row :math:`j` and channel :math:`k`, and test

.. math::

   H_0:\ \mathbf{x} = \mathbf{n}, \qquad H_1:\ \mathbf{x} = a\,\mathbf{s} + \mathbf{n},
   \qquad s_{jk} = E(\mathbf{u}_{jk})\,g(\nu_k),

with :math:`\mathbf{n}` Gaussian noise of covariance
:math:`\mathbf{C} = \mathrm{diag}(w^{-1})` from the weights, :math:`E` the
visibility amplitude of an elliptical Gaussian source, and :math:`g` a Gaussian
line. The matched filter

.. math::

   T(\mathbf{x}) = \frac{\mathbf{s}^T\mathbf{C}^{-1}\mathbf{x}}{\sqrt{\mathbf{s}^T\mathbf{C}^{-1}\mathbf{s}}}

is :math:`\mathcal{N}(0, 1)` under :math:`H_0` and has mean equal to the line's
S/N under :math:`H_1`. It does not depend on the template's amplitude, so the
response is in S/N units and the flux is not searched.

**Spatial collapse.** The template is separable, so each channel is first summed
over rows, weighting every visibility by the trial source's own amplitude:

.. math::

   y_k = \mathrm{Re}\sum_j E_{jk}\,X_{jk}\,e^{-2\pi i\,\mathbf{u}_{jk}\cdot\boldsymbol{\ell}},
   \qquad W_k = \sum_j w_{jk}\,E_{jk}^2, \qquad X = wV.

For a point source :math:`y_k / W_k` is the naturally weighted dirty image,
computed exactly. On a regular position grid the phase factorises, so every
channel is one complex matrix product (:mod:`luv_finder.kernel`).

**Line and continuum.** At each position the spectrum is fitted as a line
template :math:`\mathbf{g}` on a polynomial continuum :math:`\mathbf{P}`. The
template is projected off the polynomials,
:math:`\mathbf{h} = \mathbf{g} - \mathbf{P}(\mathbf{P}^T\mathbf{W}\mathbf{P})^{-1}\mathbf{P}^T\mathbf{W}\mathbf{g}`,
and normalised by its exact variance under the correlation :math:`\rho` between
channels:

.. math::

   T = \frac{\mathbf{h}^T\mathbf{y}}{\sqrt{\sum_{kk'} h_k h_{k'}\,\rho_{|k-k'|}\sqrt{W_k W_{k'}}}}.

ALMA's Hanning response gives :math:`\rho = (1, 2/3, 1/6)`; it is measured on
the jackknife per spectral window.

**Null hypothesis.** The jackknife differences consecutive integrations of each
baseline, which cancels the sky and keeps the noise. The same grid run on it
gives the correlation between channels, the shape of a source's response, and
the false-alarm rate of the peaks: a blind search selects peaks among many
correlated tests, whose :math:`P_\mathrm{FA}` is not that of a single value.
Candidates need a likelihood ratio
:math:`N_\mathrm{data}(\ge\gamma)/N_\mathrm{jackknife}(\ge\gamma) \ge 3` and a
fidelity :math:`\ge 0.6` (:doc:`guides/catalogue`).

**Mosaics.** Each pointing is divided by its primary beam and the pointings are
combined by inverse variance on one sky grid (:doc:`guides/mosaics`).
