import numpy as np
import scipy.stats
import types
import copy
from astropy.constants import c
import astropy.units as u

class Model:
    def __init__(self):
        self.ncomp  = 0
        self.priors = {}    # Empty dict to hold priors from all components
        self.params = []
        self.paridx = []
        self.profile = []
        self.positive = []
        self.tied = []
        self.type = []
        self.grid = {}      # New attribute for grid search parameters

    def addcomponent(self, prof, positive=None):
        self.type.append(prof.__class__.__name__)
        self.positive.append(prof.positive if positive is None else positive)
        # Iterate over all attributes in the component's __dict__
        for key, value in prof.__dict__.items():
            # Skip non-parameter attributes.
            if key in ['profile', 'positive', 'priors', 'grid']:
                continue
            # Normalize the key by stripping leading underscores.
            norm_key = key.lstrip('_')
            param_name = f'src_{self.ncomp:02d}_{norm_key}'
            self.params.append(param_name)
            # Use the prior defined in the component if available; otherwise use the attribute value.
            if hasattr(prof, 'priors') and norm_key in prof.priors:
                self.priors[param_name] = prof.priors[norm_key]
            else:
                self.priors[param_name] = value
            if isinstance(value, scipy.stats._distn_infrastructure.rv_continuous_frozen):
                self.paridx.append(len(self.params) - 1)
            if isinstance(value, (types.LambdaType, types.FunctionType)):
                self.tied.append(True)
            else:
                self.tied.append(False)
        self.profile.append(prof.profile)
        
        # Add grid information if available in the component.
        if hasattr(prof, 'grid') and prof.grid is not None:
            for key, grid_range in prof.grid.items():
                param_name = f'src_{self.ncomp:02d}_{key}'
                self.grid[param_name] = grid_range

        self.ncomp += 1
        # Set the component as an attribute for easy access.
        setattr(self, self.type[-1].lower(), prof)

    class Gaussian:
        def __init__(self, dra=None, ddec=None, total_flux=None,
                     bmin=None, bmaj=None, nu_center=None, width=None):
            """
            Initialize a Gaussian source model.

            Parameters
            ----------
            dra : float
                Right ascension offset in arcsec (converted to radians).
            ddec : float
                Declination offset in arcsec (converted to radians).
            total_flux : float
                Total integrated flux in Jy km/s.
            bmin : float
                Minor axis (source size) in arcsec (converted to radians).
            bmaj : float
                Major axis (source size) in arcsec (converted to radians).
            nu_center : float
                Central frequency in Hz.
            width : float
                Spectral FWHM in km/s (converted to sigma in Hz).
            """
            self._dra = dra if dra is not None else 0.0
            self._ddec = ddec if ddec is not None else 0.0
            self._bmin = bmin if bmin is not None else 0.0
            self._bmaj = bmaj if bmaj is not None else 0.0
            
            self.total_flux = total_flux if total_flux is not None else 1.0 
            self.nu_center = nu_center if nu_center is not None else 0.0 
            self._width = width if width is not None else 100.

            self.positive = True
            self.profile = self._uvgauss_1D2D
            self.priors = {}
            self.grid = None  # Initialize grid attribute (can be set externally)
        
        @property
        def dra(self):
            # Convert from arcsec to radians on access.
            return (self._dra * u.arcsec).to(u.radian).value

        @dra.setter
        def dra(self, new_dra):
            self._dra = new_dra

        @property
        def ddec(self):
            return (self._ddec * u.arcsec).to(u.radian).value

        @ddec.setter
        def ddec(self, new_ddec):
            self._ddec = new_ddec

        @property
        def bmin(self):
            return (self._bmin * u.arcsec).to(u.radian).value

        @bmin.setter
        def bmin(self, new_bmin):
            self._bmin = new_bmin

        @property
        def bmaj(self):
            return (self._bmaj * u.arcsec).to(u.radian).value

        @bmaj.setter
        def bmaj(self, new_bmaj):
            self._bmaj = new_bmaj

        @property
        def width(self):
            return (self.nu_center * (self._width / 2.355)) / c.to(u.km/u.s).value

        @width.setter
        def width(self, new_width):
            self._width = new_width

        def _uvgauss_1D2D(self, uvdata):
            """
            Evaluate the UV-domain Gaussian model.
            Converts total_flux from Jy km/s to Jy Hz before computing the spectral profile.
            """
            uvdata = copy.deepcopy(uvdata)
            c_km_s = c.to(u.km/u.s).value
            F_int_Hz = self.total_flux * (self.nu_center / c_km_s)
            A = F_int_Hz / (self.width * np.sqrt(2 * np.pi))
            spectral = A * np.exp(-0.5 * ((uvdata.uvfreqs - self.nu_center) / self.width)**2)
            G_spatial = np.exp(-2 * (np.pi**2) * ((self.bmaj * uvdata.uwaves)**2 +
                                                   (self.bmin * uvdata.vwaves)**2)) * \
                        np.exp(2 * np.pi * 1j * (uvdata.uwaves * self.dra +
                                                 uvdata.vwaves * self.ddec))
            uvdata.UVreals = G_spatial.real * spectral.copy()
            uvdata.UVimags = G_spatial.imag * spectral.copy()
            return uvdata
