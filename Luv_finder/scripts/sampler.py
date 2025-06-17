import copy
import nautilus
import numpy as np
import scipy
import corner
import matplotlib.pyplot as plt
import multiprocessing
import dill

# add a null run
# do a run on jackknifed data
# output catallogs

class luvfinder:
    def __init__(self, uvdata, mod, noise = 'normal'):
        """
        uvdata: SimpleNamespace with uv data (UVreals, UVimags, uvfreqs, etc.)
        mod: Model instance (from modeler.py) with a component already added.
        """
        self.uvdata = copy.deepcopy(uvdata)
        self.mod = copy.deepcopy(mod)
        self.sampler = None

        if noise=='normal':
            self.pdfnoise_real = scipy.stats.norm(loc=self.uvdata.UVreals,
                                                  scale=1/self.uvdata.uvwghts**0.5)
            self.pdfnoise_imag = scipy.stats.norm(loc=self.uvdata.UVimags,
                                                  scale=1/self.uvdata.uvwghts**0.5)
  

    def _model_wrapper(self, theta_input):
        """
        Wrapper for the model function.
        If theta_input is a numpy array (when pass_dict=False), convert it to a dict.
        """
        # Retrieve the keys from the priors built from the model.
        keys = list(self.prior.keys)
        if isinstance(theta_input, np.ndarray):
            # Convert array to dict using the prior ccckeys.
            theta_dict = { key: val for key, val in zip(keys, theta_input) }
        elif isinstance(theta_input, dict):
            theta_dict = theta_input
        else:
            raise ValueError("Unexpected input type for theta_input")
        return theta_dict

    def _get_model(self, theta_input):
        theta_dict = self._model_wrapper(theta_input)
        comp = getattr(self.mod, self.mod.type[0].lower())
        for key, value in theta_dict.items():
            attr = key.split('_', 2)[-1]
            setattr(comp, attr, value)
        return comp.profile(self.uvdata)

    def log_likelihood(self, theta_input): # not using weights
        """
        Compute the log-likelihood for the given parameters.
        theta_dict: dict with keys like 'src_00_dra', etc.
        """ 
        model_uv = self._get_model(theta_input)
        return (self.pdfnoise_real.logpdf(model_uv.UVreals) + self.pdfnoise_imag.logpdf(model_uv.UVimags)).sum()

    def _build_priors(self):
        """Build a Nautilus Prior from the model's priors."""
        prior = nautilus.Prior()
        for key, dist in self.mod.priors.items():
            if dist is not None:
                prior.add_parameter(key, dist=dist)
        self.prior = prior

    def run(self, n_live=2000, dlogz=0.01, pool=None, pass_dict=False, verbose=True):
        """
        Run nested sampling.
        n_live: Number of live points.
        dlogz: Stopping criterion.
        pool: Number of threads; defaults to 25% of available cores.
        pass_dict: If True, the sampler will pass a dictionary to the likelihood.
        """
        if pool is None:
            pool = max(1, int(multiprocessing.cpu_count() * 0.25))
        self._build_priors()

        self.sampler = nautilus.Sampler(self.prior, self.log_likelihood, n_live=n_live, 
                                        pass_dict=pass_dict, pool=pool)
        
        self.sampler.run(f_live=dlogz, verbose=verbose)
        
        samples, log_w, _ = self.sampler.posterior()
        self.samples = samples
        self.log_w = log_w
        self.logz = self.sampler.evidence()
        self.labels = list(self.prior.keys)








    def checkpoint(self, filepath='checkpoint.hdf5'):
        """Save the current state of the sampler."""
        if self.sampler is not None:
            self.sampler.save_checkpoint(filepath)
        else:
            print("Sampler has not been run yet; no checkpoint to save.")

    def dump(self,filename):
        odict = {key: self.__dict__[key] for key in self.__dict__.keys()}
        
        with open(filename,'wb') as f:
            dill.dump(odict,f,dill.HIGHEST_PROTOCOL)

    def load(self,filename):
        with open(filename,'rb') as f:
            odict = dill.load(f)
        self.__dict__.update(odict)

    def getmodel(self):
        p = np.array([np.quantile(samp,0.50) for samp in self.samples.T])
        print(p)
        model_uv = self._get_model(p)
        return model_uv

    def plot_corner(self, filename="./plots/posterior_corner.png", show=False):
        """Save a corner plot of the posterior distribution."""
        fig = corner.corner(
            self.samples,
            weights=np.exp(self.log_w),
            bins=20,
            labels=self.labels,
            color='black',
            plot_datapoints=False,
            # contour_kwargs={"levels": [0.9889]},
            range=np.repeat(0.999, len(self.labels))
        )
        fig.savefig(filename)
        if show:
            plt.show(fig)
        else:
            plt.close(fig)