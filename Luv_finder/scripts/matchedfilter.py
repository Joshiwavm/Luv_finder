import copy
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from itertools import product
from tqdm import tqdm
import functools
from functools import partial
from astropy.constants import c
import astropy.units as u

def nu_center_func(width, uvfreq_min):
    uvfreq_min_q = u.Quantity(uvfreq_min, u.Hz)
    return (uvfreq_min_q + 4/2.355 * ((width * u.km/u.s)/c * uvfreq_min_q).to(u.Hz)).value


def _compute_grid_point_response(args):
    """
    Helper function for parallel grid evaluation.
    
    Expects a tuple:
      (params, finder, n_vis, n_freq)
    where:
      - params is the parameter dictionary for this grid point.
      - finder is an instance of matchluvfinder.
      - n_vis and n_freq are the dimensions of the data.uvdata.
    
    Returns:
      A tuple (response, params) where response is the 1D matched filter response.
    """
    params, finder, n_vis, n_freq = args
    model_uv = finder._get_model(params)

    model_uv_shifted = finder.data.apply_phase_shift(
        dRA=params.get('src_00_dra'),
        dDec=params.get('src_00_ddec'),
        uvdata=model_uv)
    
    data_uv_shifted = finder.data.apply_phase_shift(
        dRA=params.get('src_00_dra'),
        dDec=params.get('src_00_ddec'),
        uvdata=finder.data.uvdata)
    
    signal = data_uv_shifted.UVreals_shifted.reshape(n_freq,n_vis)
    kernel = model_uv_shifted.UVreals_shifted.reshape(n_freq,n_vis)
    weight = data_uv_shifted.uvwghts.reshape(n_freq,n_vis)

    # estimate spectra
    signal_mean, weight_mean = np.average(signal, weights = weight, axis=1, returned=True)
    kernel_mean = np.average(kernel, weights = weight, axis=1)

    kernel_norm = kernel_mean * weight_mean / (kernel_mean @ (kernel_mean * weight_mean))**0.5

    # pad
    width_f = (4/2.355 * ((params.get('src_00_width') * u.km/u.s)/c * params.get('src_00_nu_center')*u.Hz).to(u.Hz)).to(u.Hz)
    df = np.unique(np.diff(np.unique(finder.data.uvdata.uvfreqs))) * u.Hz
    pad_size  = int(width_f/df +0.5)
    
    signal_padded = np.pad(signal_mean,pad_width = (0,pad_size*2), mode='reflect')
    kernel_padded = np.pad(kernel_norm,pad_width = (0,pad_size*2), mode='reflect')

    response = finder.delay_transform(signal_padded, kernel_padded) * 0.9 # VOODOOO
    return response[pad_size:-pad_size], params

class matchluvfinder:
    def __init__(self, data, mod, noise='normal'):
        """
        data: Data object containing uvdata (e.g., UVreals, UVimags, uvfreqs, etc.)
              and methods like apply_phase_shift, n_visbs, and n_freqs.
        mod: Model instance with a component already added.
        jackknife: Boolean. If True, the matched filter will be run on both the original and jackknifed data.
        """
        self.data = copy.deepcopy(data)
        self.mod = copy.deepcopy(mod)
        self.response = None
        self.grid_params = None
        self.response_jackknife = None
        self.grid_params_jackknife = None

    @staticmethod
    def delay_transform(signal, kernel):
        signals_fft = np.fft.fft(signal, axis=0)
        kernels_fft = np.fft.fft(kernel, axis=0)
        product_fft = signals_fft * kernels_fft
        conv_result = np.fft.ifft(product_fft, axis=0).real
        return conv_result
        return conv_result.sum(axis=1)
    
    def _get_model(self, theta_dict):
        comp = getattr(self.mod, self.mod.type[0].lower())
        for key, value in theta_dict.items():
            setattr(comp, key.split('_', 2)[-1], value)
        return comp.profile(self.data.uvdata)
    
    def get_response(self, pool=None, uvdata=None):
        """
        Computes the grid search response on the given uvdata.
        If uvdata is not provided, it defaults to self.data.uvdata.
        After computing, it restores the original uvdata.
        Returns:
            response (numpy.ndarray), grid_params (list of dicts)
        """
        if uvdata is None:
            uvdata = self.data.uvdata
        # Backup the original uvdata and temporarily override it.
        original_uvdata = self.data.uvdata
        self.data.uvdata = uvdata

        n_vis = self.data.n_visbs(self.data.uvdata)
        n_freq = self.data.n_freqs(self.data.uvdata)

        # Separate grid parameters into variable, fixed, and function parameters.
        variable_params = {}
        fixed_params = {}
        function_params = {}
        for key, val in self.mod.grid.items():
            if callable(val):
                function_params[key] = val
            else:
                arr = np.atleast_1d(val)
                if arr.size > 1:
                    variable_params[key] = arr
                else:
                    fixed_params[key] = arr.item()

        var_keys = list(variable_params.keys())
        var_values_list = [variable_params[k] for k in var_keys]
        grid_combinations = list(product(*var_values_list))
        
        # Build grid parameters list.
        grid_params_list = []
        for combo in grid_combinations:
            current_params = fixed_params.copy()
            for k, v in zip(var_keys, combo):
                current_params[k] = v
            for key, func in function_params.items():
                if isinstance(func, functools.partial):
                    argcount = func.func.__code__.co_argcount - len(func.args) - (len(func.keywords) if func.keywords is not None else 0)
                else:
                    argcount = func.__code__.co_argcount
                if argcount == 1:
                    candidate = current_params.get('width')
                    if candidate is None:
                        candidate = next((val for k, val in current_params.items() 
                                           if k.endswith('width') and not isinstance(val, dict)), None)
                    if candidate is not None:
                        current_params[key] = func(candidate)
                    else:
                        raise ValueError(f"No candidate parameter found for function {key}")
                else:
                    current_params[key] = func(current_params)
            grid_params_list.append(current_params.copy())

        arg_list = [(params, self, n_vis, n_freq) for params in grid_params_list]

        pool_size = pool if pool is not None else max(1, int(multiprocessing.cpu_count() * 0.25))
        with multiprocessing.Pool(pool_size) as p:
            results = list(tqdm(p.imap_unordered(_compute_grid_point_response, arg_list),
                                total=len(arg_list),
                                desc="Grid Search Progress"))
        responses, grid_params_out = zip(*results)

        # Restore the original uvdata.
        self.data.uvdata = original_uvdata
        return np.array(responses), list(grid_params_out)
    
    def run(self, pool=None, jackknife=False):
        """
        Runs the matched filter grid search.
        If jackknife is True, it runs the grid search on both the original and jackknifed data.
        """
        pool_size = pool if pool is not None else max(1, int(multiprocessing.cpu_count() * 0.25))
        # Compute the normal (original) matched filter response.
        self.response, self.grid_params = self.get_response(pool=pool_size, uvdata=self.data.uvdata)
        
        # If jackknife is enabled, compute the matched filter on the jackknifed data.
        if jackknife:
            self.data.uvdata_split = self.data.jackknife(self.data.uvdata)
            self.response_jackknife, self.grid_params_jackknife = self.get_response(pool=pool_size, uvdata=self.data.uvdata_split)

    def getmodel(self):
        """
        Return the model UV data corresponding to the grid point where the filter response is maximum.
        Uses the normal response (not the jackknife version).
        """
        peak_values = np.max(self.response, axis=1)
        best_idx = np.argmax(peak_values)
        best_params = self.grid_params[best_idx]
        return self._get_model(best_params)
        
    def plot_response(self, filename="./plots/filter_response.png", show=False, vline=None):
        # Determine the index of the grid point with the highest peak response.
        peak_indices = np.max(self.response, axis=1)
        best_idx = np.argmax(peak_indices)
        
        # Compute frequency axis (GHz).
        n_vis = self.data.n_visbs(self.data.uvdata)
        n_freq = self.data.n_freqs(self.data.uvdata)
        freqs = self.data.uvdata.uvfreqs.reshape(n_freq, n_vis)[:, 0] / 1e9

        # Create the figure and axis.
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Plot the response for the best grid point.
        ax.axhline(0, ls='--', color='gray')
        if vline is not None:
            ax.axvline(vline, ls='--', color='gray')
        ax.plot(freqs, self.response[best_idx], lw=2, label="Normal")
        # If jackknife response exists, plot it with a dotted line style.
        if self.response_jackknife is not None:
            ax.plot(freqs, self.response_jackknife[best_idx], lw=2, ls=':', label="Jackknife")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Response Function")
        ax.set_title("Best Grid")
        ax.legend()
        
        # Build a string for the parameter text box, converting nu_center to GHz if needed.
        best_params = self.grid_params[best_idx]
        short_keys = {k.split('_', 2)[-1]: v for k, v in best_params.items()}

        text_lines = []
        for sk, val in short_keys.items():
            if sk == "nu_center":
                continue
            else:
                text_lines.append(f"{sk}={val:.2f}")

        text_str = "\n".join(text_lines)
        
        # Place the parameter text in a box to the right of the plot.
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(
            1.05, 0.5, text_str,
            transform=ax.transAxes,
            va="center",
            bbox=props,
            fontsize=10
        )

        plt.tight_layout()
        plt.savefig(filename)
        if show:
            plt.show()
        plt.close(fig)
