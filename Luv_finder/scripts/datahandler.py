import numpy as np
from types import SimpleNamespace
import copy
import astropy.constants as c
import astropy.units as u
import os
import shutil

import casatools.msmetadata as msmetadata
import casatools.ms as ms

class DataHandler:
    def __init__(self, file):
        self.msfile = file

        # Create an object to hold the data arrays with dot notation access.
        self.uvdata = SimpleNamespace(
            UVreals = np.empty(0),
            UVimags = np.empty(0),
            uvdists = np.empty(0),
            uvwghts = np.empty(0),
            uvtimes = np.empty(0),
            uvfreqs = np.empty(0),
            uwaves  = np.empty(0),
            vwaves  = np.empty(0),
            jacked  = False
        )
        self.load_data()
        self.metadata = DataHandler.metadata(self.msfile, self.uvdata)
        
    @staticmethod
    def arcsec_to_uvdist(arcsec):
        """Convert an angular scale in arcseconds to a uv-distance in kilolambda."""
        return 1 / np.deg2rad(arcsec / 3600)
    
    @staticmethod
    def n_freqs(uvdata):
        return len(np.unique(uvdata.uvfreqs))

    def n_visbs(self, uvdata):
        return uvdata.UVreals.shape[0] // self.n_freqs(uvdata)

    def load_data(self):
        # Get field and spws from the measurement set metadata
        msmd = msmetadata()
        msmd.open(self.msfile)
        fields = msmd.fieldsforintent("*OBSERVE_TARGET*", False)  # False returns field IDs
        spws = msmd.spwsforintent("*OBSERVE_TARGET*")
        msmd.close()

        # Loop over each field and spw to extract the data
        for field in fields:
            for spw in spws:
                ms_instance = ms()
                ms_instance.open(self.msfile)
                ms_instance.selectinit(reset=True)
                ms_instance.selectinit(datadescid=int(spw))
                ms_instance.select({'field_id': int(field)})


                rec = ms_instance.getdata(['data', 'time', 'u', 'v', 'weight'])
                uvreal = ((rec['data'][0][:].real + rec['data'][1][:].real) / 2.0)
                uvimag = ((rec['data'][0][:].imag + rec['data'][1][:].imag) / 2.0)
                uvwght = 4.0 / (1.0 / rec['weight'][0] + 1.0 / rec['weight'][1])
                
                freqs = ms_instance.range('chan_freq')['chan_freq'][:, 0]
                        
                # Calculate uwave and vwave (u, v coordinates in wavelengths)
                uwave = (rec['u'].reshape(-1, 1) * freqs.reshape(1, -1) / c.c.value).T
                vwave = (rec['v'].reshape(-1, 1) * freqs.reshape(1, -1) / c.c.value).T

                # Create flattened arrays for weights, times, and frequencies
                uvwght = (np.ones_like(uwave) * uvwght.reshape(1, -1)).flatten()
                uvtime = (np.ones_like(uwave) * rec['time'].reshape(1, -1)).flatten()
                uvfreq = (np.ones_like(uwave) * freqs.reshape(-1, 1)).flatten()
                ms_instance.close()
                
                # Append the extracted data to each attribute in uvdata
                self.uvdata.uvdists = np.append(
                    self.uvdata.uvdists,
                    np.sqrt(uwave.flatten()**2 + vwave.flatten()**2) * 1e-3
                )
                self.uvdata.UVreals = np.append(self.uvdata.UVreals, uvreal.flatten())
                self.uvdata.UVimags = np.append(self.uvdata.UVimags, uvimag.flatten())
                self.uvdata.uvwghts = np.append(self.uvdata.uvwghts, uvwght)
                self.uvdata.uvtimes = np.append(self.uvdata.uvtimes, uvtime)
                self.uvdata.uvfreqs = np.append(self.uvdata.uvfreqs, uvfreq)
                self.uvdata.uwaves  = np.append(self.uvdata.uwaves, uwave)
                self.uvdata.vwaves  = np.append(self.uvdata.vwaves, vwave)


    def uv_save(self, output_name, uvdata=None):
        """
        Save the uvdata to a new measurement set file.
        
        Parameters
        ----------
        output_name : str
            Name to append to the original filename for the new MS file.
        uvdata : SimpleNamespace, optional
            Data to save. If None, uses self.uvdata.
        
        Returns
        -------
        str
            Path to the new MS file.
        """
        if uvdata is None:
            uvdata = self.uvdata
        
        # Create output filename
        base_dir = os.path.dirname(self.msfile)
        base_name = os.path.basename(self.msfile)
        new_name = os.path.splitext(base_name)[0] + "_" + output_name + os.path.splitext(base_name)[1]
        new_path = os.path.join(base_dir, new_name)
         
        # Copy the original MS file
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        shutil.copytree(self.msfile, new_path)
        
        # Get field and spws from the measurement set metadata
        msmd = msmetadata()
        msmd.open(new_path)
        fields = msmd.fieldsforintent("*OBSERVE_TARGET*", False)
        spws = msmd.spwsforintent("*OBSERVE_TARGET*")
        msmd.close()
        
        # Keep track of how many visibilities we've processed
        index = 0
        n_freqs = self.n_freqs(uvdata)
        
        # Loop over each field and spw to update the data
        for field in fields:
            for spw in spws:
                ms_instance = ms()
                ms_instance.open(new_path, nomodify=False)
                ms_instance.selectinit(reset=True)
                ms_instance.selectinit(datadescid=int(spw))
                ms_instance.select({'field_id': int(field)})
                
                # Get data to determine the number of visibilities in this selection
                rec = ms_instance.getdata(['data', 'time', 'weight'])
                n_times = rec['time'].shape[0]
                n_chans = rec['data'].shape[1]
                n_visibilities = n_times * n_chans
                
                # Reshape real and imaginary components back to their original shape
                real_reshaped = uvdata.UVreals[index:index+n_visibilities].reshape(n_chans, n_times)
                imag_reshaped = uvdata.UVimags[index:index+n_visibilities].reshape(n_chans, n_times)
                weights_reshaped = uvdata.uvwghts[index:index+n_visibilities].reshape(n_chans, n_times)
                
                # Update the index for the next field/spw
                index += n_visibilities
                
                # Assign the data back to both polarizations
                rec['data'][0][:] = real_reshaped + 1j * imag_reshaped
                rec['data'][1][:] = real_reshaped + 1j * imag_reshaped
                
                # Update weights (assuming we want to use the same weight for both polarizations)
                rec['weight'][0] = weights_reshaped[0, :] / 2
                rec['weight'][1] = weights_reshaped[0, :] / 2
                
                # Put the modified data back
                ms_instance.putdata(rec)
                ms_instance.reset()
                ms_instance.close()
        
        return new_path

    def apply_phase_shift(self, dRA, dDec, uvdata):
        """
        Apply a phase shift to the visibilities based on an offset in RA and Dec.

        Parameters
        ----------
        dRA : float
            Offset in Right Ascension (arcsec).
        dDec : float
            Offset in Declination (arcsec).

        The shifted visibilities are stored and overwrite uvreal
        """
        uvdata = copy.deepcopy(uvdata)

        # Convert the offsets from arcsec to radians
        dRA_rad = np.deg2rad(dRA / 3600.)
        dDec_rad = np.deg2rad(dDec / 3600.)

        # Form complex visibilities from original real and imaginary parts
        vis = uvdata.UVreals + 1j * uvdata.UVimags
        
        # Compute the phase factor (using (u, v) in wavelengths)
        phase_factor = np.exp(-2j * np.pi * (uvdata.uwaves * dRA_rad +
                                              uvdata.vwaves * dDec_rad))
        
        # Apply the phase shift
        new_vis = vis * phase_factor
        
        # Store the phase-shifted visibilities as new attributes
        uvdata.UVreals_shifted = new_vis.real
        uvdata.UVimags_shifted = new_vis.imag

        return uvdata
        
    def jackknife(self, uvdata):
        """
        Perform a jackknife resampling on the provided uvdata.

        This method splits the data into two sets based on the uvtimes
        (using even/odd masks from the unique scans) and then averages
        them (with a sign flip on one half for the real and imaginary parts).

        Parameters
        ----------
        uvdata : SimpleNamespace
            An object with attributes such as UVreals, UVimags, uvdists, uvtimes,
            uvfreqs, uvwghts, uwaves, and vwaves.

        Returns
        -------
        uvdata : SimpleNamespace
            The uvdata after jackknife resampling. Also, an attribute 'jacked' is set to True.
        """
        # Create a shallow copy of uvdata to work on.
        uvdata = copy.deepcopy(uvdata)
        
        # Use uvtimes to define scans and create a mask
        scans, mask = np.unique(uvdata.uvtimes, return_inverse=True)
        mask_neg = (mask % 2).astype(bool)
        mask_pos = ~mask_neg

        # If the number of scans is odd, exclude the last scan from mask_pos
        # This doesn't work if you want to save your jackknifed data back in to the MS file.
        if len(scans) % 2 == 1:
            mask_pos[mask == mask[-1]] = False

        # Combine the two halves. Note the sign flip on the negative half for the visibilities.
        uvdata.UVreals = 0.5 * (
            uvdata.UVreals[mask_neg] * -1 +
            uvdata.UVreals[mask_pos]
        )
        uvdata.UVimags = 0.5 * (
            uvdata.UVimags[mask_neg] * -1 +
            uvdata.UVimags[mask_pos]
        )
        uvdata.uvdists = 0.5 * (
            uvdata.uvdists[mask_neg] +
            uvdata.uvdists[mask_pos]
        )
        uvdata.uvtimes = 0.5 * (
            uvdata.uvtimes[mask_neg ] +
            uvdata.uvtimes[mask_pos]
        )
        uvdata.uvfreqs = 0.5 * (
            uvdata.uvfreqs[mask_neg] +
            uvdata.uvfreqs[mask_pos]
        )
        uvdata.uvwghts = 0.5 * (
            uvdata.uvwghts[mask_neg] +
            uvdata.uvwghts[mask_pos]
        )
        uvdata.uwaves = 0.5 * (
            uvdata.uwaves[mask_neg] +
            uvdata.uwaves[mask_pos]
        )
        uvdata.vwaves = 0.5 * (
            uvdata.vwaves[mask_neg] +
            uvdata.vwaves[mask_pos]
        )

        uvdata.jacked = True
        return uvdata
    
    class metadata:
        def __init__(self, msfile, uvdata):
            self.msfile = msfile
            self.uvdata = uvdata

        def _extract_dish_diameter(self):
            msmd = msmetadata()
            msmd.open(self.msfile)
            d = msmd.antennadiameter()
            msmd.close()
            return d[list(d.items())[0][0]]['value']

        def central_frequency(self):
            return self.uvdata.uvfreqs.mean()

        def primarybeamsize(self, dish_diameter=None):
            frequency = self.central_frequency()
            if dish_diameter is None:
                dish_diameter = self._extract_dish_diameter()
            wavelength = c.c.value / frequency
            hpbeam = 1.22 * wavelength / dish_diameter
            return ((hpbeam * u.rad).to(u.arcsec)).value

        def minresolution(self):
            max_baseline_klambda = self.uvdata.uvdists.max()
            max_baseline_lambda = max_baseline_klambda * 1e3
            res_rad = 1.0 / max_baseline_lambda
            res_arcsec = res_rad * (180.0 / np.pi) * 3600.0
            return res_arcsec


if __name__ == "__main__":
    # Simple test when running the module directly.
    msfile = './output/ms_files/snr5_snr10/snr5_snr10.alma.cycle10.1.ms'
    data_handler = DataHandler(msfile)