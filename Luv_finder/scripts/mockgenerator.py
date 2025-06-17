#!/usr/bin/env python
"""
A class to create mock observations and run CASA simulation tasks.
You can customize hyper‐parameters such as the number of channels, spatial dimensions,
velocity resolution, frequency, sensitivity, and also supply a list of source models.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling import models
from astropy.constants import c
import astropy.units as u
import shutil
import glob
import os

# Import CASA tasks – make sure you run this in an environment where casatasks are available.
from casatasks import simobserve, tclean, exportfits
import scripts.ownAnalysisUtilities as oau  # Custom utilities for white noise sensitivity


class MockObservation:
    def __init__(self,
             cube_shape=(30, 256, 256),
             dv="100km/s",
             freq_center="40GHz",
             cell="0.25arcsec",
             sensitivity=None,
             integration_time=None,
             fits_filename=None,
             sources=None,
             continuum=None,
             direction='J2000 00h00m00.10 -40d00m00.00',
             ptg_file='./configs/ptgfile.txt',
             alma_config='./configs/alma.cycle10.2.cfg',
             output_folder='output/ms_files',
             support_folder='./support'):
        """
        Initialize the simulation parameters.

        Parameters
        ----------
        cube_shape : tuple of int
            Cube dimensions as (n_channels, ny, nx).
        dv : str
            Channel width as a string with units (e.g., "100km/s").
        freq_center : str or Quantity
            Central observing frequency (e.g., "40GHz").
        cell : str
            Imaging cell size as a string with units (e.g., "0.25arcsec").
        sensitivity : str or float
            Per-channel sensitivity in Jy. If provided as a string, include units (e.g., "2e-3Jy").
            This is the baseline sensitivity used to scale source fluxes.
        integration_time : str or float, optional
            Total on-source integration time. If provided, it should include units (e.g., "30min").
            When specified, the sensitivity is computed from the integration time.
            If both sensitivity and integration_time are provided, integration_time takes precedence.
        fits_filename : str, optional
            Filename for the input FITS cube.
        sources : list of dict
            List of source definitions, each specified in physical units:
            - "position": tuple (dra, ddec) in arcsec (RA and Dec offsets from the image center)
            - "axis_min": minor axis (source size) in arcsec (default is the cell size)
            - "axis_maj": major axis (source size) in arcsec (default is the cell size)
            - "line": dict with keys:
                    "width": spectral line width in km/s,
                    "mean": central frequency of the line in GHz,
                    "snr": desired SNR for the line.
            - "continuum": dict with key:
                    "snr": desired integrated SNR for the continuum.
        continuum : dict, optional
            Additional continuum parameters (for backwards compatibility). Source continuum 
            components should be defined within each source in the 'sources' list.
        direction : str
            Pointing direction (e.g., 'J2000 00h00m00.10 -40d00m00.00').
        ptg_file : str
            File path for the pointing file.
        alma_config : str
            Path to the CASA ALMA configuration file.
        output_folder : str
            Directory to which simulation measurement set files will be moved.
        support_folder : str
            Directory where intermediate FITS files are written.

        Raises
        ------
        ValueError
            If neither sensitivity nor integration_time is specified.
        """
        
        self.cube_shape = cube_shape
        self.dv = dv
        self.freq_center = freq_center
        self.cell = cell
        self.fits_filename = fits_filename
        self.sources = sources
        self.continuum = continuum
        self.direction = direction
        self.ptg_file = ptg_file
        self.alma_config = alma_config
        self.output_folder = output_folder
        self.support_folder = support_folder

        self.cube = None
        self.ms_noiseless = None
        self.ms_noisy = None
        
        if integration_time is not None:
            it = u.Quantity(integration_time)
            self.integration_time = it.to(u.min).value
            if sensitivity is not None:
                print("Warning: Both sensitivity and integration_time provided; using integration_time.")
            self.sensitivity = self.estimate_sensitivity(f"{self.integration_time}min")
        elif sensitivity is not None:
            if isinstance(sensitivity, str):
                self.sensitivity = u.Quantity(sensitivity).to(u.Jy).value
            else:
                self.sensitivity = sensitivity
            self.integration_time = self.compute_totaltime()
        else:
            raise ValueError("Error: Either sensitivity or integration_time must be specified.")

    def compute_df(self):
        freq_center_q = u.Quantity(self.freq_center)
        dv_q = u.Quantity(self.dv)
        df = (freq_center_q * dv_q / c).to(u.GHz)
        return df

    def compute_totaltime(self):
        channel_sens = self.sensitivity
        continuum_sens = channel_sens / np.sqrt(self.cube_shape[0])
        totaltime = (0.05e-3 / continuum_sens)**2 * 27
        return totaltime

    def estimate_sensitivity(self, totaltime):
        if isinstance(totaltime, str):
            totaltime = float(totaltime.replace("min", ""))
        nchan = self.cube_shape[0]
        return np.sqrt(27 * (0.05e-3)**2 * nchan / totaltime)

    def create_cube(self):
        nchan, ny, nx = self.cube_shape
        self.cube = np.zeros(self.cube_shape)
        x = np.arange(nchan)
        y = np.arange(ny)
        z = np.arange(nx)
        Y, Z = np.meshgrid(y, z, indexing='ij')

        cell_val = u.Quantity(self.cell).to(u.arcsec).value
        dv_value = u.Quantity(self.dv).to(u.km / u.s).value

        # Compute the frequency channel width (df) in GHz.
        freq_center_q = u.Quantity(self.freq_center)
        df = self.compute_df()

        # Determine the center frequency (in GHz) and the starting frequency of the cube.
        f_center_val = freq_center_q.to(u.GHz).value
        f_start = f_center_val - (nchan / 2) * df.value

        if self.sources is not None:
            for source in self.sources:
                # 'position' is now specified as (dra, ddec) in arcsec.
                dra, ddec = source.get("position", (0.0, 0.0))
                # In the image, y corresponds to declination and x to right ascension.
                pos_y = ny / 2 + (ddec / cell_val)
                pos_z = nx / 2 + (dra / cell_val)

                # Read source size in arcsec; default to cell_val if not provided.
                axis_min = source.get("axis_min", cell_val)  # minor axis (RA direction)
                axis_maj = source.get("axis_maj", cell_val)  # major axis (DEC direction)
                pix_axis_min = axis_min / cell_val
                pix_axis_maj = axis_maj / cell_val

                line_flux_integrated = None
                continuum_flux_integrated = None

                # Process line component if provided.
                if "line" in source:
                    line_conf = source["line"]
                    width_kms = line_conf["width"]      # in km/s
                    f_mean = line_conf["mean"]          # in GHz
                    snr_line = line_conf["snr"]

                    # Convert line width from km/s to number of channels.
                    width_chan = width_kms / dv_value
                    mean_channel = (f_mean - f_start) / df.value

                    # Compute the integrated line flux (in Jy):
                    # sensitivity is per channel, so the integrated flux scales as sensitivity * snr * sqrt(width_chan/2.355*3)
                    flux_line = snr_line * self.sensitivity  * np.sqrt(width_chan/2.355*4)
                    # Amplitude of the spectral Gaussian.
                    amp_spec = flux_line / (width_chan/2.355 * np.sqrt(2 * np.pi))

                    spec_model = models.Gaussian1D(amplitude=amp_spec, mean=mean_channel, stddev=width_chan/2.355)
                    spec = spec_model(x)
                    spatial2d = models.Gaussian2D(
                        amplitude=spec / (2 * np.pi * pix_axis_maj * pix_axis_min),
                        x_mean=pos_y, y_mean=pos_z,
                        x_stddev=pix_axis_maj, y_stddev=pix_axis_min)
                    for i in y:
                        for j in z:
                            self.cube[:, i, j] += spatial2d(i, j)
                   
                    # Convert integrated flux to Jy km/s.
                    line_flux_integrated = flux_line * dv_value

                # Process continuum component if provided.
                if "continuum" in source:
                    cont_conf = source["continuum"]
                    continuum_snr = cont_conf["snr"]
                    # For continuum, per-channel flux required is sensitivity * continuum_snr / sqrt(nchan)
                    flux_cont_per_chan = continuum_snr * self.sensitivity / np.sqrt(nchan)
                    continuum_model = models.Gaussian2D(
                        amplitude=flux_cont_per_chan / (2 * np.pi * pix_axis_maj * pix_axis_min),
                        x_mean=pos_y, y_mean=pos_z,
                        x_stddev=pix_axis_maj, y_stddev=pix_axis_min)
                    continuum_image = continuum_model(Y, Z)
                    self.cube += continuum_image[None, :, :]
                    continuum_flux_integrated = flux_cont_per_chan

                print(f"Source at position (dra={dra} arcsec, ddec={ddec} arcsec) -> pixel position ({pos_y:.2f}, {pos_z:.2f}):")
                if line_flux_integrated is not None:
                    print(f"  Line integrated flux: {line_flux_integrated:.3e} Jy km/s")
                if continuum_flux_integrated is not None:
                    print(f"  Continuum integrated flux: {continuum_flux_integrated:.3e} Jy km/s")
        else:
            raise ValueError("Error: No source is provided")

    def save_cube(self):
        """
        Save the simulated cube to a FITS file.
        """
        if self.cube is None:
            raise RuntimeError("Cube has not been created yet. Call create_cube() first.")

        if self.fits_filename is None:
            # Generate a name based on the SNR values from each source's line and continuum components.
            snr_parts = []
            for src in self.sources:
                if "line" in src:
                    snr_parts.append(f"line{src['line']['snr']}")
                if "continuum" in src:
                    snr_parts.append(f"cont{src['continuum']['snr']}")
            snr_str = "_".join(snr_parts)
            fits_filename = f"{self.support_folder}/{snr_str}_input.fits"
        
        # Write the cube to a FITS file (overwrite if exists).
        fits.writeto(fits_filename, self.cube, overwrite=True)
        self.fits_filename = fits_filename
        print("Cube saved to FITS file:", fits_filename)

    def _update_weights(self):
        oau.getstatwtweights(self.ms_noisy)

    def simulate_observation(self):
        """
        Run the CASA simobserve task using the saved FITS cube.
        This writes the ptg file and sets up the simulation.
        """
        if self.fits_filename is None:
            raise RuntimeError("FITS file not specified. Run save_cube() first.")
        
        df = self.compute_df()
        # Write the pointing file.
        with open(self.ptg_file, 'w') as f:
            f.write(self.direction)

        # Derive a project name from the FITS filename.
        project_name = self.fits_filename.split('_input')[0].split('/')[-1]

        # Run the simobserve task.
        simobserve(project=project_name,
                   skymodel=self.fits_filename,
                   setpointings=False,
                   ptgfile=self.ptg_file,
                   overwrite=True,
                   integration='10s',
                   totaltime=f"{self.integration_time}min",
                   inbright='',
                   comp_nchan=self.cube_shape[0],
                   indirection=self.direction,
                   incell=self.cell,
                   incenter=str(self.freq_center),
                   inwidth=str(df),
                   antennalist=self.alma_config,
                   seed=242,
                   graphics='none')
        print("simobserve complete with project name:", project_name)

        # Set the names for the measurement sets.

        config_name = self.alma_config.split('/')[-1]
        self.ms_noiseless = f"{project_name}/{project_name}." + config_name.replace('cfg', 'ms')
        self.ms_noisy = f"{project_name}/{project_name}." + config_name.replace('cfg', 'noisy.ms')
        self._update_weights()

    def run_imaging(self):
        """
        Run tclean and exportfits on both the noiseless and noisy measurement sets.
        """
        if (self.ms_noiseless is None) or (self.ms_noisy is None):
            raise RuntimeError("Measurement set names not set. Run simulate_observation() first.")

        # Imaging for the noiseless dataset.
        tclean(vis=self.ms_noiseless,
               imagename=self.ms_noiseless.replace('.ms', '.im'),
               niter=0,
               imsize=self.cube_shape[1],
               cell=self.cell,
               gridder='standard',
               weighting='natural',
               specmode='cube',
               parallel=False)
        
        exportfits(imagename=self.ms_noiseless.replace('.ms', '.im.image'),
                   fitsimage=self.ms_noiseless.replace('.ms', '.im.fits'),
                   overwrite=True)
        
        print("Imaging complete for noiseless data.")

        # Imaging for the noisy dataset.
        tclean(vis=self.ms_noisy,
               imagename=self.ms_noisy.replace('.ms', '.im'),
               niter=0,
               imsize=self.cube_shape[1],
               cell=self.cell,
               gridder='standard',
               weighting='natural',
               specmode='cube',
               parallel=False)
        
        exportfits(imagename=self.ms_noisy.replace('.ms', '.im.image'),
                   fitsimage=self.ms_noisy.replace('.ms', '.im.fits'),
                   overwrite=True)
        
        print("Imaging complete for noisy data.")

    def plot_results(self):
        """
        Produce summary plots for each source:
        
        For spectral line components:
        - Determine the channel range covering 3σ of the line,
            where σ = (width_chan/2.355), and sum over these channels.
        - Plot the noiseless and noisy line maps and compute an SNR map.
        - The peak SNR is displayed in the SNR map title.
        
        For continuum components:
        - Compute continuum maps as the mean over all channels.
        - Extract a sub-region around the source and plot the noiseless and noisy continuum maps.
        - Compute and display a continuum SNR map with the peak SNR in the title.
        """
        # Ensure imaging has been performed.
        self.run_imaging()
        if self.ms_noiseless is None or self.ms_noisy is None:
            raise RuntimeError("Measurement set names not set. Run simulate_observation() and run_imaging() first.")

        # Open FITS images.
        fits_noiseless = self.ms_noiseless.replace('.ms', '.im.fits')
        fits_noisy = self.ms_noisy.replace('.ms', '.im.fits')
        hdu_noiseless = fits.open(fits_noiseless)
        hdu_noisy = fits.open(fits_noisy)
        cube_noiseless = hdu_noiseless[0].data[0]  # shape: (nchan, ny, nx)
        cube_noisy = hdu_noisy[0].data[0]
        nchan, ny, nx = cube_noiseless.shape

        # Compute frequency parameters.
        freq_center_q = u.Quantity(self.freq_center)
        dv_q = u.Quantity(self.dv)
        df = (freq_center_q * dv_q / c).to(u.GHz)
        df_val = df.value
        f_center = freq_center_q.to(u.GHz).value
        f_start = f_center - (nchan/2) * df_val
        dv_value = u.Quantity(self.dv).to(u.km/u.s).value

        # Overall continuum maps: average over all channels.
        cont_map_noiseless = np.mean(cube_noiseless, axis=0)
        cont_map_noisy = np.mean(cube_noisy, axis=0)

        if self.sources is None:
            raise ValueError("Error: No source is provided")

        for idx, source in enumerate(self.sources):
            # Derive pixel position from (dra, ddec) in arcsec.
            dra, ddec = source.get("position", (0.0, 0.0))
            cell_val = u.Quantity(self.cell).to(u.arcsec).value
            pos_y = ny/2 + (ddec / cell_val)  # y axis corresponds to declination
            pos_z = nx/2 + (dra / cell_val)   # x axis corresponds to right ascension
            axis_min = source.get("axis_min", cell_val)
            axis_maj = source.get("axis_maj", cell_val)
            pix_axis_min = axis_min / cell_val
            pix_axis_maj = axis_maj / cell_val

            # --- Spectral Line Component ---
            if "line" in source:
                line_conf = source["line"]
                width_kms = line_conf["width"]  # line width in km/s
                f_mean = line_conf["mean"]       # line center in GHz
                # Convert line width to channels.
                width_chan = width_kms / dv_value
                sigma_chan = width_chan / 2.355
                mean_channel = (f_mean - f_start) / df_val
                lower = int(max(0, np.floor(mean_channel - 3 * sigma_chan)))
                upper = int(min(nchan, np.ceil(mean_channel + 3 * sigma_chan)))
                line_map_noiseless = np.sum(cube_noiseless[lower:upper, :, :], axis=0)
                line_map_noisy = np.sum(cube_noisy[lower:upper, :, :], axis=0)
                
                plt.figure()
                plt.imshow(line_map_noiseless, origin='lower')
                plt.title(f"Noiseless Line Map (Source {idx+1}, ch {lower}-{upper})")
                plt.colorbar()
                plt.show()
                
                plt.figure()
                plt.imshow(line_map_noisy, origin='lower')
                plt.title(f"Noisy Line Map (Source {idx+1}, ch {lower}-{upper})")
                plt.colorbar()
                plt.show()
                
                diff_line = line_map_noisy - line_map_noiseless
                std_line = np.nanstd(diff_line)
                snr_map_line = line_map_noiseless / std_line if std_line != 0 else line_map_noiseless
                peak_snr_line = np.nanmax(snr_map_line)
                plt.figure()
                plt.imshow(snr_map_line, origin='lower')
                plt.title(f"Line SNR Map (Source {idx+1}, Peak SNR: {peak_snr_line:.2f})")
                plt.colorbar()
                plt.show()

            # --- Continuum Component ---
            if "continuum" in source:
                half_box = int(3 * max(pix_axis_maj, pix_axis_min))
                y_min = max(0, int(pos_y - half_box))
                y_max = min(ny, int(pos_y + half_box))
                z_min = max(0, int(pos_z - half_box))
                z_max = min(nx, int(pos_z + half_box))
                sub_cont_noiseless = cont_map_noiseless[y_min:y_max, z_min:z_max]
                sub_cont_noisy = cont_map_noisy[y_min:y_max, z_min:z_max]
                
                plt.figure()
                plt.imshow(sub_cont_noiseless, origin='lower')
                plt.title(f"Noiseless Continuum Map (Source {idx+1})")
                plt.colorbar()
                plt.show()
                
                plt.figure()
                plt.imshow(sub_cont_noisy, origin='lower')
                plt.title(f"Noisy Continuum Map (Source {idx+1})")
                plt.colorbar()
                plt.show()
                
                diff_cont = sub_cont_noisy - sub_cont_noiseless
                std_cont = np.nanstd(diff_cont)
                snr_map_cont = sub_cont_noiseless / std_cont if std_cont != 0 else sub_cont_noiseless
                peak_snr_cont = np.nanmax(snr_map_cont)
                plt.figure()
                plt.imshow(snr_map_cont, origin='lower')
                plt.title(f"Continuum SNR Map (Source {idx+1}, Peak SNR: {peak_snr_cont:.2f})")
                plt.colorbar()
                plt.show()
        
        print("Plotting complete.")


    def move_output(self):
        """
        Move the CASA simulation folder (named after the project) into the designated output folder.
        """
        if self.fits_filename is None:
            raise RuntimeError("FITS filename not set; cannot determine project name.")
        project_name = self.fits_filename.split('_input')[0].split('/')[-1]
        destination = f"{self.output_folder}/{project_name}"
        
        # Check if the destination already exists, and if so, remove it.
        if os.path.exists(destination):
            shutil.rmtree(destination)
            print(f"Removed existing destination folder: {destination}")

        shutil.move(project_name, destination)
        print(f"Moved simulation folder '{project_name}' to {self.output_folder}")

        # Remove redundant CASA log files.
        for log_file in glob.glob("*.log"):
            os.remove(log_file)
        print("Removed redundant CASA log files")

    def run_all(self):
        """
        Convenience method to run the complete simulation:
          1. Create cube,
          2. Save cube to FITS,
          3. Run simobserve,
          4. Plot results (and run imaging),
          6. Move the output.
        """
        self.create_cube()
        self.save_cube()
        self.simulate_observation()
        self.plot_results()
        self.move_output()