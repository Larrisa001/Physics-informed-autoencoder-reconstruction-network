# Experimental Data

The experimental dataset associated with this work is publicly available on **Zenodo**.

**Dataset DOI:**  
https://doi.org/10.5281/zenodo.22046308

## Dataset Description

The dataset contains **6,001 experimentally acquired speckle images** used for spectrometer calibration and spectral reconstruction.

The input wavelength was scanned from **1520 nm to 1580 nm** with a wavelength interval of **10 pm (0.01 nm)**, resulting in a total of **6,001 wavelength-resolved speckle patterns**.

The wavelength sequence is

\[
\lambda_i = 1520 + 0.01i \quad \mathrm{nm},
\]

where \(i = 0,1,\ldots,6000\).

Therefore, the dataset covers:

- **Wavelength range:** 1520–1580 nm
- **Wavelength interval:** 10 pm (0.01 nm)
- **Number of wavelength points:** 6,001
- **Data type:** Experimentally measured speckle images

## Data Access

Due to the relatively large size of the complete experimental dataset, the raw speckle images are not stored directly in this GitHub repository.

The complete dataset can be downloaded from Zenodo:

https://doi.org/10.5281/zenodo.22046308

This GitHub repository contains the source code and scripts used for data processing, calibration, transmission-matrix construction, and spectral reconstruction.

## Usage

After downloading the dataset from Zenodo, extract the image files to the appropriate data directory specified in the corresponding scripts.

Each speckle image corresponds to one calibrated input wavelength. The wavelength associated with the \(i\)-th image can be determined according to

\[
\lambda_i = 1520 + 0.01i \quad \mathrm{nm}.
\]

Please preserve the original file names and image ordering when using the dataset for transmission-matrix construction or spectral reconstruction.

## Citation

If you use this dataset in your research, please cite the corresponding Zenodo dataset:

> Experimental speckle image dataset, Zenodo.  
> DOI: **10.5281/zenodo.22046308**

Please also cite the associated publication when applicable.

## Data Availability

The experimental data supporting the findings of this study are publicly available in the Zenodo repository at:

https://doi.org/10.5281/zenodo.22046308
