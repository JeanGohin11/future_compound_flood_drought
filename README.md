# Compound Flood Droughts

## Name
A Global Analysis of Compound Flood-Drought Events under Climate Change 

## Description
This project consisted of analyzing a global hydrological model dataset to quantify historical and future occurrences
of Compound Flood-Drought events at a global scale. We specifically used VIC-WUR (https://www.wur.nl/en/research-results/chair-groups/environmental-sciences/earth-systems-and-global-change-group/research/water-climate-food/vic-wur.htm)
simulations of daily streamflow and soil moisture over historical and future periods, under two climate scenarios (i.e., SSP1-2.6 and
SSP3-7.0) forced by five GCM outputs following the Inter Sectoral Impact Model Intercomparison project 3b protocol. We distinguished 
between three CFD types: Drought & Flood (D&F) events for simultaneous occurrences, and Drought-to-Flood (DtoF) or Flood-to-Drought (FtoD) 
for successive occurrences within at most 6 months. DtoF and FtoD events were further classified as slow (>90 days lag), rapid (>30 and < 90 days lag), 
and abrupt (< 30 days lag) transitions. Data is first retrieved, merged and pre-processed from the annuna (HPC) system
(https://wiki.anunna.wur.nl/index.php/Main_Page).  Droughts and floods are first extracted with the drought_flood_extraction 
scripts. Then, the CFD event types are detected and their characteristics (frequency, severity, seasonality, drought-flood dependence) 
analyzed using the data_analysis scripts. The data was then summarized and visualised using the data_visualisation_summaries scripts.

## Installation
The extraction and analysis scripts are written in Python 3.11 language and require the installation of the following libraries: 
numpy, pandas, xarray, dask, scipy, copulas, warnings. These scripts can be ran on any Integrated Development Environment (IDE) 
such as Pycharms or Visual Studio Code. The visualisation scripts require matplotlib and seaborn and should preferably be 
ran on a notebook environment such as Jupyter notebook.

## Usage
To use the scripts, follow these steps:
- The pre-processing scripts are only relevant for this specific project, and can be ignored.
- make sure to replace the data sources with your own (this is always at the beginning of scripts). For the codes to work, 
the data should be in netCDF format, with dimensions time, lat and lon.
- The drought_flood_extraction scripts can then be run (starting with the historical), to detect drought and flood events. 
- Then, the analysis scripts can be run in any order, dependending on the goal of the analysis.
- Finally the data_visualisation_summaries scripts can be run on a notebook environment, where the data file names should be replaced


## Support
For any questions on the scripts, you can send an email at jeangohin54@gmail.com.

## Authors and acknowledgment
The codes were written by Jean Gohin (master student) with the help of Generative AI for debugging and improvements,
under the supervision of Inge de Graaf and Samuel Sutanto.

## Project status
This project has been completed.
