# Thesis_Jean_Gohin

## Name
A Global Analysis of Compound Flood-Drought Events under Climate Change - Master's Thesis

## Description
This project consisted of analyzing a global hydrological model dataset to analyze historical and future occurrences
of Compound Flood-Drought events at a global scale. We specifically used VIC-WUR (https://www.wur.nl/en/research-results/chair-groups/environmental-sciences/earth-systems-and-global-change-group/research/water-climate-food/vic-wur.htm)
simulations of streamflow and soil moisture over historical and future periods, under two climate scenarios (i.e., SSP1-2.6 and
SSP3-7.0) forced by five GCM outputs following the Inter Sectoral Impact Model Intercomparison project 3b protocol. We distinguished 
between three CFD types: Drought & Flood (D&F) events for simultaneous occurrences, and Drought-to-Flood (DtoF) or Flood-to-Drought (FtoD) 
for successive occurrences within at most 6 months. Data is first retrieved, merged and pre-processed from the annuna (HPC) system
(https://wiki.anunna.wur.nl/index.php/Main_Page) DtoF and FtoD events were further classified as slow (>90 days lag), rapid (>30 and < 90 days lag), and abrupt (< 30 days lag) transitions.
 Droughts and floods are first extracted with the drought_flood_extraction scripts. Then, the CFD event types are detected and
their characteristics (frequency, severity, seasonality, drought-flood dependence) analyzed using the data_analysis scripts. 
The data was then summarized ad visualised using the data_visualisation_summaries scripts.

## Requirements
The extraction and analysis scripts are written in Python 3.3 language and require the installation of the following libraries: 
numpy, pandas, xarray, dask, scipy, copulas, warnings. These scripts can be ran on any Integrated Development Environment (IDE).
The visualisation scripts require matplotlib and seaborn and should preferably be ran on a notebook environment such as 
Jupyter notebook.

## Usage
The scripts can be used for future research in this field. To use the scripts, make sure to replace the data sources with your own
at the required places. The drought_flood_extraction scripts should first be run (starting with the historical), to detect drought
and flood events. Then, the analysis scripts can be run in any order.

## Support
For any questions on the scripts, you can send an email at jean.gohin@wur.nl.

## Authors and acknowledgment
This project was carried out by Jean Gohin (master student) under the supervision of Inge de Graaf and Samuel Sutanto

## Project status
This project has been completed.
