# Introduction

NEXUS (Neuromorphic EXploration of Unknown Spacecraft) is a repository focused on using event cameras (or Neuromorphic cameras) for spacecraft proximity operations research. The team consists of three researchers at Carleton University, one PhD Candidate and two MASc students, each focusing on different areas of research but all united in one goal: developing a robust pipeline to take event camera measurements and transform them into useful ouputs.

# Navigation

This README will be updated as the repository is developed, but for the moment the repository will only contain simple code to records and manipulate event data collected within the Spacecraft Robotics Laboratory.

> [!IMPORTANT]
> All code in this repository will assume that a data file is available locally. Event datasets are quite large and thus cannot be stored on GitHub directly. However, the [EVOS dataset](https://doi.org/10.20383/103.01538) is available for download and contains footage of a rotating and translating spacecraft under different lighting conditions. 

The [import](https://github.com/Carleton-SRL/CNN/tree/main/import) folder in the repository contains Python code to import the AEDAT4 data into an HDF5 dataset or a MAT dataset. There are two scripts which are named according to their purpose. 

> [!CAUTION]
> The code that imports the data into a MAT file is currently only useful for smaller event datasets. For larger datasets, this code can quickly crash your computer. It is generally better to use the HDF5 converter, as HDF5 is supported in both MATLAB and Python.

The [process](https://github.com/Carleton-SRL/CNN/tree/main/process) folder contains code for importing event data into MATLAB or Python, as well as code for simple tasks like splicing out sections of the data or simple visualization tools.

The [record](https://github.com/Carleton-SRL/CNN/tree/main/record) folder contains code for recording event data and synchronizing it with ground truth PhaseSpace data in the SPOT facility.

The [tools](https://github.com/Carleton-SRL/CNN/tree/main/tools) folder contains code that may be useful but is not necessarily specific for event data processing.
