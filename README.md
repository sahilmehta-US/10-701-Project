# 10-701-Project

## Overview

## Install

To install the required dependencies, first make sure python is installed in your preferred environment. Then run the following command in the root directory:

```
python -m pip install -r requirements.txt
```

Secondly, write a `.env` file in the data folder with the following API keys:

- `FRED_API_KEY`: an API Key to fetch Federal Reserve Economic Data from the [Federal Reserve Bank of St. Louis](fred.stlouisfed.org)

Note that currently we only use the yfinance data for the project. If time permits, we will add the fred data to the project.

## Get the yfinance data

To get the yfinance data, run the following command in the data folder:

```
python yfinance_data.py
```

This will download the yfinance data and run all preprocessing steps. The intermediate files will be saved in the pipeline_steps folder. The files to be used by downstream tasks will be saved in the results folder.
