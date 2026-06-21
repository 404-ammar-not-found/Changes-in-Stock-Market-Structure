# How did the stock market change from 2004 to 2024?
This repo explores the changes in stock market structure from 2004 to 2024 (inclusive).

## Description of the project

This repo uses the S&P500 to analyse the change in the stock market. All the data is within a folder called data which is ignored due to the size of it however the process of how its made.

## Methodology 

First, I prepared the data through `data_preparation.ipynb`.

Then, I created a correlation heatmap of all the sectors to check whether there is a correlation in `correlation_heatmap.ipynb`.

Finally, I analysed the network of stocks each year through `market_structure_analysis.ipynb`, highlighting its development through the years.

There is quite a bit of explanation in each notebook therefore I will not explain further here to keep it brief.

## Source of the historical constituents

Thank you to this github repo, it was the inspiration of this repo.
https://github.com/hanshof/sp500_constituents

### Disclaimer
I used Codex to rewrite some of the files to be cleaner as they were very messy.
Also do not use this to inform any financial decisions. 
