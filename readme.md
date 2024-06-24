# Predicting Glucose Levels on a Continuous Glucose Monitor

An [Erdős Institute](https://www.erdosinstitute.org) [Data Science Bootcamp](https://www.erdosinstitute.org/programs/may-summer-2024/data-science-boot-camp) project

Awarded 1st place out of 51 projects!

Watch our [5-minute recorded presentation](https://video.wixstatic.com/video/b111ac_995a7f902a364fb18ebd4ee2a20d17d5/1080p/mp4/file.mp4)

## Team members
[Noah Gillespie](https://github.com/NoahGillespie)  
[Margaret Swerdloff](https://github.com/mswerdloffNU)   
[Oladimeji Olaluwoye](https://github.com/oladimeji360)  
[S. C. Park](https://github.com/scparkmaths)    
[Daniel Visscher](https://github.com/danielvisscher)


## Project Description
Diabetes is health condition in which the body does not appropriately manage blood glucose levels. <a href="https://www.cdc.gov/diabetes/php/data-research/index.html">According to the CDC</a>, 38.4 million people in the U.S. have diabetes (11.6% of the U.S. population), and 97.6 million people aged 18 years or older have prediabetes (38.0% of the adult U.S. population).

Meanwhile, wearable devices that collect medical data are [increasingly common](https://emedcert.com/blog/wearables-statistics-future-of-healthcare) in the US, with a 2019 survey showing more than half of Americans own one. These devices have the potential to [transform healthcare](https://postgraduateeducation.hms.harvard.edu/trends-medicine/exploring-promise-wearable-devices-further-medical-research) through real-time data collection of a person’s health status during their everyday activities.

The goal of this project is to use data from two wearable medical devices (a Continuous Glucose Monitor (CGM) and a smartwatch) along with patient logged food intake to better model blood glucose levels in prediabetics.

## Dataset
The dataset comes from BIG IDEAs Lab Glycemic Variability and Wearable Device Data: <a href="https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/">https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/</a>

> Study participants (n = 16) with elevated blood glucose in the normal range were monitored with the Dexcom G6 continuous glucose monitors and Empatica E4 wrist-worn wearable devices for 8-10 days.
>
> The Dexcom G6 measures interstitial glucose concentration (mg/dL) every 5 min and the Empatica E4 measures photoplethysmography, electrodermal activity (EDA), skin temperature (TEMP), and tri-axial accelerometry (ACC), resulting in a total of 7 features. PPG was sampled at 64 Hz, providing heart rate (HR) values every second along with a blood volume pulse (BVP) signal from which interbeat interval (IBI) data was computed. EDA and skin temperature were sampled at 4 Hz and accelerometry was sampled at 32 Hz.

The dataset consists of 16 folders, each with eight csv files: the seven features collected by wearable devices (ACC, BVP, Dexcom, EDA, HR, IBI, TEMP) along with a patient documented food log.

***Note:*** The data is not included in this repository due to its size (34.1 GB). To run scripts in this repository, download the data via the link above into a folder "data" inside your copy of the repository.

## Scripts

### ACC_highpass.ipynb

This script includes a few experiments that we used to test assumptions in order to successfully remove the gravity component from the accelerometer data.

### EDA.ipynb

This script includes a range of exploratory data analysis including:

- Introducing the helper functions for importing and preprocessing raw data
- Examining demographics including HbA1C levels
- Examining the relationship between food intake and glucose level
- Observing the instances of high bloodsugar events set by (1) a fixed level determined by each patient's bloodsugar data and (2) quantifying bloodsugar spikes by differenced data

### GI_scraping.ipynb

This script includes a web scraping algorithm to import glycemic index (GI) data from the university of Sydney's website and processing this data, including:

- Scraping data
- Assembling GI dataframe

### LLM_GI.ipynb

This script employs a large language learning model to assign 'low', 'medium', and 'high' GI guesses to each of the foods present in the food log data.

### high_glucose.ipynb

This script employs logistic regression to predict high glucose events from food intake data, including:

- Data import and pre-processing
- Train-test split
- Plotting pair plots for possible feature selection
- Baseline model (Model 0): predicting high glucose events by total carbs
- Model 1: Predicting high glucose events by total carbs and glycemic index
- Model 2: Predicting high glucose events by total carbs, sugar, protein, and total fat
- Model 3: Predicting high glucose events by total carbs, glycemic index, and starting glucose value

### time_series.ipynb

This script employs time series prediction, including: 

- Time series exploration
- Time series prediction