# Predicting Glucose Levels on a Continuous Glucose Monitor

## Team members
[Noah Gillespie](https://github.com/NoahGillespie)
[Margaret Swerdloff](https://github.com/mswerdloffNU)
[Oladimeji Olaluwoye](https://github.com/oladimeji360)
[S. C. Park](https://github.com/scparkmaths)
[Daniel Visscher](https://github.com/danielvisscher)


## Project Description
Diabetes is health condition in which the body does not appropriately manage blood glucose levels. According to the <a href="https://www.cdc.gov/diabetes/php/data-research/index.html">CDC</a>, 38.4 million people in the U.S. have diabetes (11.6% of the U.S. population), and 97.6 million people aged 18 years or older have prediabetes (38.0% of the adult U.S. population).

A Continuous Glucose Monitor (CGM) is a wearable medical device that records interstitial glucose every 5 minutes. Interstitial glucose is a good but not perfect substitute for blood glucose, but while blood glucose is measured by a more invasive finger poke and hence collected only a handful of times per day, a CGM can collect 12*24 = 288 data points per day. This provides a great opportunity to use data science techniques to learn about issues related to diabetes and diabetes management.

This project works with a dataset containing CGM data for 16 patients over about 10 days, along with a food intake log and other physical data (heart rate, skin temperature, accelerometry...). Our goal is to produce a model for one component of glucose variability: spikes in glucose levels after eating. 

## Dataset
BIG IDEAs Lab Glycemic Variability and Wearable Device Data: <a href="https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/">https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/</a>

* There were 16 study participants with 10 days of data for each.
* Data collected: tri-axial accelerometry, blood volume pulse, interstitial glucose concentration, EDA, HR, IBI, skin temperature, and Food Log.
