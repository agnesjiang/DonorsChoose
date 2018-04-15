_This is for GW Hippo Hacks April 14, 2018_


## 1. Background

Founded in 2000 by a high school teacher in the Bronx, DonorsChoose.org empowers public school teachers from across the country to request much-needed materials and experiences for their students. At any given time, there are thousands of classroom requests that can be brought to life with a gift of any amount.

DonorsChoose.org receives hundreds of thousands of proposals each year for classroom projects in need of funding. Right now, a large number of volunteers is needed to manually screen each submission before it's approved to be posted on the DonorsChoose.org website.

Next year, DonorsChoose.org expects to receive close to 500,000 project proposals. As a result, there are two main problems they need to solve:

1. How to scale current manual processes and resources to screen 500,000 projects so that they can be posted as quickly and as efficiently as possible
2. How to increase the consistency of project vetting across different volunteers to improve the experience for teachers

### - Our goals

Build an algorithm to pre-screen applications, so that DonorsChoose.org can auto-approve some applications quickly, and volunteers can spend their time on more nuanced and detailed project vetting processes.

If the model is reliable, it can help more teachers get funded more quickly, and with less cost to DonorsChoose.org, allowing them to channel even more funding directly to classrooms across the country.

### - Main technologies

We use **Google Cloud Compute Engine** and **Storage** to deploy a machine learning model in Python 3 and create a prediction application demo that can be used for future data.

### - Datasets

The dataset contains information from teachers' project applications to DonorsChoose.org including teacher attributes, school attributes, and the project proposals including application essays. We need to predict whether or not a DonorsChoose.org project proposal submitted by a teacher will be approved.

## 2. Data Preparation

The main difficulty for data preparation is to create handmade new features. We did it in two main steps, first is to create new variables from the resources dataset. The second one is to use sentiment analysis and text mining (TF-IDF) to find important words that appear frequently in teacher proposal.

## 3. Model

We used xgboost package to create a supervised machine learning model to predict whether an application will be approved. The validate AUC is almost 0.8 which shows a relatively good prediction model. We saved the model in Google Cloud Storage for further prediction in the demo. 

## 4. Demo

Save the data you want to score as "sample_test.csv" with "auto_final.py" in the some folder and enter "python auto_final.py" to shell and the shell will print same basic information about the score data and generate output scoring result as a new csv file (this demo is in demo_auto_score.ipynb) .






