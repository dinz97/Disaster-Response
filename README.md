### Table of Contents
- [1. Project Background](#project-background)
- [2. File Descriptions](#file-descriptions)
- [3. Installation](#installation)
- [4. Program Instructions](#program-execution)
- [5. Acknowledgements](#acknowledgements)

## Project Background
This project aims to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
Using a data set containing real messages that were sent during disaster events, a machine learning pipeline was created to categorize these events so that the messages could be sent to an appropriate disaster relief agency.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## File Descriptions
There are 2 data file and 1 Jupyter notebook available to showcase work related to the above questions.
* Wellbeing_and_lifestyle_data_Kaggle.csv: This dataset contains 15,977 survey responses with 24 attributes describing how respondents live their lives.
* Lifestyle&DailyStressLevel: Jupyter Notebook to prepare data and perform analysis.

## Installations
Standard libraries from Anaconda distribution of Python used to run the code as listed below:
* Pandas
* Numpy
* sys-module
* re

Additional libraries that requires installations:
* [NLTK]()
* [SQLalchemy]()
* [Flask]()
* [Plotly]()

## Program Instructions
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements
Special credit to Figure Eight for providing disaster message data. Special thanks to Udacity Data Scientist Nanodegree Program team for providing a comprehensive project structure!
