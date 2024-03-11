# finalCapstone: Sentiment Analysis of Amazon Product Reviews

## Project Overview
The "Sentiment Analysis of Amazon Product Reviews" project utilizes natural language processing to gauge public sentiment on Amazon products. The goal is to categorize product reviews into positive, negative, or neutral sentiments. This project encompasses data preprocessing, sentiment analysis, and validation through similarity checks to gain insights into customer opinions.

## Data Description
- `data/raw`: This directory contains the original dataset, which includes 5000 consumer reviews of various Amazon products.
- `data/processed`: Here you'll find the cleaned and processed data, which has been stripped of duplicates, non-textual content, and irrelevant features, ready for the sentiment analysis.

## Cleaning Process
Our data cleaning process was meticulously outlined and executed to ensure reliable analysis:

- Initial assessment to identify completeness and consistency of the data.
- Exclusion of rows with missing review text to maintain data integrity.
- Removal of duplicates and irrelevant data points to focus the analysis.
- Standardization of text formatting for uniformity.

## Data Analysis and Insights
The analysis was performed using Python scripts:

- `sentiment_analysis.py`: A Python script that preprocesses the reviews data, analyzes sentiment using TextBlob, and validates the results through similarity comparisons using spaCy.
- Sentiment classification and similarity validation insights are documented in the `sentiment_analysis_report.pdf` file within the `docs` folder.

## Tools Used
- Python: For data cleaning, sentiment analysis, and validation.
- spaCy and TextBlob libraries: For natural language processing and sentiment analysis tasks.
- Pandas: For data manipulation and analysis.

## Findings
For a comprehensive summary of our analysis findings and insights into the model's strengths and limitations, please refer to the `sentiment_analysis_report.pdf` within the `docs` directory.

## Usage
To run the sentiment analysis:
1. Ensure Python and necessary libraries are installed.
2. Navigate to the `scripts` directory.
3. Execute the script with the following command:

## Credits
This project was conceived and developed by Nattamon.
- Data provided by Datafiniti's Product Database
