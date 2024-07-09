# Grey-Literature-Classifier

## Building Datasets

### Data Collection

Download Scraped Evidence from provided Excel file, containing studies and relevant evidence/classification.

Download Synopses from (https://www.conservationevidence.com/synopsis/index)[the Conservation Evidence Website].

Download Irrelevant Data from Kacper's zip file.

Run all of src/Preprocessing/scrapers/studies_scraper.ipynb (~40 minutes)

### Data Processing

Run all of irrelevant_data_cleaner.ipynb

Run all of synopses_cleaner.ipynb

Run all of scraped_cleaning.ipynb

Run all of merge_labelled_data.ipynb

This produces the final output in data.json, which can be read by pandas.DataFrame.read_json()

---

## Training

Activate the Docker environment and virtualenv with

cd ..
cd GreyLitDocker
source venv/bin/activate
cd Grey-Literature-Classifier
