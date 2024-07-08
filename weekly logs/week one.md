## Day 1

-   Working with Kacper on project; he'll shortlist potential useful candidates and format into a JSON file
-   I'll analyse the JSON for evidence and determine whether the text is useful
-   If possible, what it's useful for

Modular approach, prototyping in Python. Work upwards from simple all-false classifier.

Create quick interface and testing setup with .ipynb file. Fake some data and start experimenting with nltk.

Obtained and cleaned up grey-literature dataset; deleted a few rows with insufficient data
Validated URLs, replacing with NaN

Formatted dataset into JSON form

Used Vectorizers with sklearn for Naive Bayes and Logistic Regression. It works in theory - perhaps alter Logistic Regression boundaries/offset?

Began working on transforming URLs to actual text, so I can test better. Also need to gather irrelevant data.

## Day 2

Finished cleaning up the dataset - got about 100 valid pieces of relevant data. Irrelevant data can always be randomly found later on.
Might still have to look out for duplicates, but that can be done later.

Touched up Bayes and Logistic Regression, and made a quick stratified classifier as a baseline comparison.

Added some useful functions in utils, changed the way I'm importing data.

Starting to build evaluation pipelines for models, so I can more directly compare them. I'll still need good data for that, but it's a start.
Changed the Classifiers to use feature/label arguments.

Evaluated classifiers with Accuracy, F1, and CrossValidation matrices. They're still quite inaccurate for classifying what a piece of grey literature relates to; I'm guessing it's due to a very small amount of data for each category (~20 folders and ~100 datapoints).

May have to scrape the Conservation Evidence website for more data - or at least try reading their RTF format exports.

Going by the Conservation Evidence website, the classes are:

Amphibians
Birds
Fish
Invertebrates
Marine Invertebrates
Mammals
Reptiles
Animals Ex-Situ
Plants and Algae
Plants and Algae
Plants Ex-Situ
Fungi
Bacteria
Coastal
Farmland
Forests
Rivers and Lakes
Grassland
Marine
Shrubland
Wetlands
Invasive Amphibians
Invasive Birds
Invasive Fish
Invasive Invertebrates
Invasive Invertebrates
Invasive Mammals
Invasive Reptiles
Invasive Plants
Invasive Fungi
Invasive Bacteria
Behaviour Change

Note a clear divide between species and habitats
Could be useful to split them into two separate classes before using a more specific classifier?

Anyways, to add more data I've taken the synopses and grouped them into each category. I'll try to do the same with the Studies (since there's so many), though that'll definitely need to be automated.

## Day Three

Started scraping the Conservation Evidence website and using the SQL Database for extra data
Added better class labelling. Once I can get irrelevant data from common-crawl, I can focus on it; for now I'll keep going on the separate classes.

Scikit-learn's a little slow already, so I'll switch over to spaCy, a much faster NLP library.
Installed spaCy's en_core_web_sm version for faster but less accurate results during development; for production, use en_core_web_trf.
I did want to make use of my GPU, but Cuda was playing up - I'll fix it later and just start developing on CPU for now.

SpaCy takes a lot more work than realised. I've set up configs and a preprocessor, along with a slightly customised pipeline. It's very messy, though - the next step is to reorganise the entire repository. I'm going to pull out spaCy into its own folder, and work solely with it from there.

It needs its own data, and train/dev/test separated into files before running on the command line (or a script if needed). I'll make these functions tomorrow.
