import pandas as pd
import argparse
from tqdm.auto import tqdm
import os
import json
import requests
import pymupdf
from bs4 import BeautifulSoup

tqdm.pandas()
pymupdf.TOOLS.mupdf_display_warnings(False)
pymupdf.TOOLS.mupdf_display_errors(False)


def bold(string):
    return f'\033[1m{string}\033[0m'

def scrape_studies_func():
    print('Scraping studies...')

    prefix = "https://www.conservationevidence.com/individual-study/" # add number to end, from 2 to 12261
    # skip 1, it's a test page

    def get_study_text(study_number):
        url = prefix + str(study_number)
        page = requests.get(url)
        if page.status_code == 404:
            raise ValueError("Study not found")
        soup = BeautifulSoup(page.content, 'html.parser')

        summary = soup.find_all('section', class_='summary')
        if len(summary) == 0:
            return None

        # get <p> tags out
        text = summary[0].find_all('p')

        # and classes
        summary_classes = summary[0].find_all('td',attrs={'data-head': 'Category'})
        classes = [c.find('img')['alt'] for c in summary_classes if c.find('img') is not None]


        return {'text':[t.get_text() for t in text][1:], 'multiclasses': classes, 'id': study_number}

    def make_request(study_number):
        try:
            return get_study_text(study_number)
        except ValueError as e: # got a 404 error
            return None
        except Exception as e:
            print(f'Error fetching study {study_number}: {e}')
            return None


    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = 12261

    results = [0 for _ in range(n-2)]

    with tqdm(total=n-2) as pbar:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(make_request, i): i for i in range(2,n)}

            for future in as_completed(futures):
                i = futures[future]
                results[i-2] = future.result()
                pbar.update(1)

    print('Fetched studies. Checking integrity...')
    results = [r for r in results if type(r) == dict and type(r['multiclasses']==list)]



    data = pd.DataFrame(results)
    data['url'] = data['id'].apply(lambda x: f'https://www.conservationevidence.com/individual-study/{x}')
    data['relevance'] = ['relevant']*len(data)
    data['text'] = data['text'].apply(lambda x: '\n'.join(x))

    data.drop(columns=['id'], inplace=True)

    print('Saving studies...')
    if not os.path.exists('../../data/unprocessed/studies/'):
        os.makedirs('../../data/unprocessed/studies/')

    data.to_json('../../data/unprocessed/studies/studies.json', orient='records', indent=4)
    print('Raw studies saved.')

def scrape_spreadsheet_func(path: str=None):

    print('Scraping spreadsheet data...')

    if path is None:
        raise FileNotFoundError('No path to spreadsheet data provided. Please export and download scraped data from the spreadsheet first.')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist. Please export and download scraped data from the spreadsheet first.')

    data = pd.read_csv(path,encoding='latin-1')

    data = data[['URL', 'Folder']]
    data['Folder'] = data['Folder'].fillna('No folder')


    def is_valid_url(url):

        i,url=url

        if (
            pd.isna(url)
            or url.startswith("http://archive.jncc.gov.uk")
            or url.startswith("http://www.snh.org.uk")
            or url.startswith("http://www.ices.dk/sites")
            or url.startswith("http://publications.naturalengland.org.uk/file")
            or url.startswith("http://randd.defra.gov.uk")
        ):  # skip empty and certain common URLs where the whole site doesn't respond, or consistently fail

            pbar.update(1)
            return i, False

        try:
            r = requests.head(url,timeout=10)
            pbar.update(1)
            if r.status_code >= 200 and r.status_code < 400: # 200-399 works
                return i, True
            elif r.status_code == 403:
                # print('-----403-----')
                # print(url) # handle manually, since I can still access it
                # print('-----403-----')
                return i,True
            else:
                # print(url, r.status_code)
                return i, False

        except: # 404 and other errors
            pbar.update(1)
            return i, False

    print('Verifying URLs...')
    from concurrent.futures import ThreadPoolExecutor

    urls = data['URL']

    with tqdm(total=len(urls)) as pbar:
        with ThreadPoolExecutor() as executor:
            errors = list(executor.map(is_valid_url, enumerate(urls)))

    errors = list(filter(lambda x: x[1] == False, errors))

    print(f'Removing {len(errors)} invalid URLS')

    data = data.drop(index=[x[0] for x in errors])

    if not os.path.exists('../../data/unprocessed/spreadsheet/'):
        os.makedirs('../../data/unprocessed/spreadsheet/')
    data.to_json('../../data/unprocessed/spreadsheet.json', orient='records', indent=4)


    def crawl_for_data(url): # search for redirects or pdf links to a small depth (i.e links in the page or redirects)

        # check for redirects to pdfs
        r = requests.get(url)
        if (r.url) != url:
            url = r.url
            if "pdf" in url or "download" in url or 'file' in url:
                return url

        # check for pdf links in the page

        html = BeautifulSoup(r.content, 'html.parser')

        for a in html.find_all("a", href=True):
            if "pdf" in a["href"] or "download" in a["href"]:
                return a["href"]


        return url


    def extract_pdf_text(url):
        r = requests.get(url)
        try:
            pdf = pymupdf.open(stream=r.content, filetype='pdf')
        except:
            print('Error opening PDF: ', url)
            return None
        return '\n\n'.join([pdf.get_page_text(i) for i in range(len(pdf))])

    # Some URLs are PDFs or something else parseable, but some just link to a page containing them
    # So this is just a mapping of which ones do that
    # plus some extra 404s that snuck past before but don't work
    # bit hacky

    url_maps = {
        "http://www.gov.scot/Resource/0050/00504418.pdf": "https://www.gov.scot/binaries/content/documents/govscot/publications/research-and-analysis/2014/10/evaluating-assessing-relative-effectiveness-acoustic-deterrent-devices-non-lethal-measures/documents/00504418-pdf/00504418-pdf/govscot%3Adocument/00504418.pdf",
        "http://www.accobams.org/new_accobams/wp-content/uploads/2016/06/ACCOBAMS_MOP2_Res.2.12.pdf": None,
        "http://randd.defra.gov.uk/Default.aspx?Menu=Menu&Module=More&Location=None&ProjectID=19358&FromSearch=Y&Publisher=1&SearchText=LM0443&SortString=ProjectCode&SortOrder=Asc&Paging=10#Descriptionhttp://randd.defra.gov.uk/Document.aspx?Document=14093_LM0443_Resurvey_of_grasslands_2014_FinalReport.pdf": None,
        "https://randd.defra.gov.uk/ProjectDetails?ProjectID=19358&FromSearch=Y&Publisher=1&SearchText=LM0443&SortString=ProjectCode&SortOrder=Asc&Paging=10#Descriptionhttp://randd.defra.gov.uk/Document.aspx?Document=14093_LM0443_Resurvey_of_grasslands_2014_FinalReport.pdf": None,
        "http://ices.dk/sites/pub/Publication%20Reports/Expert%20Group%20Report/acom/2017/WGBYC/wgbyc_2017.pdf": None,
        "http://randd.defra.gov.uk/Default.aspx?Menu=Menu&Module=More&Location=None&ProjectID=14340&FromSearch=Y&Publisher=1&SearchText=MA01031&SortString=ProjectCode&SortOrder=Asc&Paging=10#Descriptionhttp://randd.defra.gov.uk/Document.aspx?Document=13451_MA01031_finalreport.pdf": "https://nora.nerc.ac.uk/id/eprint/505290/1/N505290CR.pdf",
        "http://randd.defra.gov.uk/Document.aspx?Document=MF1003-FINALRevisedAugust2011.pdf": None,
        "https://medwet.org/publications/quelle-occupation-du-sol-au-sein-des-sites-ramsar-de-france-metropolitaine/": None, # in french
    }


    def get_text(url):
        if url in url_maps:
            url = url_maps[url]

        if pd.isna(url) or url is None: # skip invalid URLs
            pbar.update(1)
            return None

        if 'download' in url or 'pdf' in url or 'file' in url: # if it's a pdf
            try:
                text = extract_pdf_text(url)
                pbar.update(1)
                return text
            except:
                url = crawl_for_data(url) # second chance to find a pdf if it's not a direct link
                if 'download' in url or 'pdf' in url or 'file' in url:
                    pbar.update(1)
                    return extract_pdf_text(url)

        else: # second chance to try and find a pdf again
            url = crawl_for_data(url)
            if 'download' in url or 'pdf' in url or 'file' in url: # hopefully it's worked
                pbar.update(1)
                return extract_pdf_text(url)
            else: # no pdf found, take what we can get
                req = requests.get(url)
                soup = BeautifulSoup(req.content, 'html.parser')

                pbar.update(1)
                try:
                    return soup.find('p').get_text()
                except:
                    return None

        pbar.update(1)
        return None # if all else fails, return None


    print('Scraping and extracting PDFs...')


    with tqdm(total=len(data)) as pbar:
        with ThreadPoolExecutor() as executor:
            data["text"] = list(executor.map(get_text, data["URL"]))

    print('Scraped spreadsheet data. Saving...')

    data = data.dropna(subset=['text'])
    data = data.rename(columns={'URL':'url','Folder':'folder'})



    if not os.path.exists('../../data/unprocessed/spreadsheet/'):
        os.makedirs('../../data/unprocessed/spreadsheet/')


    data.to_json('../../data/unprocessed/spreadsheet.json', orient='records', indent=4)



    print('Spreadsheet data saved.')

def clean_irrelevant(path: str=None, limit_irrelevant: float=None, remove_files: bool=False):

    if path is None:
        print('No irrelevant data provided. Skipping...')
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist. Please download irrelevant data first.')

    files = os.listdir(path)
    errors = []
    data = {}
    batches = []

    print('Cleaning irrelevant data.')



    if limit_irrelevant == float('inf'):
        print(f'Loading...')
        uncapped = True # don't stop if we hit the limit due to a bad estimate
        limit_irrelevant = len(files) * len(json.load(open(path + files[0]))['Batch']) # estimate
    else:
        print(f'Loading ~{int(limit_irrelevant)} files...')
        uncapped = False

    with tqdm(files, total=int(limit_irrelevant), leave=True) as pbar:
        for file in files:
            with open(path + file,'r') as f:
                try:
                    new_data = json.load(f)['Batch']
                    batches.extend(new_data)

                    if remove_files:
                        os.remove(path + file)
                    pbar.update(len(new_data))

                    if len(batches) >= limit_irrelevant and not uncapped:
                        break

                except Exception as e:
                    errors.append(file)
                    print('Error loading', file, ':', e)
                    continue


    data = pd.DataFrame(batches,columns=['URL', 'ExtractedTextUntokenized'])

    print(f'\n{len(data)} irrelevant articles loaded. \nRemoving corrupt files...')


    for file in errors:
        try:
            print('Removing', path + file)
            os.remove(path + file)
        except FileNotFoundError:
            continue
    print('Removed', len(errors), 'broken JSON files.')

    import concurrent.futures

    def extract_pdf_text(url):
        try:
            r = requests.get(url)
        except Exception as e:
            print('\nError during request to: ', url)
            print(e)
            pbar.update(1)
            return None

        try:
            pdf = pymupdf.open(stream=r.content, filetype='pdf')
        except Exception as e:
            print('\nError opening PDF: ', url)
            print(e)
            pbar.update(1)
            return None

        pbar.update(1)
        return '\n'.join([pdf.get_page_text(i) for i in range(len(pdf))])

    pdfs = data[data['ExtractedTextUntokenized'].isnull()]
    if len(pdfs) != 0:
        print('Fetching PDFs where not provided...')
        with tqdm(total=len(pdfs)) as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(extract_pdf_text, pdfs['URL'])
                pdfs.loc[:, 'ExtractedTextUntokenized'] = list(results)
        data.update(pdfs)


    data.dropna(subset=['ExtractedTextUntokenized'], inplace=True)
    # drop empty strings too
    data = data[data['ExtractedTextUntokenized'] != '']

    print('Saving irrelevant data...')

    write_data = data[['URL', 'ExtractedTextUntokenized']]
    write_data.columns = ['url', 'text']
    write_data['relevance'] = ['irrelevant']*len(write_data)
    write_data['multiclasses'] = [[]] * len(write_data)


    write_data.to_json('../../data/level-0.5/irrelevant.json', orient='records', indent=4)

    print('Irrelevant data saved.')

def clean_spreadsheet(path: str=None):

    if path is None:
        print('No spreadsheet data provided. Skipping...')
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist. Please scrape spreadsheet data by rerunning this program with the --scrape_spreadsheet flag.')

    data = pd.read_json(path, orient='records')


    class_name_map = {
        "No folder": "No Folder",
        "1. Amphibians": "Amphibians",
        "2. Birds": "Birds",
        "3. Fish": "Fish",
        "4. Invertebrates": "Invertebrates",
        "5. Marine Invertebrates": "Marine Invertebrates",
        "6. Mammals": "Mammals",
        "7. Reptiles": "Reptiles",
        "8. Animals ex-situ": "Animals Ex-Situ",
        "9. Individual plants & algae": "Plants and Algae",
        "9. Indiviual plants & algae": "Plants and Algae",  # note mispelling
        "10. Plants ex-situ": "Plants Ex-Situ",
        "11. Fungi": "Fungi",
        "12. Bacteria": "Bacteria",
        "13. Coastal": "Coastal",
        "14. Farmland": "Farmland",
        "15. Forests": "Forests",
        "16. Rivers, lakes": "Rivers and Lakes",
        "16. Rivers,lakes": "Rivers and Lakes",
        "17. Grassland": "Grassland",
        "18. Marine": "Marine",
        "19. Shrubland": "Shrubland",
        "20. Wetlands": "Wetlands",
        "21. Invasive amphibians": "Invasive Amphibians",
        "22. Invasive birds": "Invasive Birds",
        "23. Invasive fish": "Invasive Fish",
        "24. Invasive invertebrates": "Invasive Invertebrates",
        "24. Invasive inverts": "Invasive Invertebrates",  # note abbreviation
        "25. Invasive mammals": "Invasive Mammals",
        "26. Invasive reptiles": "Invasive Reptiles",
        "27. Invasive plants": "Invasive Plants",
        "28. Invasive fungi": "Invasive Fungi",
        "29. Invasive bacteria": "Invasive Bacteria",
        "30. Behaviour Change": "Behaviour Change",
    }

    data.insert(0,'multiclasses',data["folder"].map(lambda x: class_name_map[x.strip()]))
    data['multiclasses'] = data['multiclasses'].apply(lambda x: [x] if x != 'No Folder' else [])
    data['relevance'] = ['relevant']*len(data)

    print('Saving spreadsheet data...')

    if not os.path.exists('../../data/level-0.5/spreadsheet/'):
        os.makedirs('../../data/level-0.5/spreadsheet/')

    data.to_json('../../data/level-0.5/spreadsheet/spreadsheet.json', orient='records', indent=4)

    print('Spreadsheet data saved.')

def clean_synopses(path: str=None):

    print('Cleaning synopses...')

    if path is None:
        print('No synopses data provided. Skipping...')
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist. Please download synopses data from https://www.conservationevidence.com/synopsis/index first.')

    # note that PDFs are pre-downloaded from the Conservation Evidence synopses page.
    pdf_files = []

    for root, dirs, files in os.walk(path):
        if root == path:
            continue
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file).replace('\\', '/'))


    multiclass_map = {
        'Bird': ['Birds'],
        'Farmland': ['Farmland'],
        'Natural Pest Control': ['Pests'],
        'Control of Freshwater Invasive Species': ['Fish','Invasive'],
        'Shrubland and Heathland': ['Shrubland'],
        'Reptile': ['Reptiles'],
        'Terrestrial Mammal': ['Mammals'],
        'Marsh and Swamp': ['Wetlands'],
        'Grassland': ['Grassland'],
        'Bat': ['Bats'],
        'Amphibian': ['Amphibians'],
        'Bee': ['Insects'],
        'Butterfly and Moth': ['Insects'],
        'Forest': ['Forests'],
        'Primate': ['Mammals'],
        'Peatland': ['Wetlands'],
        'Mediterranean Farmland': ['Farmland'],
        'Subtidal Benthic Invertebrate': ['Marine Invertebrates'],
        'Marine and Freshwater Mammal': ['Mammals', 'Marine','Rivers and Lakes'],
        'Management of Captive Animals': ['Animals Ex-Situ', 'Captivity'],
        'Soil Fertility': ['Farmland'],
        'Sustainable Aquaculture': ['Marine'],
        'Marine Fish': ['Marine','Fish'],
        'Biodiversity of Marine Artificial Structures': ['Marine','Plants and Algae'],
        'Invasive Freshwater Species': ['Fish','Invasive','Rivers and Lakes'],
        'Sustainable Farming': ['Farmland'],
        'Marine Fish': ['Marine','Fish'],
    }


    def parse_pdf(filepath):
        pdf = pymupdf.open(filepath)
        text = ""
        for page in pdf:
            text += page.get_text()

        folder = filepath.split("synopses/")[1].split('/')[1].removesuffix('.pdf')

        pdf.close()
        return {
            'relevance': 'relevant',
            "text": text,
            "multiclasses": set(multiclass_map[folder]),
            "url": 'https://www.conservationevidence.com/synopsis/index',
        }

    print('Parsing Synopses PDFs...')

    records = []

    for i,filepath in tqdm(enumerate(pdf_files),total=len(pdf_files)):
        record = parse_pdf(filepath)


        records.append(record)

    print('Finished parsing. Saving synopses data...')

    data = pd.DataFrame(records)

    if not os.path.exists('../../data/level-0.5/synopses/'):
        os.makedirs('../../data/level-0.5/synopses/')

    data.to_json('../../data/level-0.5/synopses/synopses.json', orient='records', indent=4)

    print('Synopses data saved.')

def clean_studies(path: str=None):

    print('Cleaning studies...')

    if path is None:
        print('No studies data provided. Skipping...')
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist. Please scrape data by rerunning this program with the --scrape_studies flag.')

    data = pd.read_json(path, orient='records')
    data = data[data['text'].apply(lambda x: x!='No summary for this intervention.')] # remove empty summaries

    multiclass_map = {
        'Bird Conservation': ['Birds'],
        'Farmland Conservation': ['Farmland'],
        'Natural Pest Control': ['Pests'],
        'Control of Freshwater Invasive Species': ['Fish','Invasive'],
        'Shrubland and Heathland Conservation': ['Shrubland'],
        'Reptile Conservation': ['Reptiles'],
        'Terrestrial Mammal Conservation': ['Mammals'],
        'Marsh and Swamp Conservation': ['Wetlands'],
        'Grassland Conservation': ['Grassland'],
        'Bat Conservation': ['Bats'],
        'Amphibian Conservation': ['Amphibians'],
        'Bee Conservation': ['Insects'],
        'Butterfly and Moth Conservation': ['Insects'],
        'Forest Conservation': ['Forests'],
        'Primate Conservation': ['Mammals'],
        'Peatland Conservation': ['Wetlands'],
        'Mediterranean Farmland': ['Farmland'],
        'Subtidal Benthic Invertebrate Conservation': ['Marine Invertebrates'],
        'Marine and Freshwater Mammal Conservation': ['Mammals', 'Marine','Rivers and Lakes'],
        'Management of Captive Animals': ['Animals Ex-Situ', 'Captivity'],
        'Soil Fertility': ['Farmland'],
        'Sustainable Aquaculture': ['Marine'],
        'Marine Fish Conservation': ['Marine','Fish'],
        'Biodiversity of Marine Artificial Structures': ['Marine','Plants and Algae']
        }

    print('Remapping classes...')

    def remap_classes(classes):
        new_classes = []
        for c in classes:
            new_classes.extend(multiclass_map[c])
        return set(new_classes)

    data['multiclasses'] = data['multiclasses'].progress_apply(remap_classes)

    print('Classes mapped. Saving studies...')

    if not os.path.exists('../../data/level-0.5/studies/'):
        os.makedirs('../../data/level-0.5/studies/')
    data.to_json('../../data/level-0.5/studies/studies.json', orient='records', indent=4)

def merge_relevant():

    print('Merging relevant data...')
    path = '../../data/level-0.5/'

    data = []


    for subdir in os.listdir(path):
        if os.path.isdir(os.path.join(path, subdir)):
            print(f'Loading {subdir}...')
            data.append(pd.read_json(os.path.join(path, subdir,f'{subdir}.json'),orient='records'))


    data = pd.concat(data)

    write_data = data[['url','text','relevance','multiclasses']].dropna(subset=['multiclasses'])

    print('Saving merged data...')

    write_data.to_json('../../data/level-0.5/relevant.json', orient='records', indent=4)

    print('Merged data saved.')

def merge_all():
    relevant = pd.read_json('../../data/level-0.5/relevant.json', orient='records')
    irrelevant = pd.read_json('../../data/level-0.5/irrelevant.json', orient='records')

    data = pd.concat([relevant,irrelevant])

    data.to_json('../../data/level-0.5/data.json', orient='records', indent=4)



def main(scrape_studies: bool = False,
         scrape_spreadsheet: bool = False,
         raw_spreadsheet_path: str = None,
         use_default_paths: bool = False,
         irrelevant_path: str = None,
         spreadsheet_path: str = None,
         synopses_path: str = None,
         studies_path: str = None,
         **kwargs):

    print('\n\n')


    if raw_spreadsheet_path is not None and not scrape_spreadsheet:
        raise ValueError('Please provide the --scrape_spreadsheet flag to scrape the spreadsheet data, or remove the raw_spreadsheet_path argument.')



    if irrelevant_path is not None and irrelevant_path != 'None' and not os.path.exists(irrelevant_path):
        raise FileNotFoundError(f'{irrelevant_path} does not exist. Please provide a valid path to the irrelevant data.')
    elif spreadsheet_path is not None and spreadsheet_path != 'None' and not os.path.exists(spreadsheet_path):
        raise FileNotFoundError(f'{spreadsheet_path} does not exist. Please provide a valid path to the spreadsheet data.')
    elif synopses_path is not None and synopses_path != 'None' and not os.path.exists(synopses_path):
        raise FileNotFoundError(f'{synopses_path} does not exist. Please provide a valid path to the synopses data.')
    elif studies_path is not None and studies_path != 'None' and not os.path.exists(studies_path):
        raise FileNotFoundError(f'{studies_path} does not exist. Please provide a valid path to the studies data.')




    if kwargs.get('remove_files', False):
        from time import sleep
        print(f'\n{bold("************ WARNING ************")}\n\n')
        print(f'You are about to clean irrelevant data. This process will remove all irrelevant data from the provided path, and combine them into a single file in level-0.5. ')
        print(f'If the original batch files are not backed up, {bold("they will be lost")}. Only proceed if this is intended.')
        print(f'\nAre you sure you want to proceed? The process will begin in 5 seconds; {bold("terminate it now if needed")}.')
        print(f'\n\n {bold("********************************")}\n\n')
        sleep(5)



    #* SCRAPING

    if scrape_studies:
        scrape_studies_func() # creates data/unprocessed/raw_studies.json
        print('\n\n')
    if scrape_spreadsheet:
        if use_default_paths:
            raw_spreadsheet_path = '../../data/unprocessed/raw-grey-literature-sources.csv'

        scrape_spreadsheet_func(raw_spreadsheet_path) # creates data/unprocessed/spreadsheet/raw_spreadsheet.json
        print('\n\n')


    if use_default_paths:
        print('Using default paths:')
        check_str_none = lambda path: None if path == 'None' else path # lets you explicitly set a path to None and skip it, or explicitly set a path to
                                                                        # a string and override defaults

        irrelevant_path = '../../data/unprocessed/irrelevant/' if irrelevant_path is None else check_str_none(irrelevant_path)
        synopses_path = '../../data/unprocessed/synopses/' if synopses_path is None else check_str_none(synopses_path)
        spreadsheet_path = '../../data/unprocessed/spreadsheet.json' if spreadsheet_path is None else check_str_none(spreadsheet_path)
        studies_path = '../../data/unprocessed/studies.json' if studies_path is None else check_str_none(studies_path)
        print('Irrelevant:', irrelevant_path)
        print('Spreadsheet:', spreadsheet_path)
        print('Synopses:', synopses_path)
        print('Studies:', studies_path)
        print('\n\n')




    if kwargs.get('only_irrelevant', False):
        clean_irrelevant(irrelevant_path, kwargs.get('limit_irrelevant')) # creates data/level-0.5/irrelevant.json
        return

    #* CLEANING
    clean_irrelevant(irrelevant_path, kwargs.get('limit_irrelevant')) # creates data/level-0.5/irrelevant.json
    print('\n\n')
    clean_spreadsheet(spreadsheet_path) # creates data/level-0.5/scraped/scraped.json
    print('\n\n')
    clean_synopses(synopses_path) # creates data/level-0.5/synopses/synopses.json
    print('\n\n')
    clean_studies(studies_path) # creates data/level-0.5/studies/studies.json
    print('\n\n')


    #* MERGING

    merge_relevant() # creates data/level-0.5/relevant.json
    print('\n\n')

    merge_all() # creates data/level-0.5/data.json
    print('\n\n')




parser = argparse.ArgumentParser()
# parser.set_defaults(func=main)
parser.add_argument('--scrape-studies', action='store_true', help='Scrape studies from Conservation Evidence website, and save raw format before cleaning.')
parser.add_argument('--scrape-spreadsheet', action='store_true', help='Scrape action evidence from spreadsheet, and save raw format before cleaning.')
parser.add_argument('--raw-spreadsheet-path', type=str, help='Path to raw spreadsheet data. Only used if --scrape-spreadsheet is set.')
parser.add_argument('--irrelevant-path', type=str, help='Path to irrelevant data. Skipped if not provided and --use-default-paths is not set.')
parser.add_argument('--spreadsheet-path', type=str, help='Path to scraped data. Skipped if not provided and --use-default-paths is not set.')
parser.add_argument('--synopses-path', type=str, help='Path to synopses data. Skipped if not provided and --use-default-paths is not set.')
parser.add_argument('--studies-path', type=str, help='Path to studies data. Skipped if not provided and --use-default-paths is not set.')
parser.add_argument('--use-default-paths', action='store_true', help='Use default paths for data sources. Override by providing explicit paths.')
parser.add_argument('--remove-files', action='store_true', help='Remove irrelevant data files after cleaning.')
parser.add_argument('--only-irrelevant', action='store_true', help='Only clean irrelevant data. Use if preprocessing data for pretrained model.')
parser.add_argument('--limit-irrelevant', type=float, default=float('inf'), help='Number of irrelevant samples to clean. Default is uncapped.')


if __name__ == '__main__':

    # handle parser arguments
    args = parser.parse_args()

    # call main
    main(**vars(args))

