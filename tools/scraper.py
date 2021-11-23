from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import openpyxl
import os

DATA_RAW_DIR = '../data/raw/'
REL_EXCEL_FILE_PATH = '../data/source/4000-Stories-with-sentiment-analysis.xlsx'
SHEET_NAME = 'Sheet1'
NEW_FILE_EXTENSION = '.txt'

# toggle min and max text id's to customize interval of raw text generation
MIN_TEXT_ID = 0
MAX_TEXT_ID = 4070
TEXT_ID_CELL_OFFSET = 2


def extract_links_from_excel_file():
    short_stories_excel = openpyxl.load_workbook(REL_EXCEL_FILE_PATH, data_only=True)
    print('loaded .xlsx workbook')
    sheet = short_stories_excel[SHEET_NAME]

    urls = []

    for row in sheet.iter_rows(MIN_TEXT_ID + TEXT_ID_CELL_OFFSET, MAX_TEXT_ID + TEXT_ID_CELL_OFFSET):
        urls.append((row[0].value, row[1].value))
    
    return urls


def scrape_story(url):
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

    driver.get(url[1])
    content = driver.page_source

    soup = BeautifulSoup(content, features="html.parser")

    story_text = ''
    story = soup.find("div", {"class":"jumbotron", "itemtype":"https://schema.org/ShortStory"}).findAll('p')
    print(len(story))

    for p in story:
        story_text += ' ' + ''.join(p.findAll(text = True))

    write_text_to_output(story_text, url[0])


def write_text_to_output(text, file_name):
    if not os.path.exists(DATA_RAW_DIR):
        os.makedirs(DATA_RAW_DIR)

    f = open(DATA_RAW_DIR + str(file_name) + NEW_FILE_EXTENSION, 'w', encoding='utf-8')
    f.write(text)
    f.close()


indexed_urls = extract_links_from_excel_file() # tuple(text_id, url)

for url in indexed_urls:
    scrape_story(url)
