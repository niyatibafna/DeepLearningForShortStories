from selenium import webdriver
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import openpyxl


def extract_links_from_excel_file():
    short_stories_excel = openpyxl.load_workbook('4000-Stories-with-sentiment-analysis.xlsx', data_only=True)
    print('loaded .xlsx workbook')
    sheet = short_stories_excel['Sheet1']

    urls = []

    for cell in sheet['B']:
        print(cell.value)
        if cell.value != "url":
            urls.append(cell.value)
    
    return urls



def scrape_story(url: str):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(url)
    content = driver.page_source

    soup = BeautifulSoup(content)

    story_text = ''
    story = soup.find("div", {"class":"jumbotron", "itemtype":"https://schema.org/ShortStory"}).findAll('p')
    print(len(story))

    for p in story:
        story_text += ' ' + ''.join(p.findAll(text = True))
    print(story_text)

urls = extract_links_from_excel_file()

for url in urls:
    scrape_story(url)
