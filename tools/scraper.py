from selenium import webdriver
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://americanliterature.com/author/eleanor-hallowell-abbott/short-story/peace-on-earth-good-will-to-dogs")
content = driver.page_source

soup = BeautifulSoup(content)

story_text = ''
story = soup.find("div", {"class":"jumbotron", "itemtype":"https://schema.org/ShortStory"}).findAll('p')
print(len(story))

for p in story:
    story_text += ' ' + ''.join(p.findAll(text = True))
print(story_text)