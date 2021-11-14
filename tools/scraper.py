from selenium import webdriver
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://americanliterature.com/author/eleanor-hallowell-abbott/short-story/peace-on-earth-good-will-to-dogs")
content = driver.page_source

data = BeautifulSoup(content)
#print(data)