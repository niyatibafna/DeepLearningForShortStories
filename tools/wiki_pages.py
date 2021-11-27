import wikipediaapi
import requests
import os


S = requests.Session()
wiki = wikipediaapi.Wikipedia('en')
URL = "https://en.wikipedia.org/w/api.php"

CLEAN_DATA_PATH = "../data/clean/"
stories = os.listdir(CLEAN_DATA_PATH)

# story_page = wiki_wiki.page("The Murders in the Rue Morgue")
# if story_page.exists():
#     print(story_page.summary[0:100])
# else:
#     print("ghanta")
wiki_found = 0
story_count = 0
indicators = ["written", "story"]
for fname in stories[1000:2000]:
    title = " ".join(fname.split("_")[1:])[:-4]

    PARAMS = {
        "action": "opensearch",
        "namespace": "0",
        "search": title + " short story",
        "limit": "5",
        "format": "json"
    }

    open_results = S.get(url=URL, params=PARAMS)
    wiki_titles = open_results.json()[1]
    print(title)
    print("WIKI ", wiki_titles)
    if len(wiki_titles)==0:
        continue
    wiki_found += 1
    for w_title in wiki_titles:
        if w_title.lower() != title and "short story" not in w_title:
            continue
        story_page = wiki.page(w_title)
        if story_page.exists():
            ind_words = [True]
            if True in ind_words:
                story_count += 1
                print(story_page.summary[0:100])
                break

print(wiki_found)
print(story_count)
