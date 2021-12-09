import json
with open("genre_buckets.json", "r") as g_file:
    f = json.load(g_file)
print(len(f))
stats_f = [(k,len(v)) for k,v in f.items()]
stats_f = sorted(stats_f, key= lambda x: x[1], reverse=True)
print(stats_f)
all_stories = set()
for k,v in f.items():
    all_stories = all_stories.union(v)
print(len(all_stories))
