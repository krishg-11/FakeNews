import json
import os

data = json.load(open("data.json")) # data = {"text":{"0":"...", "1":... etc.}, "label":{"0":1, "1", 0 etc.}}

for count in range(len(data["text"])):
    directory = "./allRealData/" if (data["label"][str(count)] == 1) else "./allFakeData/"
    filename = os.path.join(directory, str(count)+".txt")
    with open(filename, "w", encoding="utf8") as outfile:
        outfile.write(data["text"][str(count)])
