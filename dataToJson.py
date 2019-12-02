import json
import os

data = {"text":{}, "label":{}}
count = 0


for filename in os.listdir("./fake"):
    with open("./fake/" + filename, encoding='utf8') as fakeFile:
        print(filename)
        text = fakeFile.read().strip().replace("\n", " ")
        data["text"][str(count)] = text
        data["label"][str(count)] = 0
        count += 1

for filename in os.listdir("./real"):
    with open("./real/" + filename, encoding='utf8') as realFile:
        print(filename)
        text = realFile.read().strip().replace("\n", " ")
        data["text"][str(count)] = text
        data["label"][str(count)] = 1
        count += 1


with open('data.json', 'w', encoding="utf8") as outfile:
    json.dump(data, outfile)
