import json

data = json.load(open("../data.json"))
refinedData = {"text":{}, "label":{}}
setSeen = set()

count=0
for i in range(len(data["text"])):
    text, label = data["text"][str(i)], data["label"][str(i)]
    if(text in setSeen): continue
    setSeen.add(text)
    refinedData["text"][str(count)] = text
    refinedData["label"][str(count)] = label
    count += 1

with open('../data.json', 'w', encoding="utf8") as outfile:
    json.dump(refinedData, outfile)
