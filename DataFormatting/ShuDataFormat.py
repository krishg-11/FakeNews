import json
import os


def importFolderIntoData(data, folder, count, label): #given current data dict, add data from all files in given folder into data dict
    for subfolder in os.listdir(folder):
        subfolderpath = folder + "/" + subfolder
        for filename in os.listdir(subfolderpath):
            with open(subfolderpath + "/" + filename) as jsonfile:
                tempData = json.load(jsonfile)
                if(len(tempData["text"]) <= 10): #if there is no text (or near no text), don't include datapoint
                    pass
                outText = "{url}\n\n{authors}\n\n{title}\n\n{text}\n\n{summary}".format(url=tempData["url"],
                                                                                        authors=', '.join(tempData["authors"]),
                                                                                        title=tempData["title"],
                                                                                        text=tempData["text"],
                                                                                        summary=tempData["summary"])
                data["text"][str(count)] = outText
                data["label"][str(count)] = label
                count += 1

    return data, count

path = "../fakenewsnet_dataset/"

data = json.load(open("../data.json")) # data = {"text":{"0":"...", "1":... etc.}, "label":{"0":1, "1", 0 etc.}}
count = len(data["text"])



print("starting politifact fake; count =", count)
data, count = importFolderIntoData(data, path+"politifact/fake", count, 0)
print("starting politifact real; count =", count)
data, count = importFolderIntoData(data, path+"politifact/real", count, 1)
print("starting gossipcop fake; count =", count)
data, count = importFolderIntoData(data, path+"gossipcop/fake", count, 0)
print("starting gossipcop real; count =", count)
data, count = importFolderIntoData(data, path+"gossipcop/real", count, 1)

with open('../data.json', 'w', encoding="utf8") as outfile:
    json.dump(data, outfile)
