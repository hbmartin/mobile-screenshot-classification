import os

id_map = {}



with open("deduped.csv") as f:
    for line in f:
        id_map[line.split(",")[0]] = ", ".join([s.strip() for s in line.split(",")[2:]])

for dirname in os.listdir("screenshots"):
    if dirname in id_map:
        print(dirname + " to " + id_map[dirname])
        os.rename("screenshots/" + dirname, "screenshots/" + id_map[dirname])
    else:
        print("DID NOT FIND " + dirname)