# Sci-kit learn
from sklearn import tree

# CSV library
import csv

features = []
labels = []
embarked = {
    "": 0,
    "S": 1,
    "Q": 2,
    "C": 3
}
with open('train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    is_first = 0
    for row in readCSV:
        if is_first == 0:
            is_first = 1
            continue
        labels.append(row[1])
        row.pop(0)
        row.pop(0)
        row.pop(1)
        if row[1] == "male":
            row[1] = 1
        else:
            row[1] = 0
        row.pop(5)
        row.pop(6)
        row[6] = embarked[row[6]]
        rowLen = len(row)
        for i in range(rowLen):
            if row[i] == "":
                row[i] = 0
            row[i] = float(row[i])
        features.append(row)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

examples_to_predict = []
with open('test.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    is_first = 0
    for row in readCSV:
        if is_first == 0:
            is_first = 1
            continue
        current_example = []
        current_example.append(row[0])
        current_example.append([])
        row.pop(0)
        row.pop(1)
        if row[1] == "male":
            row[1] = 1
        else:
            row[1] = 0
        row.pop(5)
        row.pop(6)
        row[6] = embarked[row[6]]
        rowLen = len(row)
        for i in range(rowLen):
            if row[i] == "":
                row[i] = 0
            row[i] = float(row[i])
        current_example[1] = row
        examples_to_predict.append(current_example)

print("PassengerId,Survived")
for example in examples_to_predict:
    print("{0},{1}".format(example[0], clf.predict([example[1]])[0]))

