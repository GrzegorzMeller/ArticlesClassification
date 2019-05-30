import numpy as np

#Creating input matrix scheme
COLUMNS = 101938
ROWS = 14146
features = np.zeros(shape=(ROWS, COLUMNS))
print(features.shape)

#filling input matrix
dataset = open('./wiki10_train.txt', "r") #from extreme classification repository
i = -1
for x in dataset:
    i = i+1
    x = x.split(' ', 1)
    #print(x[1])
    for word in x[1].split():
        #print(word)
        word = word.split(':', 1)
        column_id = word[0]
        value = word[1]
        features[i][int(column_id)] = value

dataset.close()

#Creatig output matrix scheme
COLUMNS_LABELS = 30938
labels = np.zeros(shape=(ROWS, COLUMNS_LABELS))

dataset = open('./wiki10_train.txt', "r")
i = -1
for x in dataset:
    i = i+1
    x = x.split(' ', 1)
    row_labels = x[0]
    row_labels = row_labels.replace(',', ' ')
    for word in row_labels.split():
        column_id = word
        labels[i][int(column_id)] = 1


dataset.close()

#divide matricies into training, validating and testing set
X_train = features[1:8488]
Y_train = labels[1:8488]

X_validate = features[8489:11318]
Y_validate = labels[8489:11318]

X_testing = features[11319:14146]
Y_testing = labels[11319:14146]

print(X_train.shape)
print(Y_train.shape)
print(X_validate.shape)
print(Y_validate.shape)
