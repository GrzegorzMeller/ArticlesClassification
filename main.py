from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import f1_score
import time
from sklearn.svm import SVC

#DATA PREPROCESSING
DATASET_PATH = './news_en.csv'
dataset = pd.read_csv(DATASET_PATH)
training_set = dataset.loc[1:24036]
validating_set = dataset.loc[24037:32049]
testing_set = dataset.loc[32050:39599]

vectorizer = TfidfVectorizer(min_df=2,
                             max_df=0.75,
                             sublinear_tf=True,
                             use_idf=True)

train_vectors = vectorizer.fit_transform(training_set['Body'])
validate_vectors = vectorizer.transform(validating_set['Body'])
test_vectors = vectorizer.transform(testing_set['Body'])
print(train_vectors)

#SVM
start = time.time()
svm = SVC(kernel='linear', verbose=True)
print('svm started')
svm.fit(train_vectors, training_set['Kat'])
y_pred = svm.predict(validate_vectors)
print(f1_score(validating_set['Kat'], y_pred, average='micro'))
y_pred2 = svm.predict(test_vectors)
print(f1_score(testing_set['Kat'], y_pred2, average='micro'))
end = time.time()
print(end - start)

#NAIVE BAYES
start = time.time()
naive_bayes = MultinomialNB()
naive_bayes.fit(train_vectors, training_set['Kat'])
end = time.time()
print(end - start)
y_pred = naive_bayes.predict(validate_vectors)
print(f1_score(validating_set['Kat'], y_pred, average='micro'))

y_pred2 = naive_bayes.predict(test_vectors)
print(f1_score(testing_set['Kat'], y_pred2, average='micro'))
#extra_vector = vectorizer.transform(['As November’s U.S. elections approach, Republicans in the House of Representatives and the Trump administration are planning another deficit-financed tax-cut plan, but one widely seen as a vote-getting exercise with little chance of becoming law. FILE PHOTO: U.S. House Republicans, including Speaker of the House Paul Ryan and House Majority Leader Kevin McCarthy, celebrate at a news conference announcing the passage of the "Tax Cuts and Jobs Act" at the U.S. Capitol in Washington, DC, U.S., November 16, 2017. REUTERS/Aaron P. Bernstein/File PhotoThe plan, coming on the heels of deep tax cuts already approved in December, underscores both the Republican Party’s steadfast confidence in tax cuts as a winning political tool and its recent shift away from fiscal policy conservatism. House tax committee Chairman Kevin Brady says his panel and the White House are considering a measure that would make permanent $1.1 trillion in tax cuts that were approved on a temporary basis in December for individuals, families and private businesses. The cuts are set to expire in 2025. Brady, of Texas, says he aims to unveil a proposal before Congress departs Washington on July 26 for a summer campaigning break. He says he expects the House to vote on the measure before the Nov. 6 congressional elections. “This is largely a 2018 re-elect-driven effort for House Republicans,” said Rohit Kumar, a tax policy expert at accounting and consulting group PwC and former senior aide to Senate Republican leader Mitch McConnell.  The nonpartisan Congressional Budget Office has already warned that making permanent the temporary individual tax cuts would further expand the federal deficit and debt. Both measures of red ink on the U.S. taxpayers’ ledger ballooned with the Republicans’ $1.5 trillion December tax cuts package and a $1.3 trillion spending bill approved in March. Brady says he does not expect the Republicans’ new bill to be “revenue neutral,” meaning it will expand the deficit. Florida Republican Representative Carlos Curbelo, who sits on Brady’s committee, called the bill “a good second chapter for tax reform that’s going to help American families.” Republicans see the second tax bill as helping to focus voters on the growing economy, with Trump’s focus otherwise shifting haphazardly from immigration to tariffs, federal investigations, North Korea, U.S. NATO participation and Russia. The temporary cuts approved in December were part of a sweeping tax package, formally known as the Tax Cuts and Jobs Act, that was passed by Republicans, who have majorities in both chambers of Congress, over Democrats’ unanimous opposition. The package gave individuals temporary tax relief, but cut taxes permanently for corporations. Months later, the package is viewed favorably by only 36.4 percent of Americans, according to a polling average compiled by RealClearPolitics.com, which tracks political trends. Some Republicans say the December package has fared better in parts of the country. “In my district, it polls very high,” said  Republican Representative Don Bacon of Nebraska. Representative Richard Neal of  Massachusetts, the top Democrat on the House tax committee, said, “I hope the new legislation is as popular as their last tax cut.” The new bill is not without political risk, said analysts. House Republicans in competitive districts with lots of Democrats and independents may encounter voter concern about the deficit and giveaways to the rich, analysts said. The hastily drafted December package has resulted in some unanticipated complications for multinational corporations, but a bill to fix those was not expected until after the elections. In any case, prospects for passage of another round of tax cuts are low, with the fall legislative session already crammed full of other issues. Even if House Republicans win passage, a new tax bill could die in the Senate, where Republicans hold only 51 of 100 seats and would need help from Democrats, who do not seem to be inclined to offer any.  The idea of more deficit-financed tax cuts already faces opposition from Republican Senator Bob Corker of Tennessee, one of the few remaining fiscal hawks in the party. Asked if he would support making permanent individual tax cuts, Corker said, “No.” '])
#print(naive_bayes.predict(extra_vector))

#KNN
print('kNN')
start = time.time()
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(train_vectors, training_set['Kat'])

y_pred=knn.predict(validate_vectors)
print(f1_score(validating_set['Kat'], y_pred, average='micro'))

y_pred2 = knn.predict(test_vectors)
print(f1_score(testing_set['Kat'], y_pred2, average='micro'))
end = time.time()
print(end - start)
