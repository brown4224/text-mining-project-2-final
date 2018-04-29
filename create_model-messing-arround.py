import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfTransformer ,CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from nltk import PorterStemmer
from nltk.corpus import stopwords
# analyzer = CountVectorizer().build_analyzer()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, MinMaxScaler

analyzer = TfidfVectorizer().build_analyzer()
import scipy
from sklearn.svm import SVC
import gensim, logging
from gensim.models.doc2vec import LabeledSentence
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
is_debug = False
import spacy # Requirements: yum install python-devel python3-dev  libevent-dev; pip install spacy && python -m spacy download en
nlp = spacy.load('en')

from sklearn.decomposition import TruncatedSVD

'''Please feel free to modify this function (load_models)
   Make sure the function return the models required for the function below evaluate model
'''

def load_models():
    vectorizer = joblib.load('resources/vectorizer.pkl')
    clf = joblib.load('resources/classifier.pkl')
    return [vectorizer, clf]

'''Please feel free to modify this function (evaluate_model)
  Make sure the function only take three parameters: 1) Model (one or more model files), 2) Query, and 3) Document.
  The function always should return one of the positions (classes) and the confidence. In my case confidence is always 0.5.
  Preferably implement the function in a different file and call from here. Make sure keep everything else the same.
'''

#
# def evaluate_model(model, query, document):
#     query = query['query']
#     title = document['title']
#     body = document['body']
#     query_vec = model[0].transform([query])
#     title_vec = model[0].transform([title])
#     body_vec = model[0].transform([body])
#
#     cos_title = cosine_similarity(query_vec, title_vec)
#     cos_body = cosine_similarity(query_vec, body_vec)
#
#     common_title = common_terms(title, query)
#     common_body = common_terms(body, query)
#
#     # Query idf
#     mq = np.max(query_vec)
#     mqp = np.argmax(query_vec)
#     lq = text_length(query)
#     qs = np.sum(query_vec)
#     qsn = qs / lq
#
#     # Title idf
#     mt = np.max(title_vec)
#     mtp = np.argmax(title_vec)
#     lt = text_length(title)
#     ts = np.sum(title_vec)
#     tsn = ts / lt
#
#     # Body idf
#     mb = np.max(body_vec)
#     mbp = np.argmax(body_vec)
#     lb = text_length(body)
#     bs = np.sum(body_vec)
#     bsn = bs / lb
#
#
#     result = model[1].predict([[ cos_title, cos_body, common_title, common_body,
#                                  mq, mqp, qs, qsn, lq,
#                                  mt, mbp, bs, bsn, lb,
#                                  mb, mbp, bs, bsn, lb]])
#
#
#
#
#     # result = model[1].predict([[cos_title, common_title, cos_body, common_body]])
#     return result[0],0.5



def evaluate_model(model, query, document):
    query = query['query']
    title = document['title']
    body = document['body']
    query_vec = model[0].transform([query])
    title_vec = model[0].transform([title])
    body_vec = model[0].transform([body])

    cos_title = cosine_similarity(query_vec, title_vec)
    cos_body = cosine_similarity(query_vec, body_vec)

    common_title = common_terms(title, query)
    common_body = common_terms(body, query)

    # Query idf
    mq = np.max(query_vec)
    mqp = np.argmax(query_vec)
    lq = text_length(query)
    qs = np.sum(query_vec)
    qsn = qs / lq
    qp = idf_prob(query_vec)

    # Title idf
    mt = np.max(title_vec)
    mtp = np.argmax(title_vec)
    lt = text_length(title)
    ts = np.sum(title_vec)
    tsn = ts / lt
    tp = idf_prob(title_vec)

    # Body idf
    mb = np.max(body_vec)
    mbp = np.argmax(body_vec)
    lb = text_length(body)
    bs = np.sum(body_vec)
    bsn = bs / lb
    bp = idf_prob(body_vec)


    # result = model[1].predict([[ common_title, common_body,
    #                              mq, mqp, qsn, lq, qp,
    #                              mt, mbp, bsn, lb, tp,
    #                              mb, mbp, bsn, lb, bp]])

    result = model[1].predict([[
                                 mq, mqp, qsn, lq, qp,
                                 mt, mbp, bsn, lb, tp,
                                 mb, mbp, bsn, lb, bp]])


    # result = model[1].predict([[cos_title, common_title, cos_body, common_body]])
    return result[0],0.5

def common_terms(x,y, stopwords=False):
    x = x.split(" ")
    y = y.split(" ")
    if stopwords:
        x = [PorterStemmer().stem(t) for t in x]
        y = [PorterStemmer().stem(t) for t in y]

    count = 0
    for token in set(x):
        if token in set(y) and token != "" and token != " ":
            count += 1
    return count
def common_count(x,y, stopwords=False):
    x = x.split(" ")
    y = y.split(" ")
    if stopwords:
        x = [PorterStemmer().stem(t) for t in x]
        y = [PorterStemmer().stem(t) for t in y]

    count = 0
    for token in x:
        if token in y and token != "" and token != " ":
            count += 1
    return count


def cosine_similarity(x,y, w2v=False):
    if x is None or y is None:
        return 0.0
    if w2v:
        cos = cosine(np.array(x), np.array(y))
    else:
        cos = cosine(x.toarray()[0], y.toarray()[0])


    if np.isfinite(cos):
        return cos
    return 0.0


def stemming(tokens):
    return (PorterStemmer().stem(token) for token in analyzer(tokens))


def text_length(X):
    return len(X) + 1


def idf_prob(doc_sparce):
    doc = scipy.sparse.coo_matrix(doc_sparce)
    dp = 1.0
    for doc_word, doc_idf in zip(doc.col, doc.data):
        dp *= doc_idf
    return dp

def one_hot(labels):
    l = len(labels)
    Y= np.zeros((l, 4))
    for i in range(l):
        Y[i][labels[i] - 1] = 1
    return Y
def convert_to_w2v(tokens, w2v):
    tokens = tokens.split(" ")
    pattern_vec = np.zeros(w2v.layer1_size)
    n_word = 1
    if len(tokens) > 1:
        for token in tokens:
            if token in w2v:
                pattern_vec = np.add(pattern_vec, w2v[token.strip()])
                n_word += 1
    pattern_vec = np.divide(pattern_vec, n_word)
    return pattern_vec.tolist()


def convert_to_blob(text):
    opinion = TextBlob(text)
    return opinion


def sentiment(textblob):
    return textblob.sentiment


def noun_counter(textblob):
    return len(textblob.noun_phrases)
def noun_counter_union(textblob_doc, textblob_query):
    counter =0
    noun1 = textblob_doc.noun_phrases
    noun2 = textblob_query.noun_phrases
    for n1 in noun1:
        for n2 in noun2:
            if n1 == n2:
                counter +=1
    return counter

def noun_frequency(textblob):
    return textblob.np_counts

def polarity(textblob):
    return textblob.polarity

def subjectivity(textblob):
    return textblob.subjectivity


def cos_avg(x, y):
    return (float(x) + float(y)) / 2.0

def split_tuples(tup):
    p = []
    s = []
    for t in tup:
        p.append(t[0])
        s.append(t[1])
    return p, s
def spacy_similarity(doc, query):
    doc1 = nlp(doc)
    doc2 = nlp(query)
    return doc1.similarity(doc2)





def create_model(all_documents_file, relevance_file,query_file):

    '''Step 1. Creating  a dataframe with three fields query, title, and relevance(position)'''
    documents = pd.read_json(all_documents_file)[["id", "title", "body"]]
    query_file = pd.read_json(query_file)[["query number","query" ]]

    relevance = pd.read_json(relevance_file)[["query_num", "position", "id"]]
    rv = relevance.merge(query_file,left_on ="query_num", right_on="query number")[ ["id","query", "position"]]\
        .merge(documents,left_on ="id", right_on="id") [["query", "position", "title", "body"]]


    '''Step 2. Creating  a column for creating index'''

    rv ["all_text"] = rv.apply( lambda x : x["query"] + x["title"] + x["body"] , axis =1)

    ''' Step 3. Creating a model for generating TF feature'''

    # vectorizer = TfidfVectorizer()
    vectorizer = TfidfVectorizer( min_df=0.0, max_df=1.0, stop_words="english", lowercase=True, norm="l2", strip_accents='ascii')
    vectorizer = vectorizer.fit(rv["all_text"])


    ''' Word to Vec Model'''
    w2v_model = gensim.models.Word2Vec( min_count=1, workers=4)
    w2v_model.build_vocab(rv["all_text"])



    ''' Step 4. Saving the model for TF features'''
    joblib.dump(vectorizer, 'resources/vectorizer.pkl')
    w2v_model.save('resources/w2v.model')

    ''' Step 5. Converting query and title to vectors and finding cosine similarity of the vectors'''
    rv["query_vec"] = rv.apply(lambda x: vectorizer.transform([x["query"]]), axis =1)
    rv["doc_vec_title"] = rv.apply(lambda x: vectorizer.transform([x["title"] ]), axis =1)
    rv["doc_vec_body"] = rv.apply(lambda x: vectorizer.transform([ x["body"]]), axis =1)


    rv["cosine_title"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_title'], x['query_vec']), axis=1)
    rv["cosine_body"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_body'], x['query_vec']), axis=1)
    rv["common_title"] = rv.apply(lambda x: common_terms(x["title"], x["query"]), axis =1)


    '''  COS and COMMON '''
    rv["cosine_title"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_title'], x['query_vec']), axis=1)
    rv["cosine_body"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_body'], x['query_vec']), axis=1)
    rv["common_title"] = rv.apply(lambda x: common_terms(x["title"], x["query"] ), axis =1)
    rv["common_body"] = rv.apply(lambda x: common_terms(x["body"], x["query"]), axis =1)
    rv["common_title_s"] = rv.apply(lambda x: common_count(x["title"], x["query"],stopwords=True ), axis =1)
    rv["common_body_s"] = rv.apply(lambda x: common_count(x["body"], x["query"], stopwords=True) , axis =1)

    '''  Word 2V'''
    rv["query_vec_w2v"] = rv.apply(lambda x: convert_to_w2v(x["query"], w2v_model), axis=1)
    rv["doc_vec_w2v"] = rv.apply(lambda x: convert_to_w2v(x["title"], w2v_model), axis=1)
    rv["body_vec_w2v"] = rv.apply(lambda x: convert_to_w2v(x["body"], w2v_model), axis=1)

    ''' Cos W2V'''
    rv["cosine_title_w2v"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_w2v'], x['query_vec_w2v'], w2v=True), axis=1)
    rv["cosine_body_w2v"]  = rv.apply(lambda x: cosine_similarity(x['body_vec_w2v'], x['query_vec_w2v'], w2v=True), axis=1)

    ''' Cos AVG'''
    rv["avg_cos_title"] = rv.apply(lambda x: cos_avg(x['cosine_title_w2v'], x['cosine_title']), axis=1)
    rv["avg_cos_body"] = rv.apply(lambda x: cos_avg(x['cosine_body_w2v'], x['cosine_body']), axis=1)


    '''  Textblob'''
    rv["query_vec_blob"] = rv.apply(lambda x: convert_to_blob(x["query"]), axis=1)
    rv["doc_vec_blob"] = rv.apply(lambda x: convert_to_blob(x["title"]), axis=1)
    rv["body_vec_blob"] = rv.apply(lambda x: convert_to_blob(x["body"]), axis=1)

    '''' Textblob noun count Union '''
    rv["query_noun"] = rv.apply(lambda x: noun_counter( x["query_vec_blob"]), axis=1)
    rv["title_noun"] = rv.apply(lambda x: noun_counter(x["doc_vec_blob"]), axis=1)
    rv["body_noun"] = rv.apply(lambda x: noun_counter(x["body_vec_blob"]), axis=1)


    '''' Textblob noun count Union '''
    rv["title_noun_union"] = rv.apply(lambda x: noun_counter_union(x["doc_vec_blob"], x["query_vec_blob"]), axis=1)
    rv["body_noun_union"] = rv.apply(lambda x: noun_counter_union(x["body_vec_blob"], x["query_vec_blob"]), axis=1)


    '''' polarity '''
    rv["query_polarity"] = rv.apply(lambda x: polarity(x["query_vec_blob"]), axis=1)
    rv["title_polarity"] = rv.apply(lambda x: polarity(x["doc_vec_blob"]), axis=1)
    rv["body_polarity"] = rv.apply(lambda x: polarity(x["body_vec_blob"]), axis=1)

    '''' subjectivity '''
    rv["query_subjectivity"] = rv.apply(lambda x: subjectivity(x["query_vec_blob"]), axis=1)
    rv["title_subjectivity"] = rv.apply(lambda x: subjectivity(x["doc_vec_blob"]), axis=1)
    rv["body_subjectivity"] = rv.apply(lambda x: subjectivity(x["body_vec_blob"]), axis=1)



    '''cosine portability'''
    rv["cosine_title_polarity"]  = rv.apply(lambda x: cosine_similarity(x['title_polarity'], rv["query_polarity"], w2v=True), axis=1)
    rv["cosine_body_polarity"]  = rv.apply(lambda x: cosine_similarity(x['body_polarity'], x['query_polarity'], w2v=True), axis=1)

    ''' cosine subjectivity'''
    rv["cosine_title_subjectivity"]  = rv.apply(lambda x: cosine_similarity(x['title_subjectivity'], rv["query_subjectivity"], w2v=True), axis=1)
    rv["cosine_body_subjectivity"]  = rv.apply(lambda x: cosine_similarity(x['body_subjectivity'], x['query_subjectivity'], w2v=True), axis=1)


    ''' IDF Extraction'''
    rv["max_query_idf"] = rv.apply(lambda x: np.max(x["query_vec"]), axis=1)
    rv["max_pos_query_idf"] = rv.apply(lambda x: np.argmax(x["query_vec"]), axis=1)
    rv["sum_query_idf"] = rv.apply(lambda x: np.sum(x["query_vec"]), axis=1)
    rv["len_query_idf"] = rv.apply(lambda x: text_length(x["query"]), axis=1)
    rv["norm_query_idf"] = np.divide(rv["sum_query_idf"], rv["len_query_idf"])
    rv["prob_query_idf"] = rv.apply(lambda x: idf_prob(x["query_vec"]), axis =1)
    rv["mean_query_idf"] = rv.apply(lambda x: np.mean(x["query_vec"]), axis=1)


    rv["max_title_idf"] = rv.apply(lambda x: np.max(x["doc_vec_title"]), axis =1)
    rv["max_pos_title_idf"] = rv.apply(lambda x: np.argmax(x["doc_vec_title"]), axis =1)
    rv["sum_title_idf"] = rv.apply(lambda x: np.sum(x["doc_vec_title"]), axis =1)
    rv["len_title_idf"] = rv.apply(lambda x: text_length(x["title"]), axis =1)
    rv["norm_title_idf"] = np.divide(rv["sum_title_idf"] ,rv["len_title_idf"] )
    rv["prob_title_idf"] = rv.apply(lambda x: idf_prob(x["doc_vec_title"]), axis =1)
    rv["mean_title_idf"] = rv.apply(lambda x: np.mean(x["doc_vec_title"]), axis =1)


    rv["max_body_idf"] = rv.apply(lambda x: np.max(x["doc_vec_body"]), axis =1)
    rv["max_pos_body_idf"] = rv.apply(lambda x: np.argmax(x["doc_vec_body"]), axis =1)
    rv["sum_body_idf"] = rv.apply(lambda x: np.sum(x["doc_vec_body"]), axis =1)
    rv["len_body_idf"] = rv.apply(lambda x: text_length(x["body"]), axis =1)
    rv["norm_body_idf"] = np.divide(rv["sum_body_idf"] ,rv["len_body_idf"] )
    rv["prob_body_idf"] = rv.apply(lambda x: idf_prob(x["doc_vec_body"]), axis =1)
    rv["mean_body_idf"] = rv.apply(lambda x: np.mean(x["doc_vec_body"]), axis =1)

    ''' Cos Meta data'''
    rv["cosine_title_max_idf"]  = rv.apply(lambda x: cosine_similarity(x['max_title_idf'], rv["max_query_idf"], w2v=True), axis=1)
    rv["cosine_body_max_idf"]  = rv.apply(lambda x: cosine_similarity(x['max_body_idf'], rv["max_query_idf"], w2v=True), axis=1)


    rv["cosine_norm_title_idf"]  = rv.apply(lambda x: cosine_similarity(x['norm_title_idf'], rv["norm_query_idf"], w2v=True), axis=1)
    rv["cosine_norm_body_idf"]  = rv.apply(lambda x: cosine_similarity(x['norm_body_idf'], rv["norm_body_idf"], w2v=True), axis=1)

    rv["cosine_mean_title_idf"]  = rv.apply(lambda x: cosine_similarity(x['mean_title_idf'], rv["mean_query_idf"], w2v=True), axis=1)
    rv["cosine_mean_body_idf"]  = rv.apply(lambda x: cosine_similarity(x['mean_body_idf'], rv["mean_query_idf"], w2v=True), axis=1)

    rv["cosine_prob_title_idf"]  = rv.apply(lambda x: cosine_similarity(x['mean_title_idf'], rv["mean_query_idf"], w2v=True), axis=1)
    rv["cosine_prob_body_idf"]  = rv.apply(lambda x: cosine_similarity(x['mean_body_idf'], rv["mean_query_idf"], w2v=True), axis=1)






    ''' Step 6. Defining the feature and label  for classification'''
    X = rv[["cosine_title"] + ["cosine_body"]
           + ["cosine_title_w2v"]+ ["cosine_body_w2v"]
           + ["cosine_title_polarity"] + ["cosine_title_subjectivity"] + ["cosine_body_polarity"] + ["cosine_body_subjectivity"]
           + ["max_query_idf"] + ["max_pos_query_idf"] + ["norm_query_idf"] + ["len_query_idf"] + ["prob_query_idf"]+["mean_query_idf"]
           + ["max_title_idf"] + ["max_pos_title_idf"] + ["norm_title_idf"] + ["len_title_idf"] + ["prob_title_idf"]+["mean_title_idf"]
           + ["max_body_idf"] + ["max_pos_body_idf"] + ["norm_body_idf"] + ["len_body_idf"] + ["prob_body_idf"]+["mean_body_idf"]
           + ["common_title"]+ ["common_body"]
           + ["cosine_title_polarity"]+ ["cosine_body_polarity"]
           + ["cosine_title_subjectivity"]+ ["cosine_body_subjectivity"]
           ]





    Y = [v for k, v in rv["position"].items()]



    ''' Step 7. Splitting the data for validation'''
    X_train, X_test, y_train, y_test = train_test_split(    X, Y, test_size = 0.33, random_state = 42)


    from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler = MinMaxScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)


    #
    # clf = VotingClassifier(
    #     estimators=[('ab', AdaBoostClassifier()), ('gb',  GradientBoostingClassifier(n_estimators=100)) ],
    #     # estimators=[('ab', AdaBoostClassifier()), ('gb',  GradientBoostingClassifier(n_estimators=100)), ('rf', RandomForestClassifier(n_estimators=100)) ],
    #     voting='soft'
    #
    # ).fit(X_train, y_train)


    ''' Step 8. Classification and validation'''
    target_names = ['1', '2', '3','4']



    clf = GradientBoostingClassifier(n_estimators=1000).fit(X_train, y_train)



    print(classification_report(y_test,  clf.predict(X_test), target_names=target_names))

    ''' Step 9. Saving the data '''
    joblib.dump(clf, 'resources/classifier.pkl')



if __name__ == '__main__':
    create_model("resources/cranfield_data.json", "resources/cranqrel.json", "resources/cran.qry.json")