import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer ,CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from nltk import PorterStemmer
from nltk.corpus import stopwords
# analyzer = CountVectorizer().build_analyzer()
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


def noun_counter(x,y, stopwords=False):
    # x = x.split(" ")
    # y = y.split(" ")
    # if stopwords:
    #     x = [PorterStemmer().stem(t) for t in x]
    #     y = [PorterStemmer().stem(t) for t in y]

    count = 0
    doc1 = nlp(x)
    doc2 = nlp(y)
    for t1 in doc1.noun_chunks:
        for t2 in doc2.noun_chunks:
            if t1 == t2:
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


    return opinion.sentiment


# def naiveBayes(text):
#     opinion = TextBlob(text, analyzer=NaiveBayesAnalyzer())
#     print(opinion)
#     print(opinion.sentiment)
#     exit()


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



def idf_sparce(doc_vec, query_vec):
    count = []
    doc_max_pos = []
    doc_max = []
    doc_prob = []
    doc_sum = []

    query_max_pos = []
    query_max = []
    query_prob = []
    query_sum = []



    for i in range(len(doc_vec)):

        doc = scipy.sparse.coo_matrix(doc_vec[i])
        query = scipy.sparse.coo_matrix(query_vec[i])

        for doc_word, doc_idf in zip(doc.col, doc.data):
            dmp = 0
            dm = 0.0
            dp = 1.0
            ds = 0.0
            qmp = 0
            qm = 0.0
            qp = 1.0
            qs = 0.0
            c = 0
            for query_word, query_idf in zip(query.col, query.data):
                if(doc_word == query_word):
                    if( doc_idf > dm):
                        dmp = doc_word
                        dm = doc_idf
                        dp *= doc_idf
                    if(query_idf > qm):
                        qmp = query_word
                        qm = query_idf
                        qp *= query_idf
                    ds += doc_idf
                    qs += query_idf
                    c += 1
        count.append(c)
        doc_max_pos.append(dmp)
        doc_max.append(dm)
        doc_prob.append(dp)
        doc_sum.append(ds / (c + 1))
        # doc_sum.append(ds / (c + 1))
        query_max_pos.append(qmp)
        query_max.append(qm)
        query_prob.append(qp)
        query_sum.append(qs / (c + 1))
        # query_sum.append(qs / (c + 1))
    return count, doc_max_pos, doc_max, doc_prob,  doc_sum, query_max_pos, query_max, query_prob,  query_sum




def create_model(all_documents_file, relevance_file,query_file):

    '''Step 1. Creating  a dataframe with three fields query, title, and relevance(position)'''
    documents = pd.read_json(all_documents_file)[["id", "title", "body"]]
    query_file = pd.read_json(query_file)[["query number","query" ]]

    relevance = pd.read_json(relevance_file)[["query_num", "position", "id"]]
    rv = relevance.merge(query_file,left_on ="query_num", right_on="query number")[ ["id","query", "position"]]\
        .merge(documents,left_on ="id", right_on="id") [["query", "position", "title", "body"]]

    Y = [v for k, v in rv["position"].items()]
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    for y in Y:
        if y ==1:
            Y1.append(1)
            Y2.append(0)
            Y3.append(0)
            Y4.append(0)
        if y == 2:
            Y1.append(0)
            Y2.append(1)
            Y3.append(0)
            Y4.append(0)
        if y == 3:
            Y1.append(0)
            Y2.append(0)
            Y3.append(1)
            Y4.append(0)
        if y == 4:
            Y1.append(0)
            Y2.append(0)
            Y3.append(0)
            Y4.append(1)
# port preprocessing
    # Y = one_hot([v for k, v in rv["position"].items()])





    '''Step 2. Creating  a column for creating index'''

    rv ["all_text"] = rv.apply( lambda x : x["query"] + x["title"] + x["body"] , axis =1)

    ''' Step 3. Creating a model for generating TF feature'''

    vectorizer = TfidfVectorizer( min_df=0.0, max_df=1.0, stop_words="english", lowercase=True, norm="l2", strip_accents='ascii')
    vectorizer = vectorizer.fit(rv["all_text"])


    ''' Word to Vec Model'''
    w2v_model = gensim.models.Word2Vec( min_count=1, workers=4)
    w2v_model.build_vocab(rv["all_text"])


    '''  Text Blob Classifer'''


    ''' Step 4. Saving the model for TF features'''
    joblib.dump(vectorizer, 'resources/vectorizer.pkl')
    w2v_model.save('resources/w2v.model')
    # d2v_model.save('resources/d2v.model')

    # ''' Step 5. Converting query and title to vectors and finding cosine similarity of the vectors'''
    # rv["query_vec"] = rv.apply(lambda x: vectorizer.transform([x["query"]]), axis =1)
    # rv["doc_vec_title"] = rv.apply(lambda x: vectorizer.transform([x["title"] ]), axis =1)
    # rv["doc_vec_body"] = rv.apply(lambda x: vectorizer.transform([ x["body"]]), axis =1)
    #
    # '''  COS and COMMON '''
    # rv["cosine_title"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_title'], x['query_vec']), axis=1)
    # rv["cosine_body"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_body'], x['query_vec']), axis=1)
    # rv["common_title"] = rv.apply(lambda x: common_terms(x["title"], x["query"] ), axis =1)
    # rv["common_body"] = rv.apply(lambda x: common_terms(x["body"], x["query"]), axis =1)
    # rv["common_title_s"] = rv.apply(lambda x: common_count(x["title"], x["query"],stopwords=True ), axis =1)
    # rv["common_body_s"] = rv.apply(lambda x: common_count(x["body"], x["query"], stopwords=True) , axis =1)
    #
    # '''  Word 2V'''
    # rv["query_vec_w2v"] = rv.apply(lambda x: convert_to_w2v(x["query"], w2v_model), axis=1)
    # rv["doc_vec_w2v"] = rv.apply(lambda x: convert_to_w2v(x["title"], w2v_model), axis=1)
    # rv["body_vec_w2v"] = rv.apply(lambda x: convert_to_w2v(x["body"], w2v_model), axis=1)
    #
    # ''' Cos W2V'''
    # rv["cosine_title_w2v"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_w2v'], x['query_vec_w2v'], w2v=True), axis=1)
    # rv["cosine_body_w2v"]  = rv.apply(lambda x: cosine_similarity(x['body_vec_w2v'], x['query_vec_w2v'], w2v=True), axis=1)
    #
    # ''' Cos AVG'''
    # rv["avg_cos_title"] = rv.apply(lambda x: cos_avg(x['cosine_title_w2v'], x['cosine_title']), axis=1)
    # rv["avg_cos_body"] = rv.apply(lambda x: cos_avg(x['cosine_body_w2v'], x['cosine_body']), axis=1)
    #
    #
    # '''Spacy'''
    # # rv["title_spacy"] = rv.apply(lambda x: spacy_similarity(x["title"], x["query"]), axis =1)
    # # rv["body_spacy"] = rv.apply(lambda x: spacy_similarity(x["body"], x["query"]), axis =1)
    # # rv["common_title_noun"] = rv.apply(lambda x: noun_counter(x["title"], x["query"], False ), axis =1)
    # # rv["common_body_noun"] = rv.apply(lambda x: noun_counter(x["body"], x["query"], False), axis =1)
    #
    #
    #
    # '''  Textblob'''
    # rv["query_vec_blob"] = rv.apply(lambda x: convert_to_blob(x["query"]), axis=1)
    # rv["doc_vec_blob"] = rv.apply(lambda x: convert_to_blob(x["title"]), axis=1)
    # rv["body_vec_blob"] = rv.apply(lambda x: convert_to_blob(x["body"]), axis=1)
    #
    # '''Split tuples'''
    # rv["query_vec_blob_p"],rv["query_vec_blob_s"]=  split_tuples(rv["query_vec_blob"])
    # rv["doc_vec_blob_p"],rv["doc_vec_blob_s"]=  split_tuples(rv["doc_vec_blob"])
    # rv["body_vec_blob_p"],rv["body_vec_blob_s"]=  split_tuples(rv["body_vec_blob"])
    #
    # '''Sentament cosine'''
    # rv["cosine_title_pblob"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_blob_p'], rv["query_vec_blob_p"], w2v=True), axis=1)
    # rv["cosine_title_sblob"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_blob_s'], rv["query_vec_blob_s"], w2v=True), axis=1)
    # rv["cosine_body_pblob"]  = rv.apply(lambda x: cosine_similarity(x['body_vec_blob_p'], x['query_vec_blob_p'], w2v=True), axis=1)
    # rv["cosine_body_sblob"]  = rv.apply(lambda x: cosine_similarity(x['body_vec_blob_s'], x['query_vec_blob_p'], w2v=True), axis=1)
    #
    #
    #
    # ''' IDF Extraction'''
    # rv["max_query_idf"] = rv.apply(lambda x: np.max(x["query_vec"]), axis=1)
    # rv["max_pos_query_idf"] = rv.apply(lambda x: np.argmax(x["query_vec"]), axis=1)
    # rv["sum_query_idf"] = rv.apply(lambda x: np.sum(x["query_vec"]), axis=1)
    # rv["len_query_idf"] = rv.apply(lambda x: text_length(x["query"]), axis=1)
    # rv["norm_query_idf"] = np.divide(rv["sum_query_idf"], rv["len_query_idf"])
    # rv["prob_query_idf"] = rv.apply(lambda x: idf_prob(x["query_vec"]), axis =1)
    # rv["mean_query_idf"] = rv.apply(lambda x: np.mean(x["query_vec"]), axis=1)
    #
    #
    # rv["max_title_idf"] = rv.apply(lambda x: np.max(x["doc_vec_title"]), axis =1)
    # rv["max_pos_title_idf"] = rv.apply(lambda x: np.argmax(x["doc_vec_title"]), axis =1)
    # rv["sum_title_idf"] = rv.apply(lambda x: np.sum(x["doc_vec_title"]), axis =1)
    # rv["len_title_idf"] = rv.apply(lambda x: text_length(x["title"]), axis =1)
    # rv["norm_title_idf"] = np.divide(rv["sum_title_idf"] ,rv["len_title_idf"] )
    # rv["prob_title_idf"] = rv.apply(lambda x: idf_prob(x["doc_vec_title"]), axis =1)
    # rv["mean_title_idf"] = rv.apply(lambda x: np.mean(x["doc_vec_title"]), axis =1)
    #
    #
    # rv["max_body_idf"] = rv.apply(lambda x: np.max(x["doc_vec_body"]), axis =1)
    # rv["max_pos_body_idf"] = rv.apply(lambda x: np.argmax(x["doc_vec_body"]), axis =1)
    # rv["sum_body_idf"] = rv.apply(lambda x: np.sum(x["doc_vec_body"]), axis =1)
    # rv["len_body_idf"] = rv.apply(lambda x: text_length(x["body"]), axis =1)
    # rv["norm_body_idf"] = np.divide(rv["sum_body_idf"] ,rv["len_body_idf"] )
    # rv["prob_body_idf"] = rv.apply(lambda x: idf_prob(x["doc_vec_body"]), axis =1)
    # rv["mean_body_idf"] = rv.apply(lambda x: np.mean(x["doc_vec_body"]), axis =1)
    #
    #
    #
    # ''' W2V Data '''
    # rv["query_vec_w2v_pos"] = rv.apply(lambda x: np.argmax(x["query_vec_w2v"]), axis=1)
    # rv["query_vec_w2v_max"] = rv.apply(lambda x: np.max(x["query_vec_w2v"]), axis=1)
    # rv["query_vec_w2v_min"] = rv.apply(lambda x: np.min(x["query_vec_w2v"]), axis=1)
    # rv["query_vec_w2v_sum"] = rv.apply(lambda x: np.sum(x["query_vec_w2v"]), axis=1)
    # rv["query_vec_w2v_norm"] = np.divide(rv["query_vec_w2v_sum"] ,rv["len_title_idf"] )
    # rv["query_vec_w2v_prob"] = rv.apply(lambda x: idf_prob(x["query_vec_w2v"]), axis =1)
    #
    #
    #
    # rv["doc_vec_w2v_pos"] = rv.apply(lambda x: np.argmax(x["doc_vec_w2v"]), axis=1)
    # rv["doc_vec_w2v_max"] = rv.apply(lambda x: np.max(x["doc_vec_w2v"]), axis=1)
    # rv["doc_vec_w2v_min"] = rv.apply(lambda x: np.min(x["doc_vec_w2v"]), axis=1)
    # rv["doc_vec_w2v_sum"] = rv.apply(lambda x: np.sum(x["doc_vec_w2v"]), axis=1)
    # rv["doc_vec_w2v_norm"] = np.divide(rv["doc_vec_w2v_sum"] ,rv["len_title_idf"] )
    # rv["doc_vec_w2v_prob"] = rv.apply(lambda x: idf_prob(x["doc_vec_w2v"]), axis =1)
    #
    #
    # rv["body_vec_w2v_pos"] = rv.apply(lambda x: np.argmax(x["body_vec_w2v"]), axis=1)
    # rv["body_vec_w2v_max"] = rv.apply(lambda x: np.max(x["body_vec_w2v"]), axis=1)
    # rv["body_vec_w2v_min"] = rv.apply(lambda x: np.min(x["body_vec_w2v"]), axis=1)
    # rv["body_vec_w2v_sum"] = rv.apply(lambda x: np.sum(x["body_vec_w2v"]), axis=1)
    # rv["body_vec_w2v_norm"] = np.divide(rv["doc_vec_w2v_sum"] ,rv["len_body_idf"] )
    # rv["body_vec_w2v_prob"] = rv.apply(lambda x: idf_prob(x["body_vec_w2v"]), axis =1)
    #
    #
    #
    #
    # rv["title_count"], rv["title_max_pos"], rv["title_max"], rv["title_prob"],  rv["title_sum"], \
    # rv["query_title_max_pos"], rv["query_title_max"], rv["query_title_prob"], rv["query_title_sum"] =  idf_sparce(rv["doc_vec_title"], rv["query_vec"])
    # rv["body_count"], rv["body_max_pos"], rv["body_max"],  rv["body_prob"], rv["body_sum"], \
    # rv["query_body_max_pos"], rv["query_body_max"], rv["query_body_prob"], rv["query_body_sum"] =  idf_sparce(rv["doc_vec_body"], rv["query_vec"])
    #
    #
    # rv["query_title_norm"] = np.divide(rv["query_title_sum"] ,rv["len_title_idf"] )
    # rv["query_body_norm"] = np.divide(rv["query_body_sum"] ,rv["len_body_idf"] )
    #
    # rv["query_title_sum"] = np.divide(rv["query_title_sum"] ,rv["title_count"] +1 )
    # rv["query_body_sum"] = np.divide(rv["query_body_sum"] ,rv["body_count"] + 1 )
    #
    #




    ''' Step 6. Defining the feature and label  for classification'''

    # X = rv[["avg_cos_title"] + ["cosine_title"] + ["cosine_title_w2v"] + ["common_title"]  + ["common_title_s"] + ["avg_cos_body"] + [
    #     "cosine_body"] + ["cosine_body_w2v"] + ["common_body"]+ ["common_body_s"]
    #
    #
    #        + ["cosine_title_pblob"] + ["cosine_title_sblob"] + ["cosine_body_pblob"] + ["cosine_body_sblob"]
    #        + ["max_query_idf"] + ["max_pos_query_idf"] + ["norm_query_idf"] + ["len_query_idf"] + ["prob_query_idf"]+["mean_query_idf"]
    #        + ["max_title_idf"] + ["max_pos_title_idf"] + ["norm_title_idf"] + ["len_title_idf"] + ["prob_title_idf"]+["mean_title_idf"]
    #        + ["max_body_idf"] + ["max_pos_body_idf"] + ["norm_body_idf"] + ["len_body_idf"] + ["prob_body_idf"]+["mean_body_idf"]
    #        + ["query_vec_blob_p"] + ["query_vec_blob_s"]
    #        + ['doc_vec_blob_p'] + ['doc_vec_blob_s']
    #        + ['body_vec_blob_p'] + ['body_vec_blob_s']
    #
    #        # + ["query_title_max"] + ["query_body_max"]
    #        # + ["query_title_norm"] + ["query_body_norm"]
    #        # + ["query_title_prob"]+ ["query_body_prob"]
    #        #  +  ["query_title_sum"] +["query_body_sum"]
    #        # + ["query_title_max_pos"] + ["query_body_max_pos"]
    #
    #
    # ]
    #
    #

    Y = [v for k, v in rv["position"].items()]

    ''' Step 7. Splitting the data for validation'''
    X_train, X_test, y_train, y_test = train_test_split(    rv ["all_text"], Y, test_size = 0.33, random_state = 42)

    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    for y in y_train:
        if y == 1:
            Y1.append("pos")
            Y2.append("neg")
            Y3.append("neg")
            Y4.append("neg")
        if y == 2:
            Y1.append("neg")
            Y2.append("pos")
            Y3.append("neg")
            Y4.append("neg")
        if y == 3:
            Y1.append("neg")
            Y2.append("neg")
            Y3.append("pos")
            Y4.append("neg")
        if y == 4:
            Y1.append("neg")
            Y2.append("neg")
            Y3.append("neg")
            Y4.append("pos")
    from textblob.classifiers import NaiveBayesClassifier
    c1 = NaiveBayesClassifier(np.column_stack((X_train, Y1)))

    # for x in X_test:
    #     p1 = c1.prob_classify(x)
    #     print(p1.prob("pos"))
    #     print(p1.prob("neg"))
    #     print(round(p1.prob("pos"), 2))
    #     print(round(p1.prob("neg"), 2))
    #     exit(0)

    c2 = NaiveBayesClassifier(np.column_stack((X_train, Y2)))
    c3 = NaiveBayesClassifier(np.column_stack((X_train, Y3)))
    c4 = NaiveBayesClassifier(np.column_stack((X_train, Y4)))

    results = []
    print("training Complete.... Please wait")
    for tokens in  X_test:
        # print(tokens)
        place_holder = np.zeros(5)
        p1 = c1.prob_classify(tokens)
        p2 = c2.prob_classify(tokens)
        p3 = c3.prob_classify(tokens)
        p4 = c4.prob_classify(tokens)
        p1.max()
        p2.max()
        p3.max()
        p4.max()
        place_holder[1] += p1.prob("pos")
        place_holder[2] += p2.prob("pos")
        place_holder[3] += p3.prob("pos")
        place_holder[4] += p4.prob("pos")
        results.append(np.argmax(place_holder))

    ''' Step 8. Classification and validation'''
    target_names = ['1', '2', '3','4']
    print("Results", results)
    # from sklearn.cluster import KMeans
    #
    # kmeans = KMeans(n_clusters=5).fit(X)
    # # y_kmeans = kmeans.predict(X)
    # print(classification_report(y_test, kmeans.predict(X_test), target_names=target_names))

    from sklearn.svm import LinearSVC
    # clf = RandomForestClassifier().fit(X_train, y_train)
    # clf = RandomForestClassifier(random_state=0, n_estimators=3000).fit(X_train, y_train)

    # clf = RandomForestClassifier(class_weight="balanced").fit(X_train, y_train)

    # clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)

    # clf = RandomForestClassifier(class_weight="balanced_subsample").fit(X_train, y_train)
    # clf = LinearSVC(random_state=0).fit(X_train, y_train)
    # clf = SVC(random_state=0).fit(X_train, y_train)

    # model = RandomForestRegressor(n_estimator=100, oob_score=TRUE, n_jobs=-1, random_state=50, max_features="auto",
    #                               min_samples_leaf=50



    # X = preprocessing.normalize(X)


    print(classification_report(y_test,  results, target_names=target_names))

    ''' Step 9. Saving the data '''
    joblib.dump(clf, 'resources/classifier.pkl')

#  Close but not using
    # Y = [v for k, v in rv["position"].items()]
    # Y = rv["position"]
    # clf = LogisticRegression().fit(X_train, y_train)
    # LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    #                    intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    # clf = RandomForestClassifier(class_weight="balanced").fit(X_train, y_train)

    # clf = RandomForestClassifier(class_weight= [{0: 1, 1: 50}, {0: 1, 1: 25}, {0: 1, 1: 10}, {0: 1, 1: 5}] ).fit(X_train, y_train)

    # clf = LinearSVC(random_state=0).fit(X_train, y_train)
    # clf = LogisticRegression().fit(X_train, y_train)
    #
    # LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    #                    intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
# NOT USING
    # clf = MultinomialNB().fit(X_train, y_train)
    # clf = OneVsRestClassifier(MultinomialNB()).fit(X_train, y_train)
    # clf = MultinomialNB().fit(X_train, y_train)
    # clf = RandomForestClassifier().fit(X_train, y_train)
    # from sklearn.svm import LinearSVC
    # clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
    # clf = SVC().fit(X_train, y_train)


if __name__ == '__main__':
    create_model("resources/cranfield_data.json", "resources/cranqrel.json", "resources/cran.qry.json")