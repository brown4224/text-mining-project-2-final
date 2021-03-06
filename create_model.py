'''  Sean McGlincy '''
'''  Textmining:  Project 2'''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import scipy
from textblob import TextBlob
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

def idf_prob(doc_sparce):
    doc = scipy.sparse.coo_matrix(doc_sparce)
    dp = 1.0
    for doc_word, doc_idf in zip(doc.col, doc.data):
        dp *= doc_idf
    return dp

def convert_to_blob(text):
    opinion = TextBlob(text)
    return opinion

def sentiment(textblob):
    return textblob.sentiment

def polarity(textblob):
    return textblob.polarity

def subjectivity(textblob):
    return textblob.subjectivity

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
def text_length(X):
    return len(X) + 1

def evaluate_model(model, query, document):
    query = query['query']
    title = document['title']
    body = document['body']

    '''Note:    Steps Mirror those found in body of document'''
    ''' Step 5. Converting query and title to vectors and finding cosine similarity of the vectors'''
    query_vec = model[0].transform([query])
    title_vec = model[0].transform([title])
    body_vec = model[0].transform([body])

    ''' Step 5.1 Cosine'''
    cos_title = cosine_similarity(query_vec, title_vec)
    cos_body = cosine_similarity(query_vec, body_vec)

    ''' Step 5.2a IDF Extraction: Query'''
    mq = np.max(query_vec)
    sq = np.sum(query_vec)
    lq = text_length(query)
    sqn = sq / lq
    pq = idf_prob(query_vec)
    mnq = np.mean(query_vec)

    ''' Step 5.2b IDF Extraction: Query'''
    mt = np.max(title_vec)
    st = np.sum(title_vec)
    lt = text_length(title)
    stn = st / lt
    pt = idf_prob(title_vec)
    mnt = np.mean(title_vec)

    ''' Step 5.2c IDF Extraction: Query'''
    mb = np.max(body_vec)
    sb = np.sum(body_vec)
    lb = text_length(body)
    sbn = sb / lb
    pb = idf_prob(body_vec)
    mnb = np.mean(body_vec)

    ''' Step 5.3 Text Blob Object'''
    tbq  = convert_to_blob(query)
    tbt  = convert_to_blob(title)
    tbb  = convert_to_blob(body)

    ''' Step 5.4 Text Blob Polarity'''
    tbq_polarity = polarity(tbq)
    tbt_polarity = polarity(tbt)
    tbb_polarity = polarity(tbb)

    ''' Step 5.5 Text Blob Subjectivity'''
    tbq_subjectivity = subjectivity(tbq)
    tbt_subjectivity = subjectivity(tbt)
    tbb_subjectivity = subjectivity(tbb)

    ''' Step 6. Defining the feature and label  for classification'''
    result = model[1].predict([[ cos_title, cos_body,
                                 tbq_polarity, tbt_polarity, tbb_polarity,
                                 tbq_subjectivity, tbt_subjectivity, tbb_subjectivity,
                                 mq, sqn, lq, pq, mnq,
                                 mt, stn, lt, pt, mnt,
                                 mb, sbn, lb, pb, mnb
                                 ]])
    return result[0],0.5


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
    vectorizer = TfidfVectorizer( min_df=0.0, max_df=1.0, stop_words="english", lowercase=True, norm="l2", strip_accents='ascii')
    vectorizer.fit(rv["all_text"])

    ''' Step 4. Saving the model for TF features'''
    joblib.dump(vectorizer, 'resources/vectorizer.pkl')

    ''' Step 5. Converting query and title to vectors and finding cosine similarity of the vectors'''
    rv["query_vec"] = rv.apply(lambda x: vectorizer.transform([x["query"]]), axis =1)
    rv["doc_vec_title"] = rv.apply(lambda x: vectorizer.transform([x["title"] ]), axis =1)
    rv["doc_vec_body"] = rv.apply(lambda x: vectorizer.transform([ x["body"]]), axis =1)

    ''' Step 5.1 Cosine'''
    rv["cosine_title"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_title'], x['query_vec']), axis=1)
    rv["cosine_body"]  = rv.apply(lambda x: cosine_similarity(x['doc_vec_body'], x['query_vec']), axis=1)

    ''' Step 5.2a IDF Extraction: Query'''
    rv["max_query_idf"] = rv.apply(lambda x: np.max(x["query_vec"]), axis=1)
    rv["sum_query_idf"] = rv.apply(lambda x: np.sum(x["query_vec"]), axis=1)
    rv["len_query_idf"] = rv.apply(lambda x: text_length(x["query"]), axis=1)
    rv["norm_query_idf"] = np.divide(rv["sum_query_idf"], rv["len_query_idf"])
    rv["prob_query_idf"] = rv.apply(lambda x: idf_prob(x["query_vec"]), axis =1)
    rv["mean_query_idf"] = rv.apply(lambda x: np.mean(x["query_vec"]), axis=1)

    ''' Step 5.2b IDF Extraction: Query'''
    rv["max_title_idf"] = rv.apply(lambda x: np.max(x["doc_vec_title"]), axis =1)
    rv["sum_title_idf"] = rv.apply(lambda x: np.sum(x["doc_vec_title"]), axis =1)
    rv["len_title_idf"] = rv.apply(lambda x: text_length(x["title"]), axis =1)
    rv["norm_title_idf"] = np.divide(rv["sum_title_idf"] ,rv["len_title_idf"] )
    rv["prob_title_idf"] = rv.apply(lambda x: idf_prob(x["doc_vec_title"]), axis =1)
    rv["mean_title_idf"] = rv.apply(lambda x: np.mean(x["doc_vec_title"]), axis =1)

    ''' Step 5.2c IDF Extraction: Query'''
    rv["max_body_idf"] = rv.apply(lambda x: np.max(x["doc_vec_body"]), axis =1)
    rv["sum_body_idf"] = rv.apply(lambda x: np.sum(x["doc_vec_body"]), axis =1)
    rv["len_body_idf"] = rv.apply(lambda x: text_length(x["body"]), axis =1)
    rv["norm_body_idf"] = np.divide(rv["sum_body_idf"] ,rv["len_body_idf"] )
    rv["prob_body_idf"] = rv.apply(lambda x: idf_prob(x["doc_vec_body"]), axis =1)
    rv["mean_body_idf"] = rv.apply(lambda x: np.mean(x["doc_vec_body"]), axis =1)

    ''' Step 5.3 Text Blob Object'''
    rv["query_vec_blob"] = rv.apply(lambda x: convert_to_blob(x["query"]), axis=1)
    rv["doc_vec_blob"] = rv.apply(lambda x: convert_to_blob(x["title"]), axis=1)
    rv["body_vec_blob"] = rv.apply(lambda x: convert_to_blob(x["body"]), axis=1)

    ''' Step 5.4 Text Blob Polarity'''
    rv["query_polarity"] = rv.apply(lambda x: polarity(x["query_vec_blob"]), axis=1)
    rv["title_polarity"] = rv.apply(lambda x: polarity(x["doc_vec_blob"]), axis=1)
    rv["body_polarity"] = rv.apply(lambda x: polarity(x["body_vec_blob"]), axis=1)

    ''' Step 5.5 Text Blob Subjectivity'''
    rv["query_subjectivity"] = rv.apply(lambda x: subjectivity(x["query_vec_blob"]), axis=1)
    rv["title_subjectivity"] = rv.apply(lambda x: subjectivity(x["doc_vec_blob"]), axis=1)
    rv["body_subjectivity"] = rv.apply(lambda x: subjectivity(x["body_vec_blob"]), axis=1)

    ''' Step 6. Defining the feature and label  for classification'''
    X = rv[  ["cosine_title"]        + ["cosine_body"]
           + ["query_polarity"]      + ["title_polarity"]     + ["body_polarity"]
           + ["query_subjectivity"]  + ["title_subjectivity"] + ["body_subjectivity"]
           + ["max_query_idf"]       + ["norm_query_idf"]     + ["len_query_idf"]     + ["prob_query_idf"] + ["mean_query_idf"]
           + ["max_title_idf"]       + ["norm_title_idf"]     + ["len_title_idf"]     + ["prob_title_idf"] + ["mean_title_idf"]
           + ["max_body_idf"]        + ["norm_body_idf"]      + ["len_body_idf"]      + ["prob_body_idf"]  + ["mean_body_idf"]
           ]
    Y = [v for k, v in rv["position"].items()]

    ''' Step 7. Splitting the data for validation'''
    X_train, X_test, y_train, y_test = train_test_split(    X, Y, test_size = 0.33, random_state = 42)

    ''' Step 8. Classification and validation'''
    target_names = ['1', '2', '3','4']

    # clf = GradientBoostingClassifier(n_estimators=1000).fit(X_train, y_train)
    clf = GradientBoostingClassifier().fit(X_train, y_train)

    print(classification_report(y_test,  clf.predict(X_test), target_names=target_names))

    ''' Step 9. Saving the data '''
    joblib.dump(clf, 'resources/classifier.pkl')


if __name__ == '__main__':
    create_model("resources/cranfield_data.json", "resources/cranqrel.json", "resources/cran.qry.json")