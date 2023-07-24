"""
Foundations of Natural Language Processing

Assignment 1

Please complete functions, based on their doc_string description
and instructions of the assignment.

To test your code run:

```
[hostname]s1234567 python3 s1234567.py
```

Before submission executed your code with ``--answers`` flag
```
[hostname]s1234567 python3 s1234567.py --answers
```
include generated answers.py file.

Best of Luck!
"""
from collections import defaultdict, Counter

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk import MaxentClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import brown, ppattach  # import corpora

# Import the Twitter corpus and LgramModel
from nltk_model import *  # See the README inside the nltk_model folder for more information

# Import the Twitter corpus and LgramModel
from twitter.twitter import *

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()


# Some helper functions

def ppEandT(eAndTs):
    '''
    Pretty print a list of entropy-tweet pairs

    :type eAndTs: list(tuple(float,list(str)))
    :param eAndTs: entropies and tweets
    :return: None
    '''

    for entropy, tweet in eAndTs:
        print("{:.3f} [{}]".format(entropy, ", ".join(tweet)))


def compute_accuracy(classifier, data):
    """
    Computes accuracy (range 0 - 1) of a classifier.
    :type classifier: NltkClassifierWrapper or NaiveBayes
    :param classifier: the classifier whose accuracy we compute.
    :type data: list(tuple(list(any), str))
    :param data: A list with tuples of the form (list with features, label)
    :rtype float
    :return accuracy (range 0 - 1).
    """
    correct = 0
    for d, gold in data:
        predicted = classifier.classify(d)
        correct += predicted == gold
    return correct/len(data)


def apply_extractor(extractor_f, data):
    """
    Helper function:
    Apply a feature extraction method to a labeled dataset.
    :type extractor_f: (str, str, str, str) -> list(any)
    :param extractor_f: the feature extractor, that takes as input V, N1, P, N2 (all strings) and returns a list of features
    :type data: list(tuple(str))
    :param data: a list with tuples of the form (id, V, N1, P, N2, label)

    :rtype list(tuple(list(any), str))
    :return a list with tuples of the form (list with features, label)
    """
    r = []
    for d in data:
        r.append((extractor_f(*d[1:-1]), d[-1]))
    return r


class NltkClassifierWrapper:
    """
    This is a little wrapper around the nltk classifiers so that we can interact with them
    in the same way as the Naive Bayes classifier.
    """
    def __init__(self, classifier_class, train_features, **kwargs):
        """

        :type classifier_class: a class object of nltk.classify.api.ClassifierI
        :param classifier_class: the kind of classifier we want to create an instance of.
        :type train_features: list(tuple(list(any), str))
        :param train_features: A list with tuples of the form (list with features, label)
        :param kwargs: additional keyword arguments for the classifier, e.g. number of training tations.
        :return None
        """
        self.classifier_obj = classifier_class.train(
            [(NltkClassifierWrapper.list_to_freq_dict(d), c) for d, c in train_features], **kwargs)

    @staticmethod
    def list_to_freq_dict(d):
        """
        :param d: list(any)
        :param d: list of features
        :rtype dict(any, int)
        :return: dictionary with feature counts.
        """
        return Counter(d)

    def classify(self, d):
        """
        :param d: list(any)
        :param d: list of features
        :rtype str
        :return: most likely class
        """
        return self.classifier_obj.classify(NltkClassifierWrapper.list_to_freq_dict(d))

    def show_most_informative_features(self, n = 10):
        self.classifier_obj.show_most_informative_features(n)

# End helper functions

# ==============================================
# Section I: Language Identification [60 marks]
# ==============================================

# Question 1 [7 marks]
def train_LM(corpus):
    '''
    Build a bigram letter language model using LgramModel
    based on the all-alpha subset the entire corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :rtype: LgramModel
    :return: A padded letter bigram model based on nltk.model.NgramModel
    '''

    # subset the corpus to only include all-alpha tokens
    corpus_tokens = [x.lower() for x in corpus.words() if x.isalpha()]
    # Return a smoothed padded bigram letter language model
    return LgramModel(2,corpus_tokens,pad_left=True,pad_right=True)


# Question 2 [7 marks]
def tweet_ent(file_name, bigram_model):
    '''
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens

    :type file_name: str
    :param file_name: twitter file to process
    :rtype: list(tuple(float,list(str)))
    :return: ordered list of average entropies and tweets'''

    # Clean up the tweet corpus to remove all non-alpha
    # tokens and tweets with less than 5 (remaining) tokens
    list_of_tweets = xtwc.sents(file_name)

    all_alpha_tweets=[[y.lower() for y in x if y.isalpha()]for x in list_of_tweets]
    cleaned_list_of_tweets = [x for x in all_alpha_tweets if len(x)>=5]
    #print(cleaned_list_of_tweets)

    # Construct a list of tuples of the form: (entropy,tweet)
    #  for each tweet in the cleaned corpus, where entropy is the
    #  average word for the tweet, and return the list of
    #  (entropy,tweet) tuples sorted by entropy
    entropies=[(np.mean([bigram_model.entropy(y,pad_left=True,pad_right=True,perItem=True) for y in x]),x) for x in cleaned_list_of_tweets]
    sorted_entropies=sorted(entropies,key=lambda x:(x[0]))
    return sorted_entropies


# Question 3 [8 marks]
def open_question_3():
    '''
    Question: What differentiates the beginning and end of the list
    of tweets and their entropies?

    :rtype: str
    :return: your answer [500 chars max]
    '''
    return inspect.cleandoc("""The beginning of list of tweets are all in English, whereas the end of
                                list of tweets are all non-English. This is because the model is trained using a corpus that mainly consists of English, which
                                makes occurences of English words more likely. In addition, English are made of words while non-English languages such as Chinese
                                are made of characters. Tweets at the end are also much longer.""")[0:500]


# Question 4 [8 marks]
def open_question_4() -> str:
    '''
    Problem: noise in Twitter data

    :rtype: str
    :return: your answer [500 chars max]
    '''
    return inspect.cleandoc("""There are words made up of English letters that themselves are non-English words,
                                such as guten, morgen, we can increase the size of n so that we
                                can check the sequence of letters. For example, the bigrams (w,i) and (i,e) are both likely to occur in an
                                English word, but the trigram (w,i,e) is not. We can also check the percentage of English letters in a tweet and
                                remove those with few English letters, for example remove tweets with less than 10% of English letters.""")[0:500]


# Question 5 [15 marks]
def tweet_filter(list_of_tweets_and_entropies):
    '''
    Compute entropy mean, standard deviation and using them,
    likely non-English tweets in the all-ascii subset of list
    of tweets and their letter bigram entropies

    :type list_of_tweets_and_entropies: list(tuple(float,list(str)))
    :param list_of_tweets_and_entropies: tweets and their
                                    english (brown) average letter bigram entropy
    :rtype: tuple(float, float, list(tuple(float,list(str)), list(tuple(float,list(str)))
    :return: mean, standard deviation, ascii tweets and entropies,
             non-English tweets and entropies
    '''

    # Find the "ascii" tweets - those in the lowest-entropy 90%
    #  of list_of_tweets_and_entropies
    #print([x[0] for x in list_of_tweets_and_entropies])
    n=len(list_of_tweets_and_entropies)
    list_of_ascii_tweets_and_entropies = list_of_tweets_and_entropies[0:int(n*0.9)]

    # Extract a list of just the entropy values
    list_of_entropies = [x[0] for x in list_of_ascii_tweets_and_entropies]

    # Compute the mean of entropy values for "ascii" tweets
    mean = np.mean(list_of_entropies)

    # Compute their standard deviation
    standard_deviation = np.std(list_of_entropies)

    # Get a list of "probably not English" tweets, that is
    #  "ascii" tweets with an entropy greater than (mean + std_dev))
    threshold = mean+standard_deviation
    list_of_not_English_tweets_and_entropies = [x for x in list_of_ascii_tweets_and_entropies if x[0]>threshold]

    # Return mean, standard_deviation,
    #  list_of_ascii_tweets_and_entropies,
    #  list_of_not_English_tweets_and_entropies
    return mean,standard_deviation,list_of_ascii_tweets_and_entropies,list_of_not_English_tweets_and_entropies


# Question 6 [15 marks]
def open_question_6():
    """
    Suppose you are asked to find out what the average per word entropy of English is.
    - Name 3 problems with this question, and make a simplifying assumption for each of them.
    - What kind of experiment would you perform to estimate the entropy after you have these simplifying assumptions?
       Justify the main design decisions you make in your experiment.
    :rtype: str
    :return: your answer [1000 chars max]
    """
    return inspect.cleandoc("""1.We need to consider sequences of infinite length. To address this, we assume that English
                                as a stochastic process is stationary and ergodic. 2. The probability of upcoming words can be
                                dependent on events that were arbitrarily distant and time dependent. Hence, we assume that
                                the occurence of words is independent from others. 3. New text is bound
                                to contain words never seen before. Hence, we assume that we are dealing with a closed-world system.
                                Experiment: Take a long enough sequence and calculate its entropy rate, because it will contain many
                                other shorter sequences and each of theses shorter sequences will reoccur in the longer sequence
                                according to their probabilities. """)[:1000]


#############################################
# SECTION II - RESOLVING PP ATTACHMENT AMBIGUITY
#############################################

# Question 7 [15 marks]
class NaiveBayes:
    """
    Naive Bayes model with Lidstone smoothing (parameter alpha).
    """

    def __init__(self, data, alpha):
        """
        :type data: list(tuple(list(any), str))
        :param data: A list with tuples of the form (list with features, label)
        :type alpha: float
        :param alpha: \alpha value for Lidstone smoothing
        """
        self.vocab = self.get_vocab(data)
        self.alpha = alpha
        self.prior, self.likelihood = self.train(data, alpha, self.vocab)

    @staticmethod
    def get_vocab(data):
        """
        Compute the set of all possible features from the (training) data.
        :type data: list(tuple(list(any), str))
        :param data: A list with tuples of the form (list with features, label)
        :rtype: set(any)
        :return: The set of all features used in the training data for all classes.
        """
        features=sum([x[0] for x in data],[])
        return set(features)

    @staticmethod
    def train(data, alpha, vocab):
        """
        Estimates the prior and likelihood from the data with Lidstone smoothing.

        :type data: list(tuple(list(any), str))
        :param data: A list of tuples ([f1, f2, ... ], c) with the first element
                     being a list of features and the second element being its class.

        :type alpha: float
        :param alpha: \alpha value for Lidstone smoothing

        :type vocab: set(any)
        :param vocab: The set of all features used in the training data for all classes.


        :rtype: tuple(dict(str, float), dict(str, dict(any, float)))
        :return: Two dictionaries: the prior and the likelihood (in that order).
        We expect the returned values to relate as follows to the probabilities:
            prior[c] = P(c)
            likelihood[c][f] = P(f|c)
        """
        assert alpha >= 0.0

        F=len(vocab)

        # Compute raw frequency distributions
        labels=[x[1] for x in data]
        unique_labels, frequency_labels = np.unique(labels,return_counts = True)
        # Compute prior (MLE). Compute likelihood with smoothing.
        prior={}
        N=len(data)
        for i in range(len(unique_labels)):
            prior[unique_labels[i]]=frequency_labels[i]/N
        likelihood={}
        for c in unique_labels:
            likelihood[c]={}
        for x in data:
            for f in x[0]:
                for c in unique_labels:
                    if c==x[1]:
                        if f not in likelihood[c]:
                            likelihood[c][f]=1
                        else:
                            likelihood[c][f]+=1
                    elif f not in likelihood[c]:
                        likelihood[c][f]=0
        for c in unique_labels:
            count=sum(likelihood[c].values())
            for f in likelihood[c]:
                likelihood[c][f]=(likelihood[c][f]+alpha)/(count+F*alpha)
        return prior,likelihood

    def prob_classify(self, d):
        """
        Compute the probability P(c|d) for all classes.
        :type d: list(any)
        :param d: A list of features.
        :rtype: dict(str, float)
        :return: The probability p(c|d) for all classes as a dictionary.
        """
        results={}
        # print(d)
        # print(self.prior)
        total=0
        for c in self.likelihood:
            p=1
            for feature in d:
                if feature in self.likelihood[c]:
                    p*=self.likelihood[c][feature]
            p*=self.prior[c]
            total+=p
            results[c]=p
        for c in results:
            results[c]/=total
        return results

    def classify(self, d):
        """
        Compute the most likely class of the given "document" with ties broken arbitrarily.
        :type d: list(any)
        :param d: A list of features.
        :rtype: str
        :return: The most likely class.
        """
        results=self.prob_classify(d)
        max_class=''
        max_prob=0
        for c in results:
            if results[c]>max_prob:
                max_class=c
                max_prob=results[c]
        return max_class


# Question 8 [10 marks]
def open_question_8() -> str:
    """
    How do you interpret the differences in accuracy between the different ways to extract features?
    :rtype: str
    :return: Your answer of 500 characters maximum.
    """
    return inspect.cleandoc("""Extracting p yields higher accuracy than extracting v, n1 and n2. This shows that preposition is p is more important. We shall give more weights to prepositions in training. The fifth extractor yields the highest accuracy as it captures more information than the other four. The accuracy obtained using logistic regression is higher than Naive Bayes, this is because the naive bayes model assumes independence among features in the same class, which is not true in the context of English.""")[:500]


# Feature extractors used in the table:
# see your_feature_extractor for documentation on arguments and types.
def feature_extractor_1(v, n1, p, n2):
    return [v]


def feature_extractor_2(v, n1, p, n2):
    return [n1]


def feature_extractor_3(v, n1, p, n2):
    return [p]


def feature_extractor_4(v, n1, p, n2):
    return [n2]


def feature_extractor_5(v, n1, p, n2):
    return [("v", v), ("n1", n1), ("p", p),("n2",n2)]

def powerset(v, n1, p, n2):
    s=feature_extractor_5(v, n1, p, n2)
    x = len(s)
    answer=[]
    for i in range(1 << x):
        combination=[s[j] for j in range(x) if (i & (1 << j))]
        answer.append(lambda:combination[i])
    return answer

# Question 9.1 [5 marks]
def your_feature_extractor(v, n1, p, n2):
    """
    Takes the head words and produces a list of features. The features may
    be of any type as long as they are hashable.
    :type v: str
    :param v: The verb.
    :type n1: str
    :param n1: Head of the object NP.
    :type p: str
    :param p: The preposition.
    :type n2: str
    :param n2: Head of the NP embedded in the PP.
    :rtype: list(any)
    :return: A list of features produced by you.
    """
    return [("v", v), ("n1", n1),("p", p),("p1", p),(p=="of"),("v","p",v,p),("v1","p1",v,p),("n1","p",n1,p),("p","n2",p,n2),("n11","p1",n1,p),("p1","n21",p,n2),("v","p","n2",v,p,n2),("v1","p1","n21",v,p,n2)]

# Question 9.2 [10 marks]
def open_question_9():
    """
    Briefly describe your feature templates and your reasoning for them.
    Pick 3 examples of informative features and discuss why they make sense or why they do not make sense
    and why you think the model relies on them.
    :rtype: str
    :return: Your answer of 1000 characters maximum.
    """
    return inspect.cleandoc("""My feature template considers various combinations of features and puts much emphasis on prepositions.
                                This was inspired by the resuts shown in question 8 in which extracting p yields higher accuracy. It also specifically
                            considers whether the preposition is 'of', because it is often used to describe the ownership between two objects, hence it is more likely
                            that the PP consists of 'of' attaches to NP. Three informative features are -1.999 ('n1', 'this')==1 and label is 'N',
                            -1.948 ('n1', 'them')==1 and label is 'N', -2.516 ('n1', 'it')==1 and label is 'N'. This makes sense because all three words
                            are pronouns, which is unlikely to be attached by any PP. The PP is more likely to describe the verb in front of it. As shown by the data,
                            the three pronouns all have negative weights, which when switched on make prediction of NP less likely. The feature -2.167 ('p', 'of')==1 and label is 'V'
                            also supports the hypothesis mentioned earlier.""")[:1000]


"""
Format the output of your submission for both development and automarking.
!!!!! DO NOT MODIFY THIS PART !!!!!
"""

def answers():
    # Global variables for answers that will be used by automarker
    global ents, lm
    global best10_ents, worst10_ents, mean, std, best10_ascci_ents, worst10_ascci_ents
    global best10_non_eng_ents, worst10_non_eng_ents
    global answer_open_question_4, answer_open_question_3, answer_open_question_6,\
        answer_open_question_8, answer_open_question_9
    global ascci_ents, non_eng_ents

    global naive_bayes
    global acc_extractor_1, naive_bayes_acc, lr_acc, logistic_regression_model, dev_features

    print("*** Part I***\n")

    print("*** Question 1 ***")
    print('Building brown bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model built')

    print("*** Question 2 ***")
    ents = tweet_ent(twitter_file_ids, lm)
    print("Best 10 english entropies:")
    best10_ents = ents[:10]
    ppEandT(best10_ents)
    print("Worst 10 english entropies:")
    worst10_ents = ents[-10:]
    ppEandT(worst10_ents)

    print("*** Question 3 ***")
    answer_open_question_3 = open_question_3()
    print(answer_open_question_3)

    print("*** Question 4 ***")
    answer_open_question_4 = open_question_4()
    print(answer_open_question_4)

    print("*** Question 5 ***")
    mean, std, ascci_ents, non_eng_ents = tweet_filter(ents)
    print('Mean: {}'.format(mean))
    print('Standard Deviation: {}'.format(std))
    print('ASCII tweets ')
    print("Best 10 English entropies:")
    best10_ascci_ents = ascci_ents[:10]
    ppEandT(best10_ascci_ents)
    print("Worst 10 English entropies:")
    worst10_ascci_ents = ascci_ents[-10:]
    ppEandT(worst10_ascci_ents)
    print('--------')
    print('Tweets considered non-English')
    print("Best 10 English entropies:")
    best10_non_eng_ents = non_eng_ents[:10]
    ppEandT(best10_non_eng_ents)
    print("Worst 10 English entropies:")
    worst10_non_eng_ents = non_eng_ents[-10:]
    ppEandT(worst10_non_eng_ents)

    print("*** Question 6 ***")
    answer_open_question_6 = open_question_6()
    print(answer_open_question_6)
    #
    #
    print("*** Part II***\n")

    print("*** Question 7 ***")
    naive_bayes = NaiveBayes(apply_extractor(feature_extractor_5, ppattach.tuples("training")), 0.1)
    naive_bayes_acc = compute_accuracy(naive_bayes, apply_extractor(feature_extractor_5, ppattach.tuples("devset")))
    print(f"Accuracy on the devset: {naive_bayes_acc * 100}%")

    print("*** Question 8 ***")
    answer_open_question_8 = open_question_8()
    print(answer_open_question_8)

    # This is the code that generated the results in the table of the CW:
    #
    # A single iteration of suffices for logistic regression for the simple feature extractors.
    #
    # extractors_and_iterations = [feature_extractor_1, feature_extractor_2, feature_extractor_3, feature_extractor_4, feature_extractor_5]
    #
    # print("Extractor    |  Accuracy")
    # print("------------------------")

    # for i, ex_f in enumerate(extractors, start=1):
    #     training_features = apply_extractor(ex_f, ppattach.tuples("training"))
    #     dev_features = apply_extractor(ex_f, ppattach.tuples("devset"))
    #     a_logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=6, trace=0)
    #     lr_acc = compute_accuracy(a_logistic_regression_model, dev_features)
    #     print(f"Extractor {i}  |  {lr_acc*100}")

    print("*** Question 9 ***")
    training_features = apply_extractor(your_feature_extractor, ppattach.tuples("training"))
    dev_features = apply_extractor(your_feature_extractor, ppattach.tuples("devset"))
    logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=10)
    lr_acc = compute_accuracy(logistic_regression_model, dev_features)

    print("30 features with highest absolute weights")
    logistic_regression_model.show_most_informative_features(30)

    print(f"Accuracy on the devset: {lr_acc*100}")

    answer_open_question_9 = open_question_9()
    print("Answer to open question:")
    print(answer_open_question_9)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        from autodrive_embed import run, carefulBind
        import adrive1

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive1.extract_answers, errlog)
    else:
        answers()
