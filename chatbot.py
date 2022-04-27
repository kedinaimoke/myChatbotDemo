# import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open('chatbot.txt', 'r', errors='ignore')
plain_doc = f.read()
plain_doc = plain_doc.lower()  # so it converts text(s) to lowercase
nltk.download('punkt')  # using the 'Punkt' tokeniser
nltk.download('wordnet')  # using the 'WordNet dictionary
sentence_tokens = nltk.sent_tokenize(plain_doc)  # converts doc to list of sentences
word_tokens = nltk.word_tokenize(plain_doc)  # converts doc to list of words

lemmer = nltk.stem.WordNetLemmatizer()


def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def lem_normalise(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


greetings_in = ("hello", "yo", "what's good", "hi", "greetings", "what's up", "hey")
greetings_re = ["hey", "*nods*", "hi there!", "so glad i'm talking to you", "hello", "hi"]


def greet(sentence):
    for word in sentence.split():
        if word.lower() in greetings_in:
            return random.choice(greetings_re)


def response(user_response):
    robo1_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=lem_normalise, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo1_response += "My apologies! I do not understand you."
        return robo1_response
    else:
        robo1_response += sentence_tokens[idx]
        return robo1_response


flag = True
print("BOT: My name is Serendipi. Let's have a basic convo and if you're tired, just type Bye!")
while (flag == True):
    user_response = input()
    user_response = user_response.lower()
    if (user_response != 'bye'.casefold()):
        if (user_response == 'thanks'.casefold() or user_response == 'thank you'.casefold()):
            flag = False
            print("BOT: You're welcome!")
        else:
            if greet(user_response) != None:
                print(f"BOT: {greet(user_response)}")
            else:
                sentence_tokens.append(user_response)
                word_tokens += nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print("BOT: ", end="")
                print(response(user_response))
                sentence_tokens.remove(user_response)
    else:
        flag = False
        print("BOT: Byeee! Come back soon")
