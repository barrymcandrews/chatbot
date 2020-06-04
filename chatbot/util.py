from typing import List
import re

def clean(string: str) -> List[str]:
    "converts text into macine-readable tokens"
    string = re.sub(r'<[^>]*>|[*\[\]#@^&$()\":<>{}`+=~|]"', ' ', string)
    string = re.sub(r'(,)', ' ', string)
    string = re.sub(r'(;)', ' ', string)
    string = re.sub(r'(\.)', ' . ', string)
    string = re.sub(r'(\?)', ' ? ', string)
    string = re.sub(r'(\!)', ' ! ' , string)
    string = re.sub(r'[\-]', ' ' , string)
    string = re.sub(r'[\-]{2,}', ' -- ' , string)
    string = re.sub(r'[ \t]{2,}', ' ', string)

    #contractions
    string = string.lower()
    string = re.sub(r"i'm", "i am", string)
    string = re.sub(r"he's", "he is", string)
    string = re.sub(r"she's", "she is", string)
    string = re.sub(r"it's", "it is", string)
    string = re.sub(r"that's", "that is", string)
    string = re.sub(r"what's", "that is", string)
    string = re.sub(r"where's", "where is", string)
    string = re.sub(r"how's", "how is", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"won't", "will not", string)
    string = re.sub(r"can't", "cannot", string)
    string = re.sub(r"n't", " not", string)
    string = re.sub(r"n'", "ng", string)
    string = re.sub(r"'bout", "about", string)
    string = re.sub(r"'til", "until", string)
    string = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", string)
    return string.strip().split(' ')
