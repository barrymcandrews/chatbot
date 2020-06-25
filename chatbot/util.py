from typing import List
import re

def clean(string: str) -> List[str]:
    "converts text into machine-readable tokens"

    # string = re.sub(r'<[^>]*>', ' ', string)
    string = re.sub(r'(\.){2,}', ' <elipsis> ', string)
    string = re.sub(r'([0-1]?[0-9]|2[0-3]):[0-5][0-9]', ' <time> ', string)
    string = re.sub(r'(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', ' <url> ', string)
    string = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', ' <handle> ', string)
    string = re.sub(r'(¯\\_\(ツ\)_\/¯)', ' 🤷‍♂️ ', string)
    string = re.sub(r'(:[\/\\])', ' 😕 ', string)
    string = re.sub(r'(:\))|(\(:)', ' 😀 ', string)
    string = re.sub(r'(:\()|(\):)', ' 🙁 ', string)
    string = re.sub(r'[*\[\]#@^&$()\":{}`+=~|\|]"', ' ', string)
    string = re.sub(r'(,)', ' , ', string)
    string = re.sub(r'(\*)', ' * ', string)
    string = re.sub(r'(\")', ' " ', string)
    string = re.sub(r'(;)', ' ; ', string)
    string = re.sub(r'(:)', ' : ', string)
    string = re.sub(r'(\.)', ' . ', string)
    string = re.sub(r'[\)\(]', ' ', string)
    string = re.sub(r'([?!]*)((\?!)|(!\?))([?!]*)', ' ! ? ' , string)
    string = re.sub(r'(\?){1,}', ' ? ', string)
    string = re.sub(r'(\!){1,}', ' ! ' , string)
    string = re.sub(r'[\n\r]', ' ' , string)
    string = re.sub(r'[\-]', ' ' , string)
    string = re.sub(r'[\-]{2,}', ' -- ' , string)
    string = re.sub(r'([\ufff9-\uffff])', r' <unk> ', string, flags=re.UNICODE) # Unknown Character
    string = re.sub(r'([\U00010000-\U0010ffff])', r' \g<1> ', string, flags=re.UNICODE) # Emojis

    string = string.lower()

    # abbreviations
    string = re.sub(r'\bwht\b', " what ", string)
    string = re.sub(r"\br\b", " are ", string)
    string = re.sub(r"\bu\b", " you ", string)
    string = re.sub(r"\bb\b", " be ", string)
    string = re.sub(r"\bim\b", " i am ", string)
    string = re.sub(r"\bur\b", " your ", string)
    string = re.sub(r"\bill\b", " i will ", string)
    string = re.sub(r"\bive\b", " i have ", string)
    string = re.sub(r"\bcant\b", " can not ", string)

    #contractions
    string = re.sub(r"[\u2018-\u2019]", "'", string, flags=re.UNICODE)
    string = re.sub(r"i'm", "i am", string)
    string = re.sub(r"he's", "he is", string)
    string = re.sub(r"she's", "she is", string)
    string = re.sub(r"it's", "it is", string)
    string = re.sub(r"that's", "that is", string)
    string = re.sub(r"what's", "what is", string)
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
    string = re.sub(r"'s", " s ", string)

    string = re.sub(r'\'', ' ' , string)
    string = re.sub(r'[ \t]{2,}', ' ', string)

    ret = string.strip().split(' ')
    ret = [i for i in ret if i]

    if len(ret) == 0:
        return ['<unk>']

    return ret
