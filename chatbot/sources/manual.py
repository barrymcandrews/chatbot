#!/usr/bin/env python3
from convokit import Corpus, Speaker, Utterance
import os
import yaml


def build_manual_corpus() -> Corpus:
    print('Building corpus from manually created yml files...')

    manual_files = []
    for root, dirs, files in os.walk('data/manual'):
        manual_files.extend([os.path.join(root, f) for f in files])

    conversations = []
    for path in manual_files:
        with open(path) as f:
            cs = yaml.load(f.read())['conversations']
            for c in cs:
                conversations.append((c[0],c[1]))

    speakers = {'0': Speaker(id='0'), 'M1': Speaker(id='1')}


    utterances = []
    i = 0
    for prompt, response in conversations:
        id_1 = "M" + str(i)
        id_2 = "M" + str(i + 1)
        utts = [
            Utterance(
                id=id_1,
                text=prompt,
                speaker=speakers["M1"],
                root=id_1,
                reply_to=None
            ),
            Utterance(
                id=id_2,
                text=response,
                speaker=speakers["0"],
                root=id_1,
                reply_to=id_1,
            ),
        ]
        i = i + 2
        utterances.extend(utts)

    return Corpus(utterances=utterances)


def get_manual_corpus() -> Corpus:
    try:
        return Corpus(filename='build/manual')
    except:
        manual_corpus = build_manual_corpus()
        manual_corpus.dump(name='manual', increment_version=False, base_path='build')
        return manual_corpus


if __name__ == "__main__":
    corpus = get_manual_corpus()
    convos = [c for c in corpus.iter_conversations()]
    print(corpus)
    print("Number of conversations: " + str(len(convos)))
    print("Number of utterances: " + str(len([u for u in corpus.iter_utterances()])))
    for convo in convos[0:5]:
        print("Conversation:")
        for utt in convo.iter_utterances():
            print(str(utt.speaker.id) + " " + utt.text)
