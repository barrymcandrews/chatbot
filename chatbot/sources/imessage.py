import sqlite3
import pandas as pd
from convokit import Corpus, Speaker, Utterance
import os


def build_imessage_corpus() -> Corpus:
    print('Building corpus from iMessages...')
    conn = sqlite3.connect(os.path.expanduser('~/Library/Messages/chat.db'))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Handles (AKA Speakers)
    cur.execute("select ROWID as handle_id, id as phone_number from handle")
    handles = [dict(x) for x in cur.fetchall()]
    speakers = {str(h['handle_id']): Speaker(id=str(h['handle_id']), meta=h) for h in handles}
    speakers.update({'0': Speaker(id='0', meta={'phone_number': '+12155889243'})})  # don't call me unless you want

    # Chats
    chats = pd.read_sql_query("select * from chat", conn)
    chats.rename(columns={'ROWID': 'chat_id', 'chat_identifier': 'chat_name'}, inplace=True)
    chat_cols = list(chats)
    chats[chat_cols] = chats[chat_cols].astype(str)

    # Messages
    messages = pd.read_sql_query("select * from message", conn)
    messages.rename(columns={'ROWID': 'message_id'}, inplace=True)
    messages = messages[['message_id', 'text', 'handle_id', 'date', 'is_from_me']]
    messages['sender_id'] = messages.apply(lambda r: r['handle_id'] if r['is_from_me'] == 0 else '0', axis=1)

    # Add chat data to messages
    chat_message_joins = pd.read_sql_query("select * from chat_message_join", conn)
    messages = pd.merge(messages, chat_message_joins[['chat_id', 'message_id']], on='message_id', how='left').dropna()
    messages['chat_id'] = messages['chat_id'].astype(int)
    cols = list(messages)
    messages[cols] = messages[cols].astype(str)


    utterances = []
    for _, chat in chats.iterrows():
        chat_messages = messages.loc[messages['chat_id'] == chat['chat_id']].sort_values(by=['date'])
        num_messages = len(chat_messages.index)

        if num_messages == 0:
            print("Warning: chat '%s' has no messages" % chat['chat_name'])
            continue

        root_msg = chat_messages.iloc[0]
        for i in range(num_messages):
            msg = chat_messages.iloc[i]
            last_msg_id = chat_messages.iloc[i]['message_id'] if i != 0 else None

            msg_utt = Utterance(
                id=msg['message_id'],
                text=msg['text'],
                speaker=speakers[msg['sender_id']],
                root=root_msg['message_id'],
                reply_to=last_msg_id,
                meta=msg
            )
            utterances.append(msg_utt)

    return Corpus(utterances=utterances)


def get_imessage_corpus() -> Corpus:
    try:
        return Corpus(filename='build/imessages')
    except:
        imessage_corpus = build_imessage_corpus()
        imessage_corpus.dump(name='imessages', increment_version=False, base_path='build')
        return imessage_corpus


if __name__ == "__main__":
    corpus = get_imessage_corpus()
    convos = [c for c in corpus.iter_conversations()]
    print(corpus)
    print("Number of conversations: " + str(len(convos)))
    print("Number of utterances: " + str(len([u for u in corpus.iter_utterances()])))
    for convo in convos[0:5]:
        print("Conversation:")
        for utt in convo.iter_utterances():
            print(str(utt.speaker.id) + " " + utt.text)
