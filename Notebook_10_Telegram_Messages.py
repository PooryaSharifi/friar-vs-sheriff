from telethon.sync import TelegramClient
from telethon.sessions import StringSession
import socks
from pymongo import MongoClient
from elmo import build_model, LMDataGenerator, DATA_SET_DIR, MODELS_DIR, parameters as elmo_parameters
import os.path
import tensorflow as tf
import time
import re
from bson.binary import Binary
import pickle
import numpy as np
import datetime as dt

"""
# Using cPickle with fast protocol=2.
collection.remove()
print("inserting with cpickle protocol 2")
%timeit collection.insert({'cpickle': Binary(cPickle.dumps(np.random.rand(50,3), protocol=2))})
print("reading cpickle protocol 2")
%timeit -n 100 [cPickle.loads(x['cpickle']) for x in collection.find()]
"""

regex = re.compile(r'\s(\s)+')
hashtag = re.compile(r"[@#](\w+)")
url = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
url = re.compile(r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
url = re.compile(r"""(?x)
\b
(							# Capture 1: entire matched URL
  (?:
    https?:				# URL protocol and colon
    (?:
      /{1,3}						# 1-3 slashes
      |								#   or
      [a-z0-9%]						# Single letter or digit or '%'
      								# (Trying not to match e.g. "URI::Escape")
    )
    |							#   or
    							# looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj| Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)
    /
  )
  (?:							# One or more:
    [^\s()<>{}\[\]]+						# Run of non-space, non-()<>{}[]
    |								#   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\)  # balanced parens, one level deep: (…(…)…)
    |
    \([^\s]+?\)							# balanced parens, non-recursive: (…)
  )+
  (?:							# End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\)  # balanced parens, one level deep: (…(…)…)
    |
    \([^\s]+?\)							# balanced parens, non-recursive: (…)
    |									#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]		# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
  	(?<!@)			# not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj| Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)
    \b
    /?
    (?!@)			# not succeeded by a @, avoid matching "foo.na" in "foo.na@example.com"
  )
)""")


def clean(t):
    t = url.sub('', t)
    t = hashtag.sub('', t)
    t = regex.sub('\n', t)
    return t


messages = MongoClient()['telegram_migrate']['messages']
messages.create_index([('channel', 1)])
messages.create_index([('_date', 1)])

configuration = {
    'api_id': 165248,
    'api_hash': '287208e1887c8e18f37d92a545a26376',
    'title': 'SheyRoon',
    'name': 'SheyRoon',
    'public_keys': """
-----BEGIN RSA PUBLIC KEY-----
MIIBCgKCAQEAwVACPi9w23mF3tBkdZz+zwrzKOaaQdr01vAbU4E1pvkfj4sqDsm6
lyDONS789sVoD/xCS9Y0hkkC3gtL1tSfTlgCMOOul9lcixlEKzwKENj1Yz/s7daS
an9tqw3bfUV/nqgbhGX81v/+7RFAEd+RwFnK7a+XYl9sluzHRyVVaTTveB2GazTw
Efzk2DWgkBluml8OREmvfraX3bkHZJTKX4EQSjBbbdJ2ZXIsRrYOXfaA+xayEGB+
8hdlLmAjbCVfaigxX0CDqWeR1yFL9kwd9P0NsZRPsmoqVwMbMu7mStFai6aIhc3n
Slv8kg9qv1m6XHVQY3PnEw+QQtqSIXklHwIDAQAB
-----END RSA PUBLIC KEY-----

-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAruw2yP/BCcsJliRoW5eB
VBVle9dtjJw+OYED160Wybum9SXtBBLXriwt4rROd9csv0t0OHCaTmRqBcQ0J8fx
hN6/cpR1GWgOZRUAiQxoMnlt0R93LCX/j1dnVa/gVbCjdSxpbrfY2g2L4frzjJvd
l84Kd9ORYjDEAyFnEA7dD556OptgLQQ2e2iVNq8NZLYTzLp5YpOdO1doK+ttrltg
gTCy5SrKeLoCPPbOgGsdxJxyz5KKcZnSLj16yE5HvJQn0CNpRdENvRUXe6tBP78O
39oJ8BTHp9oIjd6XWXAsp2CvK45Ol8wFXGF710w9lwCGNbmNxNYhtIkdqfsEcwR5
JwIDAQAB
-----END PUBLIC KEY-----

-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvfLHfYH2r9R70w8prHbl
Wt/nDkh+XkgpflqQVcnAfSuTtO05lNPspQmL8Y2XjVT4t8cT6xAkdgfmmvnvRPOO
KPi0OfJXoRVylFzAQG/j83u5K3kRLbae7fLccVhKZhY46lvsueI1hQdLgNV9n1cQ
3TDS2pQOCtovG4eDl9wacrXOJTG2990VjgnIKNA0UMoP+KF03qzryqIt3oTvZq03
DyWdGK+AZjgBLaDKSnC6qD2cFY81UryRWOab8zKkWAnhw2kFpcqhI0jdV5QaSCEx
vnsjVaX0Y1N0870931/5Jb9ICe4nweZ9kSDF/gip3kWLG0o8XQpChDfyvsqB9OLV
/wIDAQAB
-----END PUBLIC KEY-----

-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAs/ditzm+mPND6xkhzwFI
z6J/968CtkcSE/7Z2qAJiXbmZ3UDJPGrzqTDHkO30R8VeRM/Kz2f4nR05GIFiITl
4bEjvpy7xqRDspJcCFIOcyXm8abVDhF+th6knSU0yLtNKuQVP6voMrnt9MV1X92L
GZQLgdHZbPQz0Z5qIpaKhdyA8DEvWWvSUwwc+yi1/gGaybwlzZwqXYoPOhwMebzK
Uk0xW14htcJrRrq+PXXQbRzTMynseCoPIoke0dtCodbA3qQxQovE16q9zz4Otv2k
4j63cz53J+mhkVWAeWxVGI0lltJmWtEYK6er8VqqWot3nqmWMXogrgRLggv/Nbbo
oQIDAQAB
-----END PUBLIC KEY-----

-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvmpxVY7ld/8DAjz6F6q0
5shjg8/4p6047bn6/m8yPy1RBsvIyvuDuGnP/RzPEhzXQ9UJ5Ynmh2XJZgHoE9xb
nfxL5BXHplJhMtADXKM9bWB11PU1Eioc3+AXBB8QiNFBn2XI5UkO5hPhbb9mJpjA
9Uhw8EdfqJP8QetVsI/xrCEbwEXe0xvifRLJbY08/Gp66KpQvy7g8w7VB8wlgePe
xW3pT13Ap6vuC+mQuJPyiHvSxjEKHgqePji9NP3tJUFQjcECqcm0yV7/2d0t/pbC
m+ZH1sadZspQCEPPrtbkQBlvHb4OLiIWPGHKSMeRFvp3IWcmdJqXahxLCUS1Eh6M
AQIDAQAB
-----END PUBLIC KEY-----
    """,
    'MTProto_servers': {
        'test': '149.154.167.40:443',
        'production': '149.154.167.50:443'
    },
}

string = '1BJWap1sBu4ZJVy3q9RBtr2iis3DW2jOyz0zB-HlMucgUuFY9pn5Ma7fkYlaqxvIMhRNJJRkRw3wx1gbej1qjzoi0UMJj4q4yc4SPu9Q6q3VeJYppIXN1rMjevhpJNqPErOemKNNu0zGtv8a0fxkoJOQcy4FI_NfjX4xmHTaalRNgoHtf6fEphY8Q9vII0VrJmQjU4PXwK9zRClFYsBQQ5IekXFV2BoTNnltCySgmje4VCMgz4RuqkaozHR29gHlRFQ5iFppmwqEv0omXcglnaEwjTcA7FTLQSliVv3tuSbprG-iRoimXidXTtHf6-SdvdUz_69tGHfemBdcLWWOIKQV3Heb9Mv8='
"""
iter_messages(
    entity: hints.EntityLike, 
    limit: float = None, 
    *, 
    offset_date: hints.DateLike = None, 
    offset_id: int = 0, 
    max_id: int = 0, 
    min_id: int = 0, 
    add_offset: int = 0, 
    search: str = None, 
    filter: typing.Union[types.TypeMessagesFilter, 
    typing.Type[types.TypeMessagesFilter]] = None, 
    from_user: hints.EntityLike = None, 
    wait_time: float = None, 
    ids: typing.Union[int, typing.Sequence[int]] = None, 
    reverse: bool = False
) → typing.Union[_MessagesIter, _IDsIter]
"""

_channel = 'tobourseir'
_channel = 'bbcpersian'


def update_channel(channel):
    ms = []
    m = 0
    day = None
    n = 0
    words = 0
    try:
        last_ch_m = next(iter(messages.find({'channel': channel}).sort([('_date', -1)]).limit(1)))
    except:
        last_ch_m = {'id': 0}
    print(last_ch_m)
    with TelegramClient(StringSession(string), api_id=configuration['api_id'], api_hash=configuration['api_hash'], proxy=(socks.SOCKS5, '127.0.0.1', 9050)) as client:
        print(client.get_me())
        for message in client.iter_messages(channel, min_id=last_ch_m['id']):
            # print(message.id, ':', message.text)
            # break
            if message.text:
                ms.append({
                    'text': clean(message.text),
                    '_date': message.date,
                    'id': message.id,
                    '_reply': message.reply_to_msg_id,
                    'channel': channel,
                })
            words += len([word for word in message.text.split(' ') if len(word) > 1]) if message.text else 0
            if day != message.date.day:
                n += 1
                print(n, message.date.day)
            day = message.date.day
            # if n > 1000:
            #     break
            if len(ms) > 50:
                messages.insert_many(ms)
                m += len(ms)
                ms.clear()
        if len(ms):
            messages.insert_many(ms)
            m += len(ms)
            ms.clear()
        print(f'messages count {m}')
        print(f'words count {words}')


def update_elmo(channel, elmo, g, wrk=4):
    q = []

    def _update_elmo():
        t0 = time.time()
        vectors = g.encode([m['text'] for m in q])
        output_vectors = elmo.predict(vectors)[0].astype(np.float32)
        for i, m in enumerate(q):
            m['elmo'] = Binary(pickle.dumps(output_vectors[i], protocol=2))
            messages.save(m)
        print(f'time token for {len(q)} texts is {time.time() - t0} s')
        q.clear()
    for m in messages.find({'channel': channel, 'elmo': {'$exists': False}}):
        if m['text']:
            q.append(m)
        if len(q) == wrk:
            _update_elmo()
    if len(q):
        _update_elmo()


if __name__ == 'main__':
    wrk = 256
    ms = MongoClient()['telegram_migrate']['messages']
    q = []

    def migrate():
        print('hehe')
        ms.insert_many([{
            'text': clean(m['text']),
            '_date': m['_date'],
            'id': m['id'],
            '_reply': m['_reply'],
            'channel': m['channel'],
        } for m in q if m['text']])
        q.clear()
        print('huli')
    for m in messages.find({}):
        q.append(m)
        if len(q) == wrk:
            migrate()
    if len(q):
        migrate()


if __name__ == 'main__':
    g = LMDataGenerator(os.path.join(DATA_SET_DIR, elmo_parameters['train_dataset']),
                        os.path.join(DATA_SET_DIR, elmo_parameters['vocab']),
                        sentence_maxlen=elmo_parameters['sentence_maxlen'],
                        token_maxlen=elmo_parameters['token_maxlen'],
                        batch_size=elmo_parameters['batch_size'],
                        shuffle=elmo_parameters['shuffle'])

    elmo = build_model()
    elmo.load_weights(tf.train.latest_checkpoint(MODELS_DIR))
    for c in ['bbcpersian', 'farsivoa', 'ManotoTV', 'betasahm1', 'BOURSE_365']:
        update_channel(c)
        update_elmo(c, elmo, g, wrk=32)
    # with open('telegram.txt', 'w', encoding='utf-8') as w:
    #     with open('/home/poorya/Documents/telegram.csv', 'rU', encoding='utf-8') as f:
    #         reader = csv.reader(f)
    #         texts = [t[2] for t in list(reader)[1:] if t]
    #         # texts = [re.sub(r'\n(\n)+', '\n', text) for text in texts]
    #         text = '\n'.join(texts)
    #         text = re.sub(r'\n(\n)+', '\n', text)
    #         # texts[0] = ''.join([c for c in texts[0] if c not in punctuation])
    #     w.write(text)


if __name__ == '__main__':
    messages = MongoClient()['telegram_migrate']['messages']
    messages.create_index([('channel', 1)])
    messages.create_index([('_date', 1)])
    # messages.create_index([('elmo', 1)])
    now = dt.datetime.now()
    tomorrow = dt.datetime(year=now.year, month=now.month, day=now.day) + dt.timedelta(days=2)
    for i in range(1024):
        tomorrow = tomorrow - dt.timedelta(days=1)
        today = tomorrow - dt.timedelta(days=1)
        name = f'news/{today.year}-{0 if today.month < 10 else ""}{today.month}-{0 if today.day < 10 else ""}{today.day}.npy'
        if not os.path.exists(name):
            print(today, tomorrow)
            # ms = messages.find({'_date': {'$gte': today, '$lt': tomorrow}, 'elmo': {'$exists': True}})
            ms = messages.aggregate([
                {'$match': {'_date': {'$gte': today, '$lt': tomorrow}, 'elmo': {'$exists': True}}},
                # {'$match': {'elmo': {'$exists': True}}},
                {'$sample': {'size': 64}}
            ])
            elmo = [pickle.loads(m['elmo']) for m in ms]
            if not elmo:
                continue
            elmo = np.stack(elmo).astype(np.float32)
            np.save(name, elmo)
    # t0 = time.time()

