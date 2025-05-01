import gzip
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

output_data = 'Books.txt'
gz_file = 'reviews_Books_5.json.gz'

for l in tqdm(parse(gz_file), unit='lines'):
    line += 1
    countU[l['reviewerID']] += 1
    countP[l['asin']] += 1

usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
User = dict()
pbar = tqdm(parse(gz_file), total=line, unit='lines')
for l in pbar:
    rev = l['reviewerID']
    asin = l['asin']
    time = l['unixReviewTime']
    if countU[rev] < 5 or countP[asin] < 5:
        continue

    if rev in usermap:
        userid = usermap[rev]
    else:
        usernum += 1
        userid = usernum
        usermap[rev] = userid
        User[userid] = []
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemnum += 1
        itemid = itemnum
        itemmap[asin] = itemid
    User[userid].append([time, itemid])
    pbar.set_postfix_str('usernum: %d, itemnum: %d' % (usernum, itemnum))
# sort reviews in User according to time

for userid in User.keys():
    User[userid].sort(key=lambda x: x[0])

print(usernum, itemnum)

f = open(output_data, 'w')
for user in User.keys():
    for i in User[user]:
        f.write('%d %d\n' % (user, i[1]))
f.close()