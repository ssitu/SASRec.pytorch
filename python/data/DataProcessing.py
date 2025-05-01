import gzip
from collections import defaultdict
from datetime import datetime


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

output_data = 'Books.txt'
gz_file = 'reviews_Books_5.json.gz'

usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
User = dict()
for l in parse(gz_file):
    line += 1
    asin = l['asin']
    rev = l['reviewerID']
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
# sort reviews in User according to time

for userid in User.keys():
    User[userid].sort(key=lambda x: x[0])

print(usernum, itemnum)

f = open(output_data, 'w')
for user in User.keys():
    for i in User[user]:
        f.write('%d %d\n' % (user, i[1]))
f.close()