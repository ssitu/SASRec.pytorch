import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def separate_by_popularity(user_sequences, popular_percentage):
    from collections import Counter
    item_interactions = Counter()
    user_interactions = Counter()
    for user_id, item_list in user_sequences.items():
        item_interactions.update(item_list)
        user_interactions[user_id] += len(item_list)
    items_cutoff = int(len(item_interactions) * popular_percentage)
    users_cutoff = int(len(user_interactions) * popular_percentage)
    sorted_items = sorted(item_interactions.items(), key=lambda x: x[1], reverse=True)
    sorted_users = sorted(user_interactions.items(), key=lambda x: x[1], reverse=True)
    popular_items = set(sorted([item for item, _ in sorted_items[:items_cutoff]]))
    unpopular_items = set(sorted([item for item, _ in sorted_items[items_cutoff:]]))
    popular_users = set(sorted([user for user, _ in sorted_users[:users_cutoff]]))
    unpopular_users = set(sorted([user for user, _ in sorted_users[users_cutoff:]]))
    return popular_items, unpopular_items, popular_users, unpopular_users

# train/val/test data generation
def data_partition(fname):
    # Initialize variables to track the number of users and items
    usernum = 0
    itemnum = 0

    # Dictionary to store user-item interactions
    User = defaultdict(list)

    # Dictionaries to store training, validation, and test data for each user
    user_train = {}
    user_valid = {}
    user_test = {}

    # Open the dataset file and read user-item interactions
    # Assume the file format is "user_id item_id" per line
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        # Parse the user and item IDs from the line
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)

        # Update the maximum user and item IDs encountered
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)

        # Append the item to the user's interaction list
        User[u].append(i)

    # Partition the data into train, validation, and test sets for each user
    for user in User:
        nfeedback = len(User[user])  # Number of interactions for the user

        # If the user has fewer than 3 interactions, use all for training
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            # Use all but the last two interactions for training
            user_train[user] = User[user][:-2]

            # Use the second-to-last interaction for validation
            user_valid[user] = []
            user_valid[user].append(User[user][-2])

            # Use the last interaction for testing
            user_test[user] = []
            user_test[user].append(User[user][-1])

    # Calculate popularity
    pi, ui, pu, uu = separate_by_popularity(User, 0.1)
    # Return the partitioned data along with the number of users and items
    # return [user_train, user_valid, user_test, usernum, itemnum]
    return [user_train, user_valid, user_test, usernum, itemnum], [pi, ui, pu, uu]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    # Deep copy the dataset to avoid modifying the original data
    [train, valid, test, usernum, itemnum], [pi, ui, pu, uu] = copy.deepcopy(dataset)

    # Initialize metrics for evaluation
    NDCG = 0.0  # Normalized Discounted Cumulative Gain
    HT = 0.0    # Hit Rate
    valid_user = 0.0  # Count of valid users considered
    total = set.union(pu, uu)  # Total set of users/items
    subsets = ['total', 'pu', 'uu', 'pi', 'ui']  # For storing results
    actual_subsets = {name: subset for name, subset in zip(subsets, [total, pu, uu, pi, ui])}  # The actual subsets
    valid_user_dict = {subset: 0 for subset in subsets}  # Count of valid users in each subset
    # Initialize NDCG and HT for different subsets and ranks
    NDCG_dict = {subset: {k: 0.0 for k in [5, 10, 20]} for subset in subsets}
    HT_dict = {subset: {k: 0.0 for k in [5, 10, 20]} for subset in subsets}

    # Limit the number of users to evaluate if usernum is large
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    # Iterate through the selected users
    for u in users:
        # Skip users with insufficient training or test data
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        # Prepare the sequence for the user
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]  # Add the validation item to the sequence
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        # Create a set of items already rated by the user
        rated = set(train[u])
        rated.add(0)  # Add 0 to represent padding

        # Prepare the list of items to evaluate
        item_idx = [test[u][0]]  # Include the test item
        for _ in range(100):  # Add 100 negative samples
            t = np.random.randint(1, itemnum + 1)
            while t in rated:  # Ensure the item is not already rated
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # Get predictions for the user, sequence, and items
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]  # Negate for descending order in argsort

        # Calculate the rank of the test item
        rank = predictions.argsort().argsort()[0].item()

        # Update metrics for valid users
        valid_user += 1
        if rank < 10:  # If the test item is in the top-10 predictions
            NDCG += 1 / np.log2(rank + 2)  # Update NDCG
            HT += 1  # Update Hit Rate

        for subset in subsets:
            if u in actual_subsets[subset]:
                valid_user_dict[subset] += 1
                # Update NDCG and Hit Rate for the subset
                for k in NDCG_dict[subset].keys():
                    if rank < k:
                        NDCG_dict[subset][k] += 1 / np.log2(rank + 2)
                        HT_dict[subset][k] += 1
        
        assert NDCG_dict['total'][10] == NDCG, "NDCG for total subset should match overall NDCG"
        assert HT_dict['total'][10] == HT, "Hit Rate for total subset should match overall Hit Rate"

        for subset in subsets:
            if valid_user_dict[subset] > 0:
                for k in NDCG_dict[subset].keys():
                    NDCG_dict[subset][k] /= valid_user_dict[subset]
                    HT_dict[subset][k] /= valid_user_dict[subset]

        # Print progress for every 100 valid users
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # Return the average NDCG and Hit Rate
    return (NDCG / valid_user, HT / valid_user), {"NDCG": NDCG_dict, "HT": HT_dict}


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
