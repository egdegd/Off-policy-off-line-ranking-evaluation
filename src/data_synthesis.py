import pandas as pd
import numpy as np


def compute_reward(x, t, q):
    np.random.seed(1337)
    size = len(x)
    cov = np.zeros((size, size))
    for i in range(size):
        cov[i][i] = 1
    w = np.random.multivariate_normal(np.zeros(size), cov)
    for i in range(t - 1):
        check_q = np.random.rand()
        if check_q < q:
            w += np.random.multivariate_normal(np.zeros(size), cov)
    return 1 / (1 + np.exp(- w.T @ x))


def create_triples_from_context_vectors(vectors):
    data = []
    for t, sample in enumerate(vectors):
        a = sample[0]
        x = sample[1:]
        r = compute_reward(x, t, 0.8)
        data.append((x, a, round(r)))
    return data


def create_context_vector():
    def parse_line_with_slash(line):
        split_line = line.split()
        return split_line[0], split_line[1].split('|')

    descriptionid_tokensid = []
    purchasedkeywordid_tokensid = []
    queryid_tokensid = []
    titleid_tokensid = []
    userid_profile = []

    file_1 = open('data/track2/descriptionid_tokensid.txt', 'r')
    file_2 = open('data/track2/purchasedkeywordid_tokensid.txt', 'r')
    file_3 = open('data/track2/queryid_tokensid.txt', 'r')
    file_4 = open('data/track2/titleid_tokensid.txt', 'r')
    file_5 = open('data/track2/userid_profile.txt', 'r')

    for line in file_1:
        new_info = parse_line_with_slash(line)
        descriptionid_tokensid.append(new_info)

    for line in file_2:
        new_info = parse_line_with_slash(line)
        purchasedkeywordid_tokensid.append(new_info)

    for line in file_3:
        new_info = parse_line_with_slash(line)
        queryid_tokensid.append(new_info)

    for line in file_4:
        new_info = parse_line_with_slash(line)
        titleid_tokensid.append(new_info)

    for line in file_5:
        new_info = line.split()
        userid_profile.append(new_info)

    with open('data/track2/training.txt') as file:
        head = [next(file) for _ in range(10000)]
    train = np.array(list(map(lambda x: x.split(), head)))

    data = []
    for line in train:
        AdID = line(3)
        QueryID = int(line[7])
        KeywordID = int(line[8])
        TitleID = int(line[9])
        DescriptionID = int(line[10])
        UserID = int(line[11])
        assert int(queryid_tokensid[QueryID][0]) == QueryID
        assert int(purchasedkeywordid_tokensid[KeywordID][0]) == KeywordID
        assert int(titleid_tokensid[TitleID][0]) == TitleID
        assert int(descriptionid_tokensid[DescriptionID][0]) == DescriptionID
        user_info = [-1, -1]
        if UserID != 0:
            assert int(userid_profile[UserID][0]) == UserID
            user_info = userid_profile[UserID][1]

        data.append([AdID] + user_info + queryid_tokensid[QueryID][1] + purchasedkeywordid_tokensid[KeywordID][1] +
                    titleid_tokensid[TitleID][1] + descriptionid_tokensid[DescriptionID][1])

    file_1.close()
    file_2.close()
    file_3.close()
    file_4.close()
    file_5.close()

    path = 'data/track2/my_data.txt'
    file = open(path, 'w')
    for line in data:
        s = ' '.join(line)
        file.write(s + '\n')
    file.close()
    return path


def do_binary_vectors(path, size):
    file = open(path, 'r')
    data = []
    for line in file:
        split_line = line.split()
        context = np.zeros(size + 1)
        context[0] = int(split_line[0])  # context[0] = adId, context[1:3] = gender, context[4:10] - age
        gender = int(split_line[1])
        age = int(split_line[2])
        context[gender + 1] = 1
        context[age + 4] = 1
        for num in split_line[3:]:
            context[int(num) + 10] = 1
        data.append(context)
    return data


# It is a bad function. It will be deleted later.

def create_data_from_training_file(path):
    with open(path) as file:
        head = [next(file) for _ in range(10000)]
    train = np.array(list(map(lambda x: x.split(), head)))

    df = pd.DataFrame(train,
                      columns=['Click', 'Impression', 'DisplayURL', 'AdID', 'AdvertiserID', 'Depth', 'Position',
                               'QueryID',
                               'KeywordID', 'TitleID', 'DescriptionID', 'UserID'])

    # pd.options.display.max_columns = 12
    data = []

    for t, row in df.iterrows():
        x = row['UserID']
        a = row['AdID']
        r = compute_reward(x, a, 0.8)
        data.append((x, a, r))

    return data
