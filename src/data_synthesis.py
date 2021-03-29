import pandas as pd
import numpy as np


def compute_reward(x, a):
    return 0


def create_data(path):
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
        r = compute_reward(x, a)
        data.append((x, a, r))

    return data
