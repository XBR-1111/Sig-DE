import pickle


def load_pickle(path):
    """

    :param path:
    :return:
    """
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def dump_pickle(path, data):
    """

    :param path:
    :param data:
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)
