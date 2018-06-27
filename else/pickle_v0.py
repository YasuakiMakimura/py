import pickle


def w(data, filename):
    with open(f'{filename}.pickle', 'wb') as f:
        pickle.dump(data, f)


def r(filename):
    print(f'{filename}.pickle')
    with open(f'{filename}.pickle', 'rb') as f:
        return pickle.load(f)
