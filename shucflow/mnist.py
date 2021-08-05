import gzip
import pickle
import cupy as cp


def _change_one_hot_label(X):
    T = cp.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

save_file = "D:\\Users\\chengshu\\PycharmProjects\\learncupy\\mnist.pkl"

def load_mnist(normalize=True, flatten=True, one_hot_label=True):
    with open(save_file,'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img','test_img'):
            dataset[key] = dataset[key].astype(cp.float32)
            dataset[key] /=255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])
