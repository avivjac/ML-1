import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    classifier = {
        "k": k,
        "x_train": np.array(x_train),
        "y_train": np.array(y_train).astype(int).reshape(-1)
    }
    return classifier

def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    k = classifier["k"]
    x_train = classifier["x_train"]
    y_train = classifier["y_train"]

    x_test = np.array(x_test)
    n = x_test.shape[0]
    y_pred = np.zeros((n, 1), dtype=int)

    for i in range(n):
        dists = np.linalg.norm(x_train - x_test[i], axis=1)
        nn_idx = np.argsort(dists)[:k]
        nn_labels = y_train[nn_idx]
        counts = np.bincount(nn_labels)
        y_pred[i, 0] = np.argmax(counts)

    return y_pred


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")

# Q2
def load_mnist_1_3_4_6():
    data = np.load('mnist_all.npz')
    digits = [1, 3, 4, 6]

    train_arrays = [data[f'train{d}'] for d in digits]
    test_arrays  = [data[f'test{d}']  for d in digits]

    X_test = np.vstack(test_arrays)
    y_test = np.concatenate([
        d * np.ones(arr.shape[0], dtype=int)
        for d, arr in zip(digits, test_arrays)
    ])

    return train_arrays, digits, X_test, y_test

def q2_a():
    train_arrays, digits, X_test, y_test = load_mnist_1_3_4_6()

    train_sizes = [1, 2, 5, 10, 20, 30, 50, 70, 100]
    n_repeats = 10

    mean_errors = []
    min_errors = []
    max_errors = []

    for m in train_sizes:
        errs = []
        for rep in range(n_repeats):
            X_train, y_train = gensmallm(train_arrays, digits, m)
            clf = learnknn(1, X_train, y_train)
            y_pred = predictknn(clf, X_test).reshape(-1)  
            err = np.mean(y_pred != y_test)
            errs.append(err)

        errs = np.array(errs)
        mean_errors.append(errs.mean())
        min_errors.append(errs.min())
        max_errors.append(errs.max())

    mean_errors = np.array(mean_errors)
    min_errors = np.array(min_errors)
    max_errors = np.array(max_errors)

    yerr_lower = mean_errors - min_errors
    yerr_upper = max_errors - mean_errors

    plt.figure()
    plt.errorbar(train_sizes, mean_errors,
                 yerr=[yerr_lower, yerr_upper],
                 fmt='-o')
    plt.xlabel("Training sample size m")
    plt.ylabel("Test error")
    plt.title("Q2(a): 1-NN test error vs training size")
    plt.grid(True)
    plt.show()

def q2_d():
    train_arrays, digits, X_test, y_test = load_mnist_1_3_4_6()

    ms = [50, 150, 500]   
    ks = list(range(1, 16))
    n_repeats = 30

    for m in ms:
        mean_err_per_k = []

        for k in ks:
            errs = []
            for rep in range(n_repeats):
                X_train, y_train = gensmallm(train_arrays, digits, m)
                clf = learnknn(k, X_train, y_train)
                y_pred = predictknn(clf, X_test).reshape(-1)
                err = np.mean(y_pred != y_test)
                errs.append(err)

            mean_err_per_k.append(np.mean(errs))

        plt.figure()
        plt.plot(ks, mean_err_per_k, marker='o')
        plt.xlabel("k")
        plt.ylabel("Test error")
        plt.title(f"Q2(d): Test error vs k (m={m})")
        plt.grid(True)
        plt.show()

def corrupt_labels(y, labels, noise_rate=0.3):
    """
    מקבלת וקטור תוויות y ומחזירה עותק שבו noise_rate מהדוגמאות
    קיבלו תווית אחרת אקראית מתוך labels \ {label המקורי}.
    """
    y = np.array(y).astype(int).reshape(-1)
    n = y.shape[0]

    corrupted = y.copy()
    labels = np.array(labels, dtype=int)

    # בוחרים אילו אינדקסים "להרוס"
    mask = np.random.rand(n) < noise_rate
    idxs = np.where(mask)[0]

    for i in idxs:
        current = corrupted[i]
        other_labels = labels[labels != current]
        corrupted[i] = np.random.choice(other_labels)

    return corrupted

def q2_e():
    train_arrays, digits, X_test, y_test_clean = load_mnist_1_3_4_6()

    ms = [50, 150, 500]      
    ks = list(range(1, 16))
    n_repeats = 30

    for m in ms:
        mean_err_per_k = []

        for k in ks:
            errs = []

            for rep in range(n_repeats):
                X_train, y_train_clean = gensmallm(train_arrays, digits, m)

                # noise
                y_train_noisy = corrupt_labels(y_train_clean, digits, noise_rate=0.3)
                y_test_noisy  = corrupt_labels(y_test_clean,  digits, noise_rate=0.3)

                clf = learnknn(k, X_train, y_train_noisy)
                y_pred = predictknn(clf, X_test).reshape(-1)

                err = np.mean(y_pred != y_test_noisy)
                errs.append(err)

            mean_err_per_k.append(np.mean(errs))

        plt.figure()
        plt.plot(ks, mean_err_per_k, marker='o')
        plt.xlabel("k")
        plt.ylabel("Test error (30% label noise)")
        plt.title(f"Q2(e): Test error vs k (m={m}, noisy labels)")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':

    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    q2_d()

