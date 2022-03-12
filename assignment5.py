import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-0.005 * x))

def sigmoid_derivative(x):
    return 0.005 * x * (1 - x)

def read_and_divide_into_train_and_test(csv_file):
    df = pd.read_csv(csv_file)
    if "Code_number" in df.columns:
        df.drop("Code_number", axis=1, inplace=True)
    df = df[df.Bare_Nuclei != '?']
    df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'])
    ax = plt.subplot()
    im = ax.imshow(df.drop("Class", axis=1).corr(), cmap=plt.cm.gist_heat)
    ax.set_xticks(np.arange(len(list(df.drop("Class", axis=1).columns))))
    ax.set_yticks(np.arange(len(list(df.drop("Class", axis=1).columns))))
    ax.set_xticklabels(list(df.drop("Class", axis=1).columns))
    ax.set_yticklabels(list(df.drop("Class", axis=1).columns))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    cbar = ax.figure.colorbar(im, ax=ax)
    for i in range(len(list(df.drop("Class", axis=1).columns))):
        for j in range(len(list(df.drop("Class", axis=1).columns))):
            text = ax.text(round(j, 2), i, (df.drop("Class", axis=1).corr().round(2)).to_numpy()[i, round(j, 2)],
                           ha="center", va="center", color="black")
    plt.show()

    training_inputs = df.iloc[:546, :-1].to_numpy()
    training_labels = df.iloc[:546, -1:].to_numpy()
    test_inputs = df.iloc[546:, :-1].to_numpy()
    test_labels = df.iloc[546:, -1:].to_numpy()
    return training_inputs, training_labels, test_inputs, test_labels


def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0
    total = 0
    test_outputs = sigmoid(np.dot(test_inputs, weights))
    test_predictions = test_outputs.round()

    for predicted_val, label in zip(test_predictions, test_labels):
        if predicted_val == label:
            tp += 1
        total += 1
    accuracy = tp / total
    return accuracy


def plot_loss_accuracy(accuracy_array, loss_array, itr):
    plt.plot(np.arange(0,itr),loss_array,label = "loss",color="red")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.plot(np.arange(0,itr),accuracy_array,label="accuracy")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def main():
    csv_file = './breast-cancer-wisconsin.csv'
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):
        input = training_inputs
        outputs = np.dot(input, weights)
        outputs = sigmoid(outputs)
        loss = training_labels - outputs
        tuning = loss * sigmoid_derivative(outputs)
        weights += np.dot(input.transpose(), tuning)
        accuracy_array.append(run_on_test_set(test_inputs, test_labels, weights))
        loss = np.mean(loss)
        loss_array.append(loss)

    plot_loss_accuracy(accuracy_array, loss_array, iteration_count)

if __name__ == '__main__':
    main()