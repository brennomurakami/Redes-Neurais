import openpyxl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import seaborn as sns
import time


def plot_graphs_with_decision_lines(dados70, dados30, initial_weights, trained_weights):
    fig = plt.figure()

    # 1th graphic: dados70 with initial weights
    ax1 = fig.add_subplot(131, projection='3d')
    plot_3d_graph(ax1, dados70, "Training Data (70%)")
    plot_decision_line(ax1, initial_weights, label="Initial")
    ax1.set_title('Initial Random Weights')

    # 2nd graphic: dados70 with trained weights
    ax2 = fig.add_subplot(132, projection='3d')
    plot_3d_graph(ax2, dados70, "Training Data (70%)")
    plot_decision_line(ax2, trained_weights, label="Final")
    ax2.set_title('Trained Weights')

    # 3rd graphic: dados30 with trained weights
    ax3 = fig.add_subplot(133, projection='3d')
    plot_3d_graph(ax3, dados30, "Test Data (30%)")
    plot_decision_line(ax3, trained_weights, label="Final")
    ax3.set_title('Test with Trained Weights')

    plt.show()


def plot_decision_line(ax, weights, label):
    # Extract weights to a plan
    bias, w1, w2 = weights[0], weights[1], weights[2]

    # Defining limits for x1 and x2
    x1_min, x1_max = ax.get_xlim()
    x2_min, x2_max = ax.get_ylim()

    # Calculating the decision straight points
    x1_points = np.array([x1_min, x1_max])
    x2_points = (-bias - w1 * x1_points) / w2

    # Ploting decision straight
    ax.plot(x1_points, x2_points, [0, 0], label=label)


def plot_confusion_matrix(TP, TN, FP, FN):
    confusion_matrix = np.array([[TN, FP],
                                 [FN, TP]])

    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    # Adding legends
    plt.text(0.5, 1.95, 'False Positive (FP)', ha='center')
    plt.text(1.5, 1.95, 'True Positive (TP)', ha='center')
    plt.text(0.5, 0.1, 'True Negative (TN)', ha='center')
    plt.text(1.5, 0.1, 'False Negative (FN)', ha='center')

    plt.xticks([0.5, 1.5], ['0', '1'])
    plt.yticks([0.5, 1.5], ['0', '1'])
    plt.show()


def plot_3d_graph(ax, matriz, label):
    x1 = matriz[:, 0]
    x2 = matriz[:, 1]
    saida = matriz[:, 2]

    x1_saida0 = x1[saida == 0]
    x2_saida0 = x2[saida == 0]
    x1_saida1 = x1[saida == 1]
    x2_saida1 = x2[saida == 1]

    ax.scatter(x1_saida0, x2_saida0, c='b', marker='o', label='Output 0')
    ax.scatter(x1_saida1, x2_saida1, c='r', marker='x', label='Output 1')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Output')

    ax.legend()


def read_excel(filename):
    wb = openpyxl.load_workbook(filename)
    ws = wb.active

    data = []

    for row in ws.iter_rows(values_only=True):
        # read the values of the first 3 columns
        row_data = row[:3]
        data.append(row_data)

    return data


def activation_function(weighted_sum):
    if weighted_sum >= 0:
        return 1
    else:
        return 0


def matrix_multiply(matrix, weights):
    result = []
    for row in matrix:
        # Multiplying between data and weights
        row_result = [x * w for x, w in zip(row, weights)]
        result.append(row_result)
    return result


def perceptron_test(matrix, weights):
    TP = TN = FP = FN = 0
    correct_predictions = 0
    total_samples = len(matrix)

    for row in matrix:
        inputs = row[:2]  # Inputs
        target = row[2]   # Correct Outputs

        # Calculating weighted average of inputs
        weighted_sum = sum([inputs[i] * weights[i+1] for i in range(2)]) + weights[0]

        # Applying activation function
        output = activation_function(weighted_sum)

        # Updating confusion matrix counts
        if output == 1 and target == 1:
            TP += 1
        elif output == 0 and target == 0:
            TN += 1
        elif output == 1 and target == 0:
            FP += 1
        elif output == 0 and target == 1:
            FN += 1

        # Verifying if the output is correct
        if output == target:
            correct_predictions += 1

    # Calculating accuracy
    accuracy = correct_predictions / total_samples

    return accuracy, TP, TN, FP, FN


def perceptron_training(matrix, weights):

    # defining learning rate and desired error
    learning_rate = 0.2
    desired_error = 0

    # training loop
    epoch = 0
    while True:
        total_error = 0
        for row in matrix:
            inputs = row[:2]  # inputs
            target = row[2]   # outputs wanted

            # calculating weighted sums of inputs
            weighted_sum = sum([inputs[i] * weights[i+1] for i in range(2)]) + weights[0]

            # applying to the activation function
            output = activation_function(weighted_sum)

            # calculating error
            error = target - output
            total_error += abs(error)

            # updating weights
            for i in range(2):
                weights[i+1] += learning_rate * error * inputs[i]  # updating input weights
            weights[0] += learning_rate * error  # updating limiar weight

        # calculating the percent average of errors per epoch
        average_error = total_error / len(matrix)

        # verifying if the average error reached the desired percent
        if average_error <= desired_error:
            return weights, epoch + 1

        epoch += 1


# 70% data for training
filename = "dados70.xlsx"
dados70 = read_excel(filename)
# Convert the list in a numpy array
dados70 = np.array(dados70)

# starting random weights
initial_weights = [random.random() for _ in range(3)]  # 3 weights: 1 for bias e 1 for each input
print("Initial Weights: ", initial_weights)

# training weights and stating training timer, ending timer after training and calculating it
start_training_time = time.time()
trained_weights, epochs = perceptron_training(dados70, initial_weights.copy())
end_training_time = time.time()
training_duration = end_training_time - start_training_time
print("Weights trained:", trained_weights, "\nEpochs: ", epochs)

# 30% data for training
filename = "dados30.xlsx"
dados30 = read_excel(filename)
# Convert the list in a numpy array
dados30 = np.array(dados30)

# test trained perceptron, starting test step timer, ending timer and calculating it
start_testing_time = time.time()
accuracy, TP, TN, FP, FN = perceptron_test(dados30, trained_weights)
end_testing_time = time.time()
testing_duration = end_testing_time - start_testing_time

print("Perceptron precision:", accuracy*100, "%")

# calculating total duration
total_duration = training_duration + testing_duration

# printing all durations
print("Training time:", training_duration, "seconds")
print("Testing time:", testing_duration, "seconds")
print("Total time:", total_duration, "seconds")

# print the graphics and confusion matrix
plot_graphs_with_decision_lines(dados70, dados30, initial_weights, trained_weights)
plot_confusion_matrix(TP, TN, FP, FN)

