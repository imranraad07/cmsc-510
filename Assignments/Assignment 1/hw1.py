import matplotlib.pyplot as plt
import numpy as np


def main(x, y, number_of_iterations, gamma, polynomial_degree, tolerance):
    total_samples_m = x.shape[0]
    w = np.zeros(polynomial_degree + 1)
    y_predicted = np.zeros(total_samples_m)
    mean_squared_error = 0
    prev_error = sum(y) / total_samples_m
    gradientFactor = np.ones((polynomial_degree + 1, total_samples_m), dtype=float)
    for i in range(total_samples_m):
        for j in range(1, polynomial_degree + 1):
            gradientFactor[j][i] = x[i] ** j

    for idx in range(number_of_iterations):
        const = np.dot(w, gradientFactor) - y
        gradient = 2 * np.dot(gradientFactor, const)
        avg_gradient = gradient / total_samples_m
        w = w - gamma * avg_gradient
        y_predicted = np.matmul(w, gradientFactor)
        loss = (y_predicted - y) ** 2
        mean_squared_error = sum(loss) / total_samples_m
        if abs(prev_error - mean_squared_error) < tolerance:
            print("iteration ended at: ", idx)
            break
        prev_error = mean_squared_error
        # print(mean_squared_error)

    print("coefficients: ", w)
    print("mean_squared_error", mean_squared_error)
    # print(y_predicted)
    plt.scatter(x, y_predicted, color='blue', label='Predicted')
    plt.scatter(x, y, color='red', label='Actual')
    plt.xlabel('x')
    plt.ylabel('y')
    title = str(polynomial_degree) + ' Degree Polynomial'
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    x = np.array([-1.67245526, -2.36540279, -2.14724263, 1.40539096, 1.24297767, -1.71043904,
                  2.31579097, 2.40479939, -2.22112823])
    y = np.array(
        [-18.56122168, -24.99658931, -24.41907817, -2.688209, -1.54725306, -19.18190097,
         1.74117419, 3.97703338, -24.80977847])

    gamma = [0.001, 0.001, 0.001, 0.00008, 0.00008]
    number_of_iterations = 10000000
    tolerance = 1e-9
    for polynomial_degree in range(1, 6):
        main(x, y, number_of_iterations, gamma[polynomial_degree - 1], polynomial_degree, tolerance)
