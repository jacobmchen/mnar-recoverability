import csv
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f = open("groundTruth.txt", "r")
    lines = f.readlines()
    f.close()
    groundTruth = float(lines[0])
    width = 6
    groundTruth = [groundTruth] * width

    sizes = [500, 2500, 5000, 10000]

    for i in range(len(sizes)):

        dataSize = pd.read_csv("dataSize"+str(i)+".csv")

        fig, ax = plt.subplots()
        ax.plot(list(range(0, width)), groundTruth, "r", linewidth=2)
        ax.legend(['Ground truth causal effect'])

        dataSize.boxplot(ax=ax)
        plt.title("Sample size = " + str(sizes[i]))

        plt.show()
    