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

    # plot data in 4 separate graphs
    for i in range(3, len(sizes)):

        dataSize = pd.read_csv("dataSize"+str(i)+".csv")

        # rename colummn correct adjustment to oracle adjustment
        dataSize = dataSize.rename(columns={"Correct\nadjustment": "Oracle",
                                            "Wrong\nbackdoor set": "Wrong\nconfounding\nadjustment",
                                            "Method": "Using\nC1-C4"})

        fig, ax = plt.subplots()
        ax.plot(list(range(0, width)), groundTruth, "r", linewidth=2)
        ax.legend(['Ground truth causal effect'])

        dataSize.boxplot(ax=ax)
        plt.title("Sample size = " + str(sizes[i]))
        plt.ylim([0, 0.9])

        plt.show()

    # code below plots all the data in one large graph

    # width = 18
    # groundTruth = [groundTruth] * width
    # aggregate_dataframe = pd.DataFrame()
    # for i in range(0, len(sizes)):
    #     dataSize = pd.read_csv("dataSize"+str(i)+".csv")

    #     # aggregate_dataframe["No missing\nadjustment\nsize "+str(sizes[i])] = dataSize["No missing\nadjustment"]
    #     # aggregate_dataframe["Wrong\nbackdoor set\nsize "+str(sizes[i])] = dataSize["Wrong\nbackdoor set"]
    #     # aggregate_dataframe["Method\nsize "+str(sizes[i])] = dataSize["Method"]
    #     # aggregate_dataframe["Correct\nadjustment\nsize "+str(sizes[i])] = dataSize["Correct\nadjustment"]

    #     aggregate_dataframe["I,\n"+str(sizes[i])] = dataSize["No missing\nadjustment"]
    #     aggregate_dataframe["II,\n"+str(sizes[i])] = dataSize["Wrong\nbackdoor set"]
    #     aggregate_dataframe["III,\n"+str(sizes[i])] = dataSize["Method"]
    #     aggregate_dataframe["IV,\n"+str(sizes[i])] = dataSize["Correct\nadjustment"]
    
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(list(range(0, width)), groundTruth, "r", linewidth=2)
    # ax.legend(['Ground truth causal effect'])

    # aggregate_dataframe.boxplot(ax=ax)
    # plt.title("Estimation Experiments")
    # plt.vlines(x=[4.5, 8.5, 12.5], ymin=-0.464, ymax=1.992, colors=["black"])

    # plt.show()
    