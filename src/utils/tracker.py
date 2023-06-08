import os
import json
import matplotlib.pyplot as plt


class Tracker:
    def __init__(self):
        self.y_loss = {}
        self.y_loss["train"] = []
        self.y_loss["val"] = []
        self.y_err = {}
        self.y_err["train"] = []
        self.y_err["val"] = []

        self.x_epoch = []

        self.epochs_completed = 0

    def plotLossGraph(self, current_epoch):
        self.fig = plt.figure()
        self.ax0 = self.fig.add_subplot(121, title="loss")
        self.ax1 = self.fig.add_subplot(122, title="top1err")
        self.x_epoch.append(current_epoch)
        self.ax0.plot(self.x_epoch, self.y_loss["train"], "bo-", label="train")
        # self.ax0.plot(self.x_epoch, self.y_loss["val"], "ro-", label="val")
        # self.ax1.plot(self.x_epoch, self.y_err["train"], "bo-", label="train")
        # self.ax1.plot(self.x_epoch, self.y_err["val"], "ro-", label="val")
        if current_epoch == 0:
            self.ax0.legend()
            self.ax1.legend()
        self.fig.savefig(os.path.join("./output", "train.jpg"))

    def save(self, file_path):
        # create a dictionary to hold the variables
        data = {"y_loss": self.y_loss, "y_err": self.y_err, "x_epoch": self.x_epoch}

        # save the data to a json file
        with open(file_path, "w") as file:
            json.dump(data, file)

    def load(self, file_path):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except IOError:
            print("Error: File '{}' not found".format(file_path))

        self.y_loss = data["y_loss"]
        self.y_err = data["y_err"]
        self.x_epoch = data["x_epoch"]
