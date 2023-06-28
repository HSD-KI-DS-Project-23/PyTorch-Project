# Author: Leon Kleinschmidt

import os
import json
import matplotlib.pyplot as plt


class Tracker:
    """Eine Klasse, welche die Daten des Trainings abspeichert. Erlaubt so die Rekonstruktion des vollständigen Trainings, z.B. Loss-Plot auch wenn Training zwischenzeitig beendet wurde bzw. Jupyterkernel neugestartet wurde."""

    def __init__(self):
        self.y_loss = {}
        self.y_loss["train"] = []
        self.y_loss["val"] = []
        self.y_err = {}
        self.y_err["train"] = []
        self.y_err["val"] = []

        self.x_epoch = []
        self.epochs_completed = 0
        self.learning_rate = []

    def plotLossGraph(self):
        self.fig = plt.figure()
        self.ax0 = self.fig.add_subplot(121, title="loss")
        self.ax1 = self.fig.add_subplot(122, title="top1err")
        self.ax0.plot(self.x_epoch, self.y_loss["train"], "bo-", label="train")
        self.ax0.plot(self.x_epoch, self.y_loss["val"], "ro-", label="val")

        # self.ax1.plot(self.x_epoch, self.y_err["train"], "bo-", label="train")
        # self.ax1.plot(self.x_epoch, self.y_err["val"], "ro-", label="val")

        self.fig.savefig(os.path.join("./output", "train.jpg"))
        plt.show()
        plt.close()

    def save(self, file_path):
        # create a dictionary to hold the variables
        data = {
            "y_loss": self.y_loss,
            "y_err": self.y_err,
            "x_epoch": self.x_epoch,
            "learning_rate": self.learning_rate,
        }

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
        self.epochs_completed = data["x_epoch"][-1]
        self.learning_rate = data["learning_rate"]
