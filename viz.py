import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix


class Visualization:

    def __init__(self, df):
        """
        Initialize the Visualization class.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the dataset to visualize.
        """
        self.df = df
        self.sns = sns
        self.plt = plt

    def scatter(self, x, y, hue="default", title=None):
        """
        Generate a scatter plot to visualize the relationship between two variables.

        Parameters
        ----------
        x : str
            Name of the variable to be plotted on the x-axis.
        y : str
            Name of the variable to be plotted on the y-axis.
        hue : str, optional
            Column used for color encoding (default is "default").
        title : str, optional
            Custom title for the plot.

        Returns
        -------
        None
        """
        self.plt.figure(figsize=(8, 5))

        self.sns.scatterplot(
            data=self.df,
            x=x,
            y=y,
            hue=hue,
            palette=["#2c3e50", "#e74c3c"],
            marker="^",
            s=70,
            alpha=0.8
        )

        self.plt.title(title if title else f"{x} vs {y}", fontsize=14)
        self.plt.tight_layout()
        self.plt.show()

    def hist(self, column):
        """
        Plot the distribution of a single variable using a histogram.

        Parameters
        ----------
        column : str
            Name of the column to visualize.

        Returns
        -------
        None
        """
        self.plt.figure(figsize=(7, 5))
        self.sns.histplot(self.df[column], bins=30)
        self.plt.title(f"Distribution of {column}")
        self.plt.show()

    def box(self, x, y):
        """
        Create a boxplot to analyze the distribution of a variable across categories.

        Parameters
        ----------
        x : str
            Categorical variable.
        y : str
            Numerical variable.

        Returns
        -------
        None
        """
        self.plt.figure(figsize=(7, 5))
        self.sns.boxplot(x=x, y=y, data=self.df)
        self.plt.title(f"{y} by {x}")
        self.plt.show()

    def roc_curve(self, y_test, y_prob):
        """
        Plot the ROC curve and compute the AUC score.

        Parameters
        ----------
        y_test : array-like
            True binary labels.
        y_prob : array-like
            Predicted probabilities for the positive class.

        Returns
        -------
        None
        """
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        self.plt.figure(figsize=(7, 5))
        self.plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#2c3e50")
        self.plt.plot([0, 1], [0, 1], linestyle='--', color="#b60000")
        self.plt.xlabel("FPR")
        self.plt.ylabel("TPR")
        self.plt.title("ROC Curve")
        self.plt.legend()
        self.plt.show()

    def confusion_matrix(self, y_test, y_pred):
        """
        Plot a confusion matrix with annotated labels (TN, FP, FN, TP).

        Parameters
        ----------
        y_test : array-like
            True labels.
        y_pred : array-like
            Predicted labels.

        Returns
        -------
        None
        """
        cm = confusion_matrix(y_test, y_pred)

        labels = np.array([
            ["TN", "FP"],
            ["FN", "TP"]
        ])

        annotated = np.empty_like(cm).astype(str)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotated[i, j] = f"{labels[i, j]}\n{cm[i, j]}"

        cmap = LinearSegmentedColormap.from_list(
            "custom",
            ["#dbdbdb", "#2c3e50", "#e74c3c"]
        )

        self.plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=annotated,
            fmt="",
            cmap=cmap,
            cbar=False
        )

        self.plt.title("Confusion Matrix", fontsize=14)
        self.plt.xlabel("Predicted")
        self.plt.ylabel("Actual")
        self.plt.tight_layout()
        self.plt.show()

    def compare_accuracy(self, model_names, y_tests, y_preds):
        """
        Compare accuracy scores across multiple models and visualize them.

        Parameters
        ----------
        model_names : list of str
            Names of the models.
        y_tests : list of array-like
            List of true labels for each model.
        y_preds : list of array-like
            List of predicted labels for each model.

        Returns
        -------
        None
        """
        accuracies = []

        for y_test, y_pred in zip(y_tests, y_preds):
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        self.plt.figure(figsize=(7, 5))
        self.plt.bar(model_names, accuracies)
        self.plt.title("Model Accuracy Comparison")
        self.plt.ylabel("Accuracy")
        self.plt.ylim(0, 1)
        self.plt.show()

        for name, acc in zip(model_names, accuracies):
            print(f"{name}: {acc:.4f}")
