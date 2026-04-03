import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix



class Visualization:
    
    def __init__(self, df):        
        self.df = df
        self.sns = sns
        self.plt = plt


    # Scatter plot (interacción entre variables)
    def scatter(self, x, y, hue="default", title=None):
        self.plt.figure(figsize=(8,5))
        
        self.sns.scatterplot(
            data=self.df,
            x=x,
            y=y,
            hue=hue,
            palette=["#2c3e50", "#e74c3c"],  # 🔥 colores
            marker="^",                      # 🔺 triángulos
            s=70,
            alpha=0.8
        )
        
        self.plt.title(title if title else f"{x} vs {y}", fontsize=14)
        self.plt.tight_layout()
        self.plt.show()
        
    # Histogram
    def hist(self, column):
        self.plt.figure(figsize=(7,5))
        self.sns.histplot(self.df[column], bins=30)
        self.plt.title(f"Distribution of {column}")
        self.plt.show()

    # Boxplot
    def box(self, x, y):
        self.plt.figure(figsize=(7,5))
        self.sns.boxplot(x=x, y=y, data=self.df)
        self.plt.title(f"{y} by {x}")
        self.plt.show()

    # ROC Curve
    def roc_curve(self, y_test, y_prob):

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        self.plt.figure(figsize=(7,5))
        self.plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color ="#2c3e50")
        self.plt.plot([0,1], [0,1], linestyle='--', color= "#b60000")
        self.plt.xlabel("FPR")
        self.plt.ylabel("TPR")
        self.plt.title("ROC Curve")
        self.plt.legend()
        self.plt.show()

    # Confusion Matrix
    def confusion_matrix(self, y_test, y_pred):

        cm = confusion_matrix(y_test, y_pred)

        labels = np.array([
            ["TN", "FP"],
            ["FN", "TP"]
        ])

        annotated = np.empty_like(cm).astype(str)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotated[i, j] = f"{labels[i, j]}\n{cm[i, j]}"

        # 🔥 tu paleta personalizada
        cmap = LinearSegmentedColormap.from_list(
            "custom",
            ["#dbdbdb", "#2c3e50", "#e74c3c"]  
        )

        self.plt.figure(figsize=(6,5))
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
        
        accuracies = []
        
        for y_test, y_pred in zip(y_tests, y_preds):
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        self.plt.figure(figsize=(7,5))
        self.plt.bar(model_names, accuracies)
        self.plt.title("Model Accuracy Comparison")
        self.plt.ylabel("Accuracy")
        self.plt.ylim(0,1)
        self.plt.show()

        # print resultados
        for name, acc in zip(model_names, accuracies):
            print(f"{name}: {acc:.4f}")