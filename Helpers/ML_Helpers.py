import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, plot_roc_curve, recall_score, precision_score

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    rec = round(recall_score(y, y_pred), 2)
    prec = round(precision_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title(f"Accuracy Score: {acc},Recall Score: {rec}, Precision Score: {prec} ",
              size=10)
    plt.show()
def plot_ROC_curve(model,X_test,y_test):
    plot_roc_curve(model, X_test, y_test)
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.show()


