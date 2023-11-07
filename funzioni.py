import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# --- Create List of Color Palletes ---
black_grad = ['#100C07', '#3E3B39', '#6D6A6A', '#9B9A9C', '#CAC9CD']
my_color = ['#ff1104','#6f004c']
categoric_color=['#ff1104','#bcdaab','#ffe0b5','#ff8b37','#6f004c']
two_color=['#ff1104', '#0078F9']

def rocCurve (model, modelName, y_test, X_test ):
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#C97575", lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(modelName + ' ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    

def confusionMatrix (y_test, y_pred, modelName):
    cf_matrix = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(8,6))
    plt.title(modelName + " Confusion Matrix")
    sns.heatmap(cf_matrix,annot=True,fmt = 'd')
    plt.show()
    


def metricsVisualization (data, metric):
    
    fig = px.bar(data, x='Test ' + metric, y='Classifier', orientation='h', color='Test ' + metric,
             title='Test ' + metric + ' Scores by Classifiers', text='Test ' + metric, color_continuous_scale='inferno')

    fig.update_layout(
    xaxis_title='Test ' + metric,
    yaxis_title='Classifier',
    xaxis=dict(range=[0, 1]),
    yaxis=dict(categoryorder='total ascending'),
    showlegend=False,
    height=600,
    width=900
    )
    fig.show()
    

def metricsVisualizationHyperparameters (data, metric):
    
    fig = px.bar(data, x='Test ' + metric, y='Classifier', orientation='h', color='Test ' + metric,
             title='Test ' + metric + ' Scores by Classifiers Hyperparameters', text='Test ' + metric, color_continuous_scale='plasma')

    fig.update_layout(
    xaxis_title='Test ' + metric,
    yaxis_title='Classifier',
    xaxis=dict(range=[0, 1]),
    yaxis=dict(categoryorder='total ascending'),
    showlegend=False,
    height=600,
    width=900
    )
    fig.show()

    

def hypeVsNoHypeVisualization(modelList, metricList, metricListHype, metric):

    x = np.arange(len(modelList))

    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, metricList, width, label='Model ' + metric, color = my_color[0])

    rects2 = ax.bar(x + width/2, metricListHype, width, label='Model ' + metric +' with Hyperparameters', color = my_color[1])

    ax.set_xlabel('Classifiers')
    ax.set_ylabel(metric)
    ax.set_ylim ([0,1])
    ax.set_title('Comparison of ' + metric +' and ' + metric +' with Hyperparameters')
    ax.set_xticks(x)
    ax.set_xticklabels(modelList)
    ax.legend()
    
    for rect1, rect2 in zip(rects1, rects2):
        height1 = rect1.get_height()
        height2 = rect2.get_height()
        ax.annotate(f'{height1:.5f}', xy=(rect1.get_x() + rect1.get_width() / 2, 0), xytext=(0, 3),
            textcoords="offset points", ha='center', va='bottom')
        ax.annotate(f'{height2:.5f}', xy=(rect2.get_x() + rect2.get_width() / 2, 0), xytext=(0, 3),
            textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    

def categoricalDataVisualization(dataset,column,nameFeature,numberLabels):
        
    # --- Setting Colors, Labels, Order ---
    colors=categoric_color
    data = dataset[column].value_counts()
    # --- Size for Both Figures ---
    plt.figure(figsize=(16, 8))
    plt.suptitle( nameFeature + ' Distribution', fontweight='heavy', 
                fontsize='16', fontfamily='sans-serif', color=black_grad[0])
    # --- Pie Chart ---
    plt.subplot(1, 2, 2)
    plt.title('Pie Chart', fontweight='bold', fontsize=14,
            fontfamily='sans-serif', color=black_grad[0])
    plt.pie(dataset[column].value_counts(), labels=data.index,colors=colors, pctdistance=0.7,
            autopct='%.2f%%', wedgeprops=dict(alpha=0.8, edgecolor=black_grad[1]),
            textprops={'fontsize': 9})
    centre=plt.Circle((0, 0), 0.45, fc='white', edgecolor=black_grad[1])
    plt.gcf().gca().add_artist(centre)
    # --- Histogram ---
    countplt = plt.subplot(1, 2, 1)
    plt.title('Histogram', fontweight='bold', fontsize=14, 
            fontfamily='sans-serif', color=black_grad[0])
    ax = sns.countplot(x=column, data=dataset, palette=colors,
                    edgecolor=black_grad[2], alpha=0.85)
    for rect in ax.patches:
        ax.text (rect.get_x()+rect.get_width()/2, 
                rect.get_height()+4.25,rect.get_height(), 
                horizontalalignment='center', fontsize=8, 
                bbox=dict(facecolor='none', edgecolor=black_grad[0], 
                        linewidth=0.25, boxstyle='round'))
    plt.xlabel(nameFeature, fontweight='bold', fontsize=11, fontfamily='sans-serif', 
            color=black_grad[1])
    plt.ylabel('Total', fontweight='bold', fontsize=11, fontfamily='sans-serif', 
            color=black_grad[1])
    plt.xticks(numberLabels, fontsize = 8)
    plt.grid(axis='y', alpha=0.4) 
    plt.show()

def numericDataVisualization(dataset, nameFeature, title):
    # --- Variable, Color & Plot Size ---
    var = nameFeature
    color = my_color[0]
    color_boxPlot = my_color[1]
    fig=plt.figure(figsize=(12, 12))
    # --- General Title ---
    fig.suptitle(title + ' Column Distribution', fontweight='bold', fontsize=16, 
                fontfamily='sans-serif', color=black_grad[0])
    fig.subplots_adjust(top=0.9)
    # --- Histogram ---
    ax_1=fig.add_subplot(1, 2, 2)
    plt.title('Histogram Plot', fontweight='bold', fontsize=14, 
            fontfamily='sans-serif', color=black_grad[1])
    sns.histplot(data=dataset, x=var, kde=True, color=color)
    plt.ylabel('Total', fontweight='regular', fontsize=11, 
            fontfamily='sans-serif', color=black_grad[1])
    plt.xlabel(title, fontweight='regular', fontsize=11, fontfamily='sans-serif', 
            color=black_grad[1])
    # --- Box Plot ---
    ax_2=fig.add_subplot(1, 2, 1)
    plt.title('Box Plot', fontweight='bold', fontsize=14, fontfamily='sans-serif', 
            color=black_grad[1])
    sns.boxplot(data=dataset, y=var, color=color_boxPlot, boxprops=dict(alpha=0.8), linewidth=1.5)
    plt.ylabel(title, fontweight='regular', fontsize=11, fontfamily='sans-serif', 
            color=black_grad[1])

    plt.show()

def dataExploration(dataset, feature, title, labelFeature, numberLabel):
    # --- Labels Settings ---
    labels = ['No', 'Yes']
    # --- Creating Bar Chart ---
    ax = pd.crosstab(dataset[feature], dataset['output']).plot(kind='bar', figsize=(8, 5), 
                                            color=my_color, 
                                            edgecolor=black_grad[2], alpha=0.85)
    # --- Bar Chart Settings ---
    for rect in ax.patches:
        ax.text (rect.get_x()+rect.get_width()/2, 
                rect.get_height()+1.25,rect.get_height(), 
                horizontalalignment='center', fontsize=10)
    plt.suptitle('Heart Disease Distribution based on ' + title, fontweight='heavy', 
                x=0.065, y=0.98, ha='left', fontsize='16', fontfamily='sans-serif', 
                color=black_grad[0])
    plt.tight_layout(rect=[0, 0.04, 1, 1.025])
    plt.xlabel(title, fontfamily='sans-serif', fontweight='bold', 
            color=black_grad[1])
    plt.ylabel('Total', fontfamily='sans-serif', fontweight='bold', 
            color=black_grad[1])
    plt.xticks(numberLabel, labelFeature, rotation=0)
    plt.grid(axis='y', alpha=0.4)
    plt.grid(axis='x', alpha=0)
    plt.legend(labels=labels, title='$\\bf{Output}$', fontsize='8', 
            title_fontsize='9', loc='upper left', frameon=True)
    plt.show()

def dataExplorationNumeric(dataset, feature, title):
        # -- Scatter Plot Size & Titles Settings ---
        plt.figure(figsize=(10, 8))
        plt.suptitle('Heart Disease Scatter Plot based on Age and ' + title, fontweight='heavy', 
                x=0.048, y=0.98, ha='left', fontsize='16', fontfamily='sans-serif', 
                color=black_grad[0])
        # --- Creating Scatter Plot ---
        plt.scatter(x=dataset.age[dataset.output==0], y=feature[(dataset.output==0)], c=two_color[0])
        plt.scatter(x=dataset.age[dataset.output==1], y=feature[(dataset.output==1)], c=two_color[1])
        # --- Scatter Plot Legend & Labels Settings ---
        plt.legend(['False', 'True'], title='$\\bf{Type}$', fontsize='7', 
                title_fontsize='8', loc='upper right', frameon=True)
        plt.xlabel('Age', fontweight='bold', fontsize='11',
                fontfamily='sans-serif', color=black_grad[1])
        plt.ylabel(title, fontweight='bold', fontsize='11', 
                fontfamily='sans-serif', color=black_grad[1])
        plt.ticklabel_format(style='plain', axis='both')
        plt.grid(axis='both', alpha=0.4, lw=0.5)
        plt.show()

def rocCurveComparison(model, modelHype, X_test, y_test,modelNameHype, modelName):

        #No_hype
        y_scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        roc_auc = roc_auc_score(y_test, y_scores)

        #hype
        y_scores_hype = modelHype.predict_proba(X_test)[:, 1]
        fpr_hype, tpr_hype, thresholds_hype = roc_curve(y_test, y_scores_hype)
        roc_auc_hype = roc_auc_score(y_test, y_scores_hype)

        #sns.set_style('whitegrid')
        plt.figure(figsize=(10,5))
        plt.title('ROC Curve')
        plt.plot(fpr,tpr, lw = 2, color=my_color[0], label=modelName + f'ROC Curve (AUC = {roc_auc:.2f})' )
        plt.plot(fpr_hype,tpr_hype, lw = 2, color = my_color[1], label=modelNameHype + f'ROC Curve (AUC = {roc_auc_hype:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(' ROC Curve')
        plt.legend(loc='lower right')
        plt.legend()
        plt.show()
        