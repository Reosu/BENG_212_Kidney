import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import model_selection, linear_model, metrics, tree, cluster
import random

def plot_gene_weights(data,component,cutoffs):
    fig,ax = plt.subplots()
    for i in range(len(data[component].index)):
        if data[component].iloc[i] > cutoffs[component]:
            ax.plot(i,data[component].iloc[i],"o",color="tab:blue")
        else:
            ax.plot(i,data[component].iloc[i],"o",color="tab:grey")
    ax.set_xlim(-10,len(data[component].index))
    ax.plot([-10,len(data[component].index)],[cutoffs[component],cutoffs[component]],"--",color="tab:grey")
    ax.plot([-10,len(data[component].index)],[-cutoffs[component],-cutoffs[component]],"--",color="tab:grey")
    
    return ax

def iModulon_genes(data,component,cutoff):
    iMod = []
    for i in range(len(data[component].index)):
        if data[component].iloc[i] > cutoff[component]:
            iMod.append(data.index[i])
    return iMod

def plot_gene_activities(data,component,comp_num = None):
    fig,ax = plt.subplots(figsize=[40,5])
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple',
              'tab:brown','tab:pink','tab:gray','tab:olive', 'tab:cyan',
              'black', 'salmon', 'chocolate', 'orange', 'gold', 'lawngreen',
              'turquoise', 'steelblue', 'navy', 'violet', 'deeppink',
              'firebrick', 'sandybrown','olivedrab','darkgreen', 'aqua',
              'slategray', 'blue', 'pink']
    current_label = ""
    previous_label = ""
    x_labels = []
    color_count = 0
    for i in data.index:
        previous_label = current_label
        current_label = i.split("_")[0]
        if comp_num is None:
            ax.bar(i,data[component].loc[i],color=colors[color_count]
                   ,width=1)
            if previous_label != current_label:
                color_count+=1
                if color_count > len(colors)-2:
                    color_count=0
                x_labels.append(current_label)
            else:
                x_labels.append("")
        elif current_label in comp_num and comp_num is not None:
            ax.bar(i,data[component].loc[i],color="tab:blue",width=1)
            x_labels.append(i.split("_")[1])
    plt.xticks(range(len(x_labels)),x_labels,rotation = 45, ha = "right")
    
    return ax
        