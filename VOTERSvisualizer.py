import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def importance_counter(sample):
    unimpts = []   #3.0
    notverys = []  #2.0
    somewhats = [] #1.0
    veryimpts = [] #0.0
    for feature in sample:
        counts = sample[feature].value_counts()
        if 0.0 in list(counts.index):
            unimpts.append(counts[0.0])
        else:
            unimpts.append(0)
        if 1.0 in list(counts.index):
            notverys.append(counts[1.0])
        else:
            notverys.append(0)
        if 2.0 in list(counts.index):
            somewhats.append(counts[2.0])
        else:
            somewhats.append(0)
        if 3.0 in list(counts.index):
            veryimpts.append(counts[3.0])
        else:
            veryimpts.append(0)

    return unimpts, notverys, somewhats, veryimpts


def importance_relativizer(unimpts, notverys, somewhats, veryimpts, sample):
    total = len(sample)
    prprtn_unimpts = np.true_divide(unimpts, total) * 100 / 33.33
    prprtn_notverys = np.true_divide(notverys, total) * 100 / 33.33
    prprtn_somewhats = np.true_divide(somewhats, total) * 100 / 33.33
    prprtn_veryimpts = np.true_divide(veryimpts, total) * 100 / 33.33

    return prprtn_unimpts, prprtn_notverys, prprtn_somewhats, prprtn_veryimpts


def base_visualization(unimpts, notverys, somewhats, veryimpts, voters_sample):
    prprtn_unimpts, prprtn_notverys, prprtn_somewhats, prprtn_veryimpts = importance_relativizer(unimpts, notverys, somewhats, veryimpts, voters_sample)
    plt.figure(figsize=(8,10))

    plt.barh(voters_sample.columns, prprtn_unimpts, label='Unimportant', color='#EEEEEE')
    plt.barh(voters_sample.columns, prprtn_notverys, label='Not Very Important', color='#CCCCCC', left=prprtn_unimpts)
    plt.barh(voters_sample.columns, prprtn_somewhats, label='Somewhat Important', color='#AAAAAA', left=prprtn_unimpts+prprtn_notverys)
    plt.barh(voters_sample.columns, prprtn_veryimpts, label='Very Important', color='#999999', left=prprtn_unimpts+prprtn_notverys+prprtn_somewhats)

    ##source: http://benalexkeen.com/bar-charts-in-matplotlib/
    return plt.gca()

def cluster_visualization(cluster, proportional_data, voters_data, cluster_no="Unknown"):

    scale_factor = (3 / (proportional_data.mean().mean() * 2))

    total = len(voters_data)
    unimpts, notverys, somewhats, veryimpts = importance_counter(voters_data)
    base_visualization(unimpts, notverys, somewhats, veryimpts, voters_data)

    g = sns.scatterplot(y = cluster.columns, x = (cluster.mean()*scale_factor), zorder=10, palette="cubehelix", hue=cluster.columns, s=100, legend=False)
    g.set_xlim(0,3)

    for column_no in range(len(cluster.columns)):
        if (proportional_data.mean()[column_no]) > (cluster.mean()[column_no]):
            change_line_color = 'red'
        else:
            change_line_color = 'blue'
        plt.plot((proportional_data.mean()[column_no]*scale_factor,cluster.mean()[column_no]*scale_factor),(column_no,column_no), color=change_line_color, label=None)

    plt.xticks(ticks=[0.0, 1.5, 3.0], labels = ["Relatively\nUnimportant", "Average\nImportance", "Relatively\nImportant"])
    plt.grid(axis='x', c="#666666")
    plt.legend(loc=(1.04,0))
    plt.title("Cluster #{}, n={}".format(cluster_no, total))
    return plt.gca()


def show_all_clusters(clusters_proportional, total_proportional, clusters_absolute):
    for cluster in range(len(clusters_proportional)):
        cluster_visualization(clusters_proportional[cluster], total_proportional, clusters_absolute[cluster], str(cluster+1))
    return None
