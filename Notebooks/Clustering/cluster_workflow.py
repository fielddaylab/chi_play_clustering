from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans,DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
from math import pi

def df_np_df(func):
    def convert_call_reconvert_df(self, df, *args, **kwargs):
        nparray = df.to_numpy()
        nparray, meta = func(self, nparray, *args, **kwargs)
        assert nparray.shape[1] == len(df.columns)
        return pd.DataFrame(nparray, columns=df.columns), meta
    return convert_call_reconvert_df

class Workflow:
# fields
    DEFAULT_SCALE = "robust"
  
    def __init__(self, savename = None, graph_dir = None):
        self.meta = []
        self.savename = savename
        self.graph_path = graph_dir
        
        # flags
        self.verbose: False
            
        # steps
        self.pre_histogram = True
        self.do_scaling = True
        self.do_normalization = True
        self.post_histogram = True
        self.plot_correlation = True
        self.do_PCA = True 
        self.plot_scree = True
        self.do_clustering = True
        self.plot_silhouettes = True

        # scikitlearn
        self.outlier_method = None
        self.scaling_method = 'Robust'
        self.normalization_method = 'Normalizer'
        self.pca_dimension_count = 5
        self.clustering_method = "KMeans"
        self.clustering_count = 4

        # viz
        self.color_dict = {i:v for i,v in enumerate(plt.cm.get_cmap('tab10').colors)}
        self.histogram = None # not sure what this was meant for...
        self.feature_names = None
    
    def get_filename(self):
        return None #some_string

    def Histogram(self, df: pd.DataFrame, num_bins:int = None, title:str=None, log_scale=True):
        title = title or 'Histograms'
        num_rows = len(df.index)
        num_bins = num_bins or min(25, num_rows)

        axes = df.plot(kind='hist',subplots=True, figsize=(20,5),bins=num_bins,
                title=title,layout=(1,len(df.columns)),color='k',sharex=False,
                sharey=True, logy=log_scale,bottom=1)
        # for axrow in axes:
        #     for ax in axrow:
        #         print(ax)
        #         ax.set_yscale('log')

    def Correlations(self, df, heat_range=0.3):
        seaborn.set(style="ticks")
        corr = df.corr()
        g = seaborn.heatmap(corr,  vmax=heat_range, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
        seaborn.despine()
        g.figure.set_size_inches(14,10)
        
    # @df_np_df
    def Scaled(self, df, scaling_method:str =None):
        nparray = df.to_numpy()
        scaling_method = scaling_method or self.scaling_method
        if scaling_method == "Standard":
            nparray = RobustScaler().fit_transform(nparray)
        elif scaling_method == "Robust":
            nparray = StandardScaler().fit_transform(nparray)
        return pd.DataFrame(nparray, columns=df.columns), []
        
    # @df_np_df
    def Normalized(self, df):
        nparray = df.to_numpy()
        nparray = Normalizer().fit_transform(nparray)
        return pd.DataFrame(nparray, columns=df.columns), []


    def PCA(self, df, dimension_count:int = None):
        nparray = df.to_numpy()
        dimension_count = dimension_count or self.pca_dimension_count
        nparray = PCA(n_components=dimension_count).fit_transform(nparray)
        PCA_names = [f"PCA_{i+1}" for i in range(dimension_count)]
        return pd.DataFrame(nparray, columns=PCA_names), []
      
    def Scree(self, df):
        nparray = df.to_numpy()
        U,S,V = np.linalg.svd(nparray)
        eigvals = S**2 / np.sum(S**2)
        fig = plt.figure(figsize=(8,5))
        singular_vals = np.arange(nparray.shape[1]) + 1
        plt.plot(singular_vals, eigvals, 'ro-', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        return

    def Cluster(self, df, cluster_count:int, clustering_method=None):
        cluster_count = cluster_count or self.clustering_count
        clustering_method = clustering_method or self.clustering_method
        nparray = df.to_numpy()

        if clustering_method == "KMeans":
            clusterer = KMeans(n_clusters=cluster_count)
            labels = clusterer.fit_predict(nparray)
            # For future, include calculated distances.
            # In the future, this will let us find centers:
            # distances = clusterer.transform(nparray)
            # nparray = np.concatenate((distances, labels))
            nparray = labels
        # elif clustering_method == "FuzzyCMeans":
        #     pass
        elif clustering_method == "DBSCAN":
            labels = DBSCAN(eps=0.3, min_samples=10).fit_predict(nparray)
            nparray = labels
        return pd.DataFrame(nparray, columns=["labels"]), []

        # for a,l in zip(PCA_dims, labels):
        #     b =  clustering.cluster_centers_[l]
        #     distances.append(a-b)
        # df['label'] = labels
        # df['PCA1 Offset'] = np.array(distances)[:,0]
        # df['PCA2 Offset'] = np.array(distances)[:,1]

    def Silhouettes(self, dimension_data: pd.DataFrame, labels: pd.DataFrame, title = None):
        np_dimensions = dimension_data.to_numpy()
        np_labels = labels.to_numpy().flatten()
        silhouette_vals = silhouette_samples(np_dimensions, np_labels)

        # Silhouette plot
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
        y_ticks = []
        y_lower, y_upper = 0, 0
        for i, cluster in enumerate(np.unique(np_labels)):
            cluster_silhouette_vals = silhouette_vals[np_labels == cluster]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
            ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
            y_lower += len(cluster_silhouette_vals)

        # Get the average silhouette score and plot it
        avg_score = np.mean(silhouette_vals)
        ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
        ax1.set_yticks([])
        ax1.set_xlim([-0.1, 1])
        ax1.set_xlabel('Silhouette coefficient values')
        ax1.set_ylabel('Cluster labels')
        title = title or 'Silhouette plot for the various clusters'
        ax1.set_title(title, y=1.02)

        return
      
    def SaveMeta(self, meta_list):
        print("SaveMeta: Stubbed function!")
        return None, []

    def RunWorkflow(self, df, meta):
        data = df.copy()
        new_meta = meta.copy()
        if self.pre_histogram:
            self.Histogram(data, title='Pre Normalization Histogram')
        if self.do_scaling:
            data, md = self.Scaled(data)
            new_meta.extend(md)
        if self.do_normalization:
            data, md = self.Normalized(data)
            new_meta.extend(md)

            self.Histogram(data, title='Post Normalization Histogram')
        if self.plot_correlation:
            self.Correlations(data)
        if self.do_PCA:
            data, md = self.PCA(data)
            new_meta.extend(md)
        if self.plot_scree:
            self.Scree(data)
        if self.do_clustering:
            column_names=[f"PCA{i}" for i in range(self.pca_dimension_count)]
            labels, md = self.Cluster(data[column_names], cluster_count=self.clustering_count)
            new_meta.extend(md)
        if self.plot_silhouettes:
            self.Silhouettes(data, labels)
        return data, new_meta