from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, Normalizer, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
from math import pi
import src.utils as utils
import src.cluster_utils as cu
import src.utils as utils
import os


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

    def __init__(self, filter_options, base_output_dir=None, nested_folder_output=True):
        self.plot_cluster_scatter = True
        self.plot_radars = True
        utils.init_path()
        self.filter_options = filter_options
        self._nested_folder_output = nested_folder_output
        self._base_output_dir = base_output_dir

        # flags
        self.verbose: False

        # steps
        self.pre_histogram = True
        self.do_logtransform = True
        self.do_scaling = True
        self.do_normalization = False
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
        self.color_dict = {i: v for i, v in enumerate(plt.cm.get_cmap('tab10').colors)}
        self.histogram = None  # not sure what this was meant for...
        self.feature_names = None


    def clustering_abbrev(self):
        cluster_abbrev = 'k' if self.clustering_method is "KMeans" else self.clustering_method
        return f'_z{self.filter_options.zthresh}pca{self.pca_dimension_count}{cluster_abbrev}{self.clustering_count}'

    def get_base_output_dir(self):
        if self._base_output_dir:
            save_dir = self._base_output_dir
        else:
            logtransform = '_logtransform' if self.do_logtransform else ''
            clustering_suffix = '' if self._nested_folder_output else self.clustering_abbrev()
            suffix = f'{logtransform}{clustering_suffix}'
            save_dir = os.path.join('Results',self.filter_options.game.lower().capitalize(), self.filter_options.name+suffix)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        return save_dir

    def get_cluster_output_dir(self):
        if not self._nested_folder_output:
            return self.get_base_output_dir()
        else:
            return os.path.join(self.get_base_output_dir(), self.clustering_abbrev())

    def get_filename(self):
        return None  # some_string

    def Histogram(self, df: pd.DataFrame, num_bins: int = None, title: str = None, log_scale=True, save=True):
        title = title or 'Histograms'
        num_rows = len(df.index)
        num_bins = num_bins or min(25, num_rows)

        axes = df.plot(kind='hist', subplots=True, figsize=(20, 5), bins=num_bins,
                       title=title, layout=(1, len(df.columns)), color='k', sharex=False,
                       sharey=True, logy=log_scale, bottom=1)
        # for axrow in axes:
        #     for ax in axrow:
        #         print(ax)
        #         ax.set_yscale('log')
        if save:
            savepath = os.path.join(self.get_base_output_dir(), f'{title}.png')
            plt.savefig(savepath)

    # TODO: Graph is part cut off, i think there might be some stuff hardcoded.
    def Correlations(self, df, heat_range=0.3, save=True):
        plt.figure()
        seaborn.set(style="ticks")
        corr = df.corr()
        g = seaborn.heatmap(corr, vmax=heat_range, center=0,
                            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
        seaborn.despine()
        g.figure.set_size_inches(14, 10)

        title = 'Correlations'
        if save:
            savepath = os.path.join(self.get_base_output_dir(), f'{title}.png')
            g.figure.savefig(savepath)

    def LogTransformed(self, df):
        nparray = df.to_numpy()
        nparray = np.log1p(nparray)
        return pd.DataFrame(nparray, columns=df.columns), []
        pass

    # @df_np_df
    def Scaled(self, df, scaling_method: str = None):
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

    def PCA(self, df, dimension_count: int = None):
        nparray = df.to_numpy()
        dimension_count = dimension_count or self.pca_dimension_count
        nparray = PCA(n_components=dimension_count).fit_transform(nparray)
        PCA_names = [f"PCA_{i}" for i in range(dimension_count)]
        return pd.DataFrame(nparray, columns=PCA_names), []

    def Scree(self, df, save=True):
        nparray = df.to_numpy()
        U, S, V = np.linalg.svd(nparray)
        eigvals = S ** 2 / np.sum(S ** 2)
        fig = plt.figure(figsize=(8, 5))
        singular_vals = np.arange(nparray.shape[1]) + 1
        plt.plot(singular_vals, eigvals, 'ro-', linewidth=2)
        title = 'Scree Plot'
        plt.title(title)
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')

        if save:
            savepath = os.path.join(self.get_base_output_dir(), f'{title}.png')
            plt.savefig(savepath)
        return

    def Cluster(self, df, cluster_count: int, clustering_method=None):
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
        # elif clustering_method == "FuzzyCMeans":
        #     pass
        elif clustering_method == "DBSCAN":
            labels = DBSCAN(eps=0.3, min_samples=10).fit_predict(nparray)
        else:
            labels = []
        return labels,[]

        # for a,l in zip(PCA_dims, labels):
        #     b =  clustering.cluster_centers_[l]
        #     distances.append(a-b)
        # labels = labels
        # df['PCA1 Offset'] = np.array(distances)[:,0]
        # df['PCA2 Offset'] = np.array(distances)[:,1]

    def Silhouettes(self, dimension_data: pd.DataFrame, labels: pd.DataFrame, title=None, save=True):
        np_dimensions = dimension_data.to_numpy()
        silhouette_vals = silhouette_samples(np_dimensions, labels)

        # Silhouette plot
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
        y_ticks = []
        y_lower, y_upper = 0, 0
        for i, cluster in enumerate(np.unique(labels)):
            cluster_silhouette_vals = silhouette_vals[labels == cluster]
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
        title = title or f'Silhouette Plot Avg={int(avg_score*100)}%'
        ax1.set_title(title, y=1.02)
        if save:
            savepath = os.path.join(self.get_base_output_dir(), f'{title}.png')
            plt.savefig(savepath)

        return


    def scatter(self, df, labels, save=True):
        num_cols = len(df.columns)
        color_array = [self.color_dict[c] for c in labels]
        fig, axs = plt.subplots(num_cols, num_cols, figsize=(30, 30))
        for x in range(num_cols):
            for y in range(num_cols):
                axs[x, y].scatter(df.iloc[:, x], df.iloc[:, y], c=color_array)
                axs[x, y].set_xlabel(df.columns[x])
                axs[x, y].set_ylabel(df.columns[y])
        title = 'Scatter'
        if save:
            savepath = os.path.join(self.get_base_output_dir(), f'{title}.png')
            plt.savefig(savepath)

    def radarCharts(self, df, labels, save=True):
        clusters = set(labels)
        categories = self.filter_options.finalfeats_readable
        description_df = df.describe()
        summary_df = pd.DataFrame(columns=description_df.columns)
        clusters = set(labels)
        cluster_dict = {}
        for c in clusters:
            cluster_dict[c] = df[labels == c]
            cluster_df = cluster_dict[c].describe()
            summary_df.loc[f'C{c}_zscore', :] = (cluster_df.loc['mean', :] - description_df.loc['mean', :]) / description_df.loc[
                                                                                                       'std', :]
            summary_df.loc[f'C{c}_%mean', :] = (cluster_df.loc['mean', :] / description_df.loc['mean', :]) * 100
            summary_df.loc[f'C{c}_%std', :] = (cluster_df.loc['std', :] / description_df.loc['std', :]) * 100
        summary_df = summary_df.apply(lambda x: (x * 100) // 1 * .01)

        def make_spider(color, i):
            offset = .25 * pi
            # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
            angles = [n / float(N) * 2 * pi + offset for n in range(N)]
            angles += angles[:1]
            ax = plt.subplot(nrows, ncols, i + 1, polar=True)
            plt.xticks(angles[:-1], categories, color='grey', size=12)
            ax.set_rlabel_position(0)
            if var == 'zscore':
                plt.yticks([-2, -1, 0, 1, 2], color="grey", size=7)
                plt.ylim(-2, 2)
            elif '%' in var:
                plt.yticks(range(0, 1000, 100), color="grey", size=7)
                plt.ylim(0, 400)
            values = list(tdf.iloc[i, :-3])
            values += values[:1]
            graph_name = tdf.index[i]
            print(angles, values)
            ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
            ax.fill(angles, values, color=color, alpha=0.4)
            plt.title(graph_name + f' (n={len(cluster_dict[i])})', size=11, color=color, y=1.1)

        # number of variable
        for var in ['zscore', '%mean', '%std']:
            tdf = summary_df.loc[[idx for idx in summary_df.index if var in idx], :]
            if not categories:
                categories = list(tdf.columns)
            N = len(categories)
            num_groups = len(tdf.index)
            #   nrows = 2
            #   ncols = num_groups//2 if not num_groups%2 else num_groups//2 + 1
            nrows = 1
            ncols = num_groups
            fig = plt.figure(figsize=(20, 5))
            fig.suptitle(f'{var} Radar Charts')
            for i in range(num_groups):
                make_spider(self.color_dict[i], i)
            fig.subplots_adjust(wspace=0.4)
            if save:
                plt.savefig(f'{self.get_cluster_output_dir()}/radar_{var}.png')
        pass

    def SaveMeta(self, meta_list):
        print("SaveMeta: Stubbed function!")
        return None, []

    def RunWorkflow(self, get_df_func):
        original_df, meta = cu.full_filter(get_df_func, self.filter_options)
        df = original_df.copy()
        original_cols = list(df.columns)
        # Preprocessing - LogTransform, Scaling, Normalization #
        # show df before any processing

        if self.pre_histogram:
            self.Histogram(df, title='Raw Histogram')
        # do log transform
        if self.do_logtransform:
            df, md = self.LogTransformed(df)
            meta.extend(md)
        # scale df
        if self.do_scaling:
            df, md = self.Scaled(df)
            meta.extend(md)
        # do normalization
        if self.do_normalization:
            df, md = self.Normalized(df)
            meta.extend(md)
        # show df after transformation
        if self.post_histogram:
            self.Histogram(df, title='Preprocessed Histogram')

        # correlation
        if self.plot_correlation:
            self.Correlations(df)

        # scree and PCA
        if self.plot_scree:
            self.Scree(df)
        if self.do_PCA:
            pca_df, md = self.PCA(df)
            meta.extend(md)
            cluster_df = pca_df
        else:
            cluster_df = df

        # silhouette and clustering
        if self.do_clustering:
            labels, md = self.Cluster(cluster_df, cluster_count=self.clustering_count)
            meta.extend(md)

            if self.plot_silhouettes:
                self.Silhouettes(cluster_df, labels)
            if self.plot_cluster_scatter:
                self.scatter(df, labels)

            if self.plot_radars:
                self.radarCharts(original_df, labels)

        return df, meta




def add_cluster_features_to_df(pipeline, df, data):
    pipeline.fit(data)
    PCA_dims = pipeline[:-1].transform(data)
    clustering = pipeline[-1]
    labels = clustering.predict(PCA_dims)
    distances = []
    for a, l in zip(PCA_dims, labels):
        b = clustering.cluster_centers_[l]
        distances.append(a - b)
    labels = labels
    df['PCA1 Offset'] = np.array(distances)[:, 0]
    df['PCA2 Offset'] = np.array(distances)[:, 1]


def main():
    utils.init_path()
    # test_on_1125_data()
    w = Workflow(cu.options.lakeland_achs_achs_per_sess_second_sessDur, r'G:\My Drive\Field Day\Research and Writing Projects\2020 CHI Play - Lakeland Clustering\Jupyter\Results\Lakeland\test')
    w.plot_silhouettes, w.plot_cluster_scatter = False,False
    w.RunWorkflow(cu.getDecJanLogDF)


def test_on_1125_data():

    base_dir = r'Results/Lakeland/1125_Clusters'
    for options, pca_dims, k in [(cu.options.lakeland_feedback_lv01, 3, 7),
                                 (cu.options.lakeland_player_lvl01, 3, 6),
                                 (cu.options.lakeland_achs_achs_per_sess_second_sessDur, 3, 6)]:
        df, basemeta = cu.full_filter(cu.getLakelandNov25ClassDF, options)
        save_dir = f'{base_dir}/{options.name}_logtransform_z{options.zthresh}pca{pca_dims}k{k}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        pipeline = make_pipeline(FunctionTransformer(np.log1p, validate=True), RobustScaler(), PCA(pca_dims),
                                 KMeans(k))
        meta = basemeta + [f'Pipeline: {pipeline}']
        with open(f'{save_dir}/meta.txt', 'w+') as f:
            print(*meta, sep='\n', file=f)
        df = df.copy()
        add_cluster_features_to_df(pipeline, df, df.to_numpy())
        df.to_csv(f'{save_dir}/clusters.csv')
        print(f'Saved: {save_dir}/clusters.csv')
    pass


if __name__ == '__main__':
    main()
