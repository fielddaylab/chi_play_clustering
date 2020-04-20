from src import settings
from src.cluster_workflow import Workflow
from src import cluster_utils as cu
from src.options import Options


if __name__ == '__main__':
    # import setup
    # setup.init_path()

    filter_options = Options.lakeland_feedback_lv01_with_bloom
    output_foler = settings.OUTPUT_DIR
    df_getter = cu.getLakelandDecJanLogDF

    w = Workflow(filter_options=filter_options, nested_folder_output=False)
    w.clustering_method = "KMeans"
    w.pca_dimension_count = 2
    w.eps_min_list = [(eps, min_samples) for eps in [.01, .02, .05, .07, .1, .2, .3] for min_samples in [5]]
    w.min_cluster_size_list = [15,30,60,100]
    w.plot_silhouettes = True
    w.plot_radars = True
    w.clustering_counts = [7]
    w.RunWorkflow(get_df_func=df_getter)
