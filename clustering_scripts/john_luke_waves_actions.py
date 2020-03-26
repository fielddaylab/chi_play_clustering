from src import settings
from src.cluster_workflow import Workflow
from src import cluster_utils as cu
from src.options import Options



if __name__ == '__main__':
    # import setup
    # setup.init_path()

    filter_options = Options.waves_actions_lv016
    output_foler = settings.OUTPUT_DIR
    df_getter = cu.getWavesDecJanLogDF

    w = Workflow(filter_options=filter_options)
    w.pca_dimension_count = 2
    for k in range(4,8):
        w.clustering_count = k
        w.RunWorkflow(get_df_func=df_getter)
