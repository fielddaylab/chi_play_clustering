from src import settings
from src.cluster_workflow import Workflow
from src import cluster_utils as cu
from src.options import Options


if __name__ == '__main__':
    # import setup
    # setup.init_path()

    filter_options = Options.lakeland_feedback_lv01
    output_foler = settings.OUTPUT_DIR
    df_getter = cu.getLakelandDecJanLogDF

    w = Workflow(filter_options=filter_options)
    w.pca_dimension_count = 2
    w.clustering_count = 7
    w.RunWorkflow(get_df_func=df_getter)
