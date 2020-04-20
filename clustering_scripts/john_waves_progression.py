from src import settings
from src.cluster_workflow import Workflow
from src import cluster_utils as cu
from src.options import Options



if __name__ == '__main__':
    # import setup
    # setup.init_path()

    filter_options = Options.waves_progression
    output_foler = settings.OUTPUT_DIR
    df_getter = cu.getWavesDecJanLogDF

    w = Workflow(filter_options=filter_options)
    w.further_filter_query_list = [f'sum_lvl_0_to_34_totalLevelTime < {50*60}']  # 50 mins
    w.clustering_counts = range(3,8)
    w.verbose = True
    w.RunWorkflow(get_df_func=df_getter)
