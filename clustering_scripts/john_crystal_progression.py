from src import settings
from src.cluster_workflow import Workflow
from src import cluster_utils as cu
from src.options import Options


def main():


    filter_options = Options.crystal_progression
    output_foler = settings.OUTPUT_DIR
    df_getter = cu.getCrystalDecJanLogDF

    w = Workflow(filter_options=filter_options, nested_folder_output=False)
    w.pca_dimension_count = 2
    w.clustering_counts = [4]
    w.RunWorkflow(get_df_func=df_getter)

if __name__ == '__main__':
# import setup
    # setup.init_path()

    main()
