import pandas as pd
import os
from src.settings import BASE_DIR
import json
# print(BASE_DIR)
def main(only_labels=True, output_dir = 'Results', game='Lakeland'):

    game = game.upper().capitalize()
    best_path = os.path.join(BASE_DIR, f'Results/{game}/best.json')


    with open(best_path) as f:
        cluster_paths = json.load(f)
    df = None
    for k, v in cluster_paths.items():
        path = os.path.join(BASE_DIR, v['path'])
        label_name = f"{k} ({v['name']})"
        index_col = ['sessID', 'num_play'] if game.upper() == 'LAEKLAND' else ['sessionID']
        tdf = pd.read_csv(path, index_col=index_col, comment='#')
        if only_labels:
            tdf = tdf[['label']]
        tdf = tdf.rename({'label': label_name}, axis=1)
        if df is None:
            df = tdf
        else:
            df = df.join(tdf, how='inner', sort=True)
            print(f'merged {label_name}!')

    df.to_csv(os.path.join(BASE_DIR, output_dir, 'merged_clusters.csv'))

# dfs = [pd.read_csv(p, index_col=['sessID', 'num_play'], comment='#') for p in cluster_paths]
# full_df = dfs[0]
# for df in dfs[1:]:
#
#
# for df in dfs:


if __name__ == '__main__':
   main(game='Crystal')

    # dfs = [pd.read_csv(p, index_col=['sessID', 'num_play'], comment='#') for p in cluster_paths]
    # full_df = dfs[0]
    # for df in dfs[1:]:
    #
    #
    # for df in dfs:




