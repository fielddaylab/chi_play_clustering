"""
Note: Utils in this file are 

Usage:
If the first cell ran correctly, changing the CWD to the Jupyter file and adding '.' to sys path, then.
import Notebooks.Clustering.cluster_utils as cu

Otherwise:
from google.colab import drive
drive.mount('/content/drive')

import os
FIELDDAY_DIR = '/content/drive/My Drive/Field Day' # the field day directory on the mounted drive
JUPYTER_DIR = os.path.join(FIELDDAY_DIR,'Research and Writing Projects/2020 Lakeland EDM/Jupyter')
os.chdir(JUPYTER_DIR)

import sys
sys.path.append('.')
import Notebooks.Clustering.cluster_utils as cu

"""

from zipfile import ZipFile
import pandas as pd
import urllib.request
from io import BytesIO
import src.utils as utils
import ipywidgets as widgets
from collections import namedtuple
import numpy as np
from scipy import stats
from typing import Optional, List, Iterable





def print_options(meta):
    """
    Takes in meta text and outputs text for an options group.
    :param meta: meta text. Expected format will be like:

    Metadata:
Import from fhttps://opengamedata.fielddaylab.wisc.edu/data/LAKELAND/LAKELAND_20191201_to_20191231_de09c18_proc.zip
Import from fData/Raw Log Data/LAKELAND_20200101_to_20200131_a9720c1_proc.zip
*arg* filter_args = {'query_list': ['debug == 0', 'sess_ActiveEventCount >= 10', 'sessDuration >= 300', '_continue == 0'], 'one_query': False, 'fillna': 0, 'verbose': True}
Query: Intial Shape, output_shape: (32227, 1647)
Query: debug == 0, output_shape: (32221, 1647)
Query: sess_ActiveEventCount >= 10, output_shape: (26934, 1647)
Query: sessDuration >= 300, output_shape: (16109, 1647)
Query: _continue == 0, output_shape: (10591, 1647)
Filled NaN with 0
*arg* new_feat_args = {'verbose': False, 'avg_tile_hover_lvl_range': None}
*arg* lvlfeats = ['count_blooms', 'count_deaths', 'count_farmfails', 'count_food_produced', 'count_milk_produced']
*arg* lvlrange = range(0, 2)
Describe Level Feats lvls 0 to 1. Assuming WINDOW_SIZE_SECONDS=300 and WINDOW_OVERLAP_SECONDS=30, filtered by (sessDuration > 570)
*arg* finalfeats = ['avg_lvl_0_to_1_count_deaths', 'avg_lvl_0_to_1_count_farmfails', 'avg_lvl_0_to_1_count_food_produced', 'avg_lvl_0_to_1_count_milk_produced']
Original Num Rows: 6712
*arg* zthresh = 3
Removed points with abs(ZScore) >= 3. Reduced num rows: 6497

    where all args are denoted by an initial *arg* and values are after =.


    """
    if type(meta) == str:
        meta = meta.split('\n')
    inner = ',\n\t'.join(["'GAME'", "'NAME'"] + [l[6:].split(' = ')[1] for l in meta if l.startswith('*arg*')] + ['[]'])
    print(f'options({inner}\n)')


def openZipFromURL(url):
    """

    :param url: url pointing to a zipfile
    :return: zipfile object, list of metadata lines
    """
    metadata = [f'Import from f{url}']
    resp = urllib.request.urlopen(url)
    zipfile = ZipFile(BytesIO(resp.read()))

    return zipfile, metadata


def openZipFromPath(path):
    """

    :param path: path pointing to a zipfile
    :return: zipfile object, list of metadata lines
    """
    metadata = [f'Import from f{path}']
    zipfile = ZipFile(path)

    return zipfile, metadata


def readCSVFromPath(path):
    """

    :param path: path pointing to a csv
    :return: dataframe, List[str] of metadata lines
    """
    import os
    print(os.getcwd())
    metadata = [f'Import from f{path}']
    df = pd.read_csv(path, index_col=['player_id', 'sessID', 'num_play'], comment='#')
    return df, metadata


def getLakelandNov25ClassDF():
    """

    :return: (df, metadata List[str])
    """
    path = 'Data/Filtered Log Data/Only school day 11-25 pid sessions.csv'
    return readCSVFromPath(path)


# consider making a general version with parameter for filename, index columns
def getLakelandDecJanLogDF():
    """

    :return: (df, metadata List[str])
    """
    # define paths for DecJanLog
    _proc_zip_url_dec = 'https://opengamedata.fielddaylab.wisc.edu/data/LAKELAND/LAKELAND_20191201_to_20191231_de09c18_proc.zip'
    _proc_zip_path_jan = 'Data/Raw Log Data/LAKELAND_20200101_to_20200131_a9720c1_proc.zip'
    # get the data
    metadata = []
    zipfile_dec, meta = openZipFromURL(_proc_zip_url_dec)
    metadata.extend(meta)
    zipfile_jan, meta = openZipFromPath(_proc_zip_path_jan)
    metadata.extend(meta)
    # put the data into a dataframe
    df = pd.DataFrame()
    for zf in [zipfile_dec, zipfile_jan]:
        with zf.open(zf.namelist()[0]) as f:
            df = pd.concat([df, pd.read_csv(f, index_col=['sessID', 'num_play'], comment='#')], sort=True)
    df['sessID'] = [x[0] for x in df.index]
    df['num_play'] = [x[1] for x in df.index]
    return df, metadata


def getWavesDecJanLogDF():
    """

    :return: (df, metadata List[str])
    """
    # define paths for DecJanLog
    _proc_zip_path_dec = 'https://opengamedata.fielddaylab.wisc.edu/data/WAVES/WAVES_20191201_to_20191231_de09c18_proc.zip'
    _proc_zip_path_jan = 'https://opengamedata.fielddaylab.wisc.edu/data/WAVES/WAVES_20200101_to_20200131_de09c18_proc.zip'
    # get the data
    metadata = []
    zipfile_dec, meta = openZipFromURL(_proc_zip_path_dec)
    metadata.extend(meta)
    zipfile_jan, meta = openZipFromURL(_proc_zip_path_jan)
    metadata.extend(meta)
    # put the data into a dataframe
    df = pd.DataFrame()
    for zf in [zipfile_dec, zipfile_jan]:
        with zf.open(zf.namelist()[0]) as f:
            df = pd.concat([df, pd.read_csv(f, index_col=['sessionID'], comment='#')], sort=True)
    df['sessionID'] = [x for x in df.index]
    return df, metadata


def getCrystalDecJanLogDF():
    """

    :return: (df, metadata List[str])
    """
    # define paths for DecJanLog
    _proc_zip_path_dec = 'https://opengamedata.fielddaylab.wisc.edu/data/CRYSTAL/CRYSTAL_20191201_to_20191231_de09c18_proc.zip'
    _proc_zip_path_jan = 'https://opengamedata.fielddaylab.wisc.edu/data/CRYSTAL/CRYSTAL_20200101_to_20200131_de09c18_proc.zip'
    # get the data
    metadata = []
    zipfile_dec, meta = openZipFromURL(_proc_zip_path_dec)
    metadata.extend(meta)
    zipfile_jan, meta = openZipFromURL(_proc_zip_path_jan)
    metadata.extend(meta)
    # put the data into a dataframe
    df = pd.DataFrame()
    for zf in [zipfile_dec, zipfile_jan]:
        with zf.open(zf.namelist()[0]) as f:
            df = pd.concat([df, pd.read_csv(f, index_col=['sessionID'], comment='#')], sort=True)
    df['sessionID'] = [x for x in df.index]
    return df, metadata


def get_lakeland_default_filter(lvlstart: Optional[int]=None, lvlend: Optional[bool]=None, no_debug: Optional[bool]=True,
              min_sessActiveEventCount: Optional[int]=10,
              min_lvlstart_ActiveEventCount: Optional[int]=3,
              min_lvlend_ActiveEventCount: Optional[int]=3, min_sessDuration: Optional[int]=300, max_sessDuration: Optional[int]=None, cont: Optional[bool]=False) -> List[str]:
    """

    :param lvlstart: levelstart to be used for other parameters (None if not used)
    :param lvlend: levelend to be used for other parameters (None if not used)
    :param no_debug: boolean whether or not to use only players that have used SPYPARTY or only not used SPYPARTY  (None if not used)
    :param min_sessActiveEventCount:  (None if not used)
    :param min_lvlstart_ActiveEventCount:  (None if not used)
    :param min_lvlend_ActiveEventCount:  (None if not used)
    :param min_sessDuration:  (None if not used)
    :param max_sessDuration:  (None if not used)
    :param cont:  (None if not used)
    :return:
    """
    get_lakeland_default_filter()
    query_list = []


    if no_debug:
        query_list.append('debug == 0')
    if min_sessActiveEventCount is not None:
        query_list.append(f'sess_ActiveEventCount >= {min_sessActiveEventCount}')
    if lvlstart is not None and min_lvlstart_ActiveEventCount is not None:
        query_list.append(f'lvl{lvlstart}_ActiveEventCount >= {min_lvlstart_ActiveEventCount}')
    if lvlend is not None and min_lvlend_ActiveEventCount is not None:
        query_list.append(f'lvl{lvlend}_ActiveEventCount >= {min_lvlend_ActiveEventCount}')
    if min_sessDuration is not None:
        query_list.append(f'sessDuration >= {min_sessDuration}')
    if max_sessDuration is not None:
        query_list.append(f'sessDuration <= {max_sessDuration}')
    if cont is not None:
        query_list.append(f'_continue == {int(cont)}')

    return query_list

def get_waves_default_filter(through_QA1=False, through_QA3=False):
    query_list = []
    if through_QA1:
        query_list.append('QA1_questionCorrect==QA1_questionCorrect')
    if through_QA3:
        query_list.append('QA3_questionCorrect==QA3_questionCorrect')

    return query_list

def get_crystal_default_filter():
    return []


# split out query creation per-game
def filter_df(df: pd.DataFrame, query_list: List[str], one_query: bool = False, fillna: object = 0, verbose: bool = True) -> (pd.DataFrame, List[str]):
    """

    :param df: dataframe to filter
    :param query_list: list of queries for filter
    :param one_query: bool to do the query (faster) as one query or seperate queries (slower, gives more info)
    :param fillna: value to fill NaNs with
    :param verbose: whether to input information
    :return: (df, List[str])
    """
    df = df.rename({'continue': '_continue'}, axis=1)
    filter_args = locals()
    filter_args.pop('df')
    filter_meta = [f'*arg* filter_args = {filter_args}']

    def append_meta_str(q, shape):
        outstr = f'Query: {q}, output_shape: {shape}'
        filter_meta.append(outstr)
        if verbose:
            print(outstr)

    append_meta_str('Intial Shape', df.shape)

    if not one_query:
        for q in query_list:
            df = df.query(q)
            append_meta_str(q, df.shape)
    else:  # do the whole query at once
        full_query = ' & '.join([f"({q})" for q in query_list])
        print('full_query:', full_query)
        df = df.query(full_query)
        append_meta_str(full_query, df.shape)

    if fillna is not None:
        df = df.fillna(fillna)
        filter_meta.append(f'Filled NaN with {fillna}')
    return df.rename({'_continue': 'continue'}), filter_meta


def create_new_base_features_lakeland(df, verbose=False, avg_tile_hover_lvl_range=None):
    new_base_feature_args = locals()
    new_base_feature_args.pop('df')
    new_feat_meta = [f'*arg* new_feat_args = {new_base_feature_args}']
    items = ['home', 'food', 'farm', 'fertilizer', 'livestock', 'skimmer', 'sign', 'road']
    # player hover, buy aggregate features
    if verbose:
        print('Calculating player hover, buy aggregate features...')
    hover_f_avg = lambda i, item: f'lvl{i}_avg_num_tiles_hovered_before_placing_{item}'
    hover_f_tot = lambda i, item: f'lvl{i}_tot_num_tiles_hovered_before_placing_{item}'
    for i in range(10):
        for item in items:
            df[hover_f_tot(i, item)] = df[hover_f_avg(i, item)].fillna(0) * df[f'lvl{i}_count_buy_{item}'].fillna(0)
        df[hover_f_tot(i, "buys")] = df.loc[:, hover_f_tot(i, "home"):hover_f_tot(i, "road")].sum(axis=1)
        df[f'lvl{i}_count_buys'] = df.loc[:, f'lvl{i}_count_buy_home':f'lvl{i}_count_buy_road'].fillna(0).sum(axis=1)
    if avg_tile_hover_lvl_range is not None:
        prefix = f'weighted_avg_lvl_{avg_tile_hover_lvl_range[0]}_to_{avg_tile_hover_lvl_range[-1]}_'
        for item in items:
            fname = f'{prefix}avg_num_tiles_hovered_before_placing_{item}'
            item_sum_query = '(' + ' + '.join([hover_f_tot(i, item) for i in avg_tile_hover_lvl_range]) + ')'
            tot_sum_query = '(' + ' + '.join([f'lvl{i}_count_buys' for i in avg_tile_hover_lvl_range]) + ')'
            df = df.eval(f'{fname} = {item_sum_query} / {tot_sum_query}')

    # count achievement features, achs_per_sec features
    if verbose:
        print('Calculating count achievement features, achs_per_sec features...')
    time_to_ach = lambda ach: f'sess_time_to_{ach}_achievement'
    pop_achs = ['exist', 'group', 'town', 'city']
    farm_achs = ['farmer', 'farmers', 'farmtown', 'megafarm']
    money_achs = ['paycheck', 'thousandair', 'stability', 'riches']
    bloom_achs = ['bloom', 'bigbloom', 'hugebloom', 'massivebloom']
    df['count_pop_achs'] = df[[time_to_ach(ach) for ach in pop_achs]].astype(bool).sum(axis=1)
    df['count_farm_achs'] = df[[time_to_ach(ach) for ach in farm_achs]].astype(bool).sum(axis=1)
    df['count_money_achs'] = df[[time_to_ach(ach) for ach in money_achs]].astype(bool).sum(axis=1)
    df['count_bloom_achs'] = df[[time_to_ach(ach) for ach in bloom_achs]].astype(bool).sum(axis=1)
    for ach_type in ['pop', 'farm', 'money', 'bloom']:
        df[f'{ach_type}_achs_per_sess_second'] = df[f'count_{ach_type}_achs'] / df['sessDuration']

    return df, new_feat_meta

def create_new_base_features_waves(df, verbose=False):
    """
    Currently a stub. Used to create new features from existing ones. See create_new_base_features_lakeland for example.
    :param df:
    :param verbose:
    :return:
    """
    new_base_feature_args = locals()
    new_base_feature_args.pop('df')
    new_feat_meta = [f'*arg* new_feat_args = {new_base_feature_args}']

    random_lvls = [9, 17, 29, 32, 34]
    random_completeCount_sum = ' + '.join([f'lvl{lvl}_completeCount' for lvl in random_lvls])
    df = df.eval(f'sum_random_complete_count = {random_completeCount_sum}')

    return df, new_feat_meta


def create_new_base_features_crystal(df, verbose=False):
    """

    Currently a stub. Used to create new features from existing ones. See create_new_base_features_lakeland for example.
    :param df:
    :param verbose:
    :return:
    """
    new_base_feature_args = locals()
    new_base_feature_args.pop('df')
    new_feat_meta = [f'*arg* new_feat_args = {new_base_feature_args}']

    return df, new_feat_meta


def describe_lvl_feats_lakeland(df, fbase_list, lvl_range, level_time=300, level_overlap=30):

    """
    Calculates sum/avg of given level base features (fnames without lvlN_ prefix) in the level range.
    Will automatically filter out players who did not complete the given level range in the df
    May have a bug.

    :param level_time: number of seconds per level (window)
    :param level_overlap: number of overlap seconds per level (window)
    :rtype: (df, List[str]) where the new df includes sum_ and avg_lvl_A_to_B.
    :param df: dataframe to pull from and append to
    :param fbase_list: list of feature bases (fnames without lvlN_ prefix)
    :param lvl_range: range of levels to choose. typically range(min_level, max_level+1)
    """
    metadata = []
    metadata.append(f'*arg* lvlfeats = {fbase_list}')
    metadata.append(f'*arg* lvlrange = {lvl_range}')
    if not fbase_list:
        return df, metadata
    lvl_start, lvl_end = lvl_range[0], lvl_range[-1]
    query = f'sessDuration > {(level_time - level_overlap) * (lvl_end) + level_time}'
    df = df.query(query)
    metadata.append(
        f'Describe Level Feats lvls {lvl_start} to {lvl_end}. Assuming WINDOW_SIZE_SECONDS={level_time} and WINDOW_OVERLAP_SECONDS={level_overlap}, filtered by ({query})')
    fromlvl, tolvl = lvl_range[0], lvl_range[-1]
    sum_prefix = f'sum_lvl_{fromlvl}_to_{tolvl}_'
    avg_prefix = f'avg_lvl_{fromlvl}_to_{tolvl}_'
    for fn in fbase_list:
        tdf = df[[f'lvl{i}_{fn}' for i in lvl_range]].fillna(0)
        df[sum_prefix + fn] = tdf.sum(axis=1)
        df[avg_prefix + fn] = tdf.mean(axis=1)
    return df, metadata

def describe_lvl_feats_crystal(df, fbase_list, lvl_range):
    """
    Calculates sum/avg of given level base features (fnames without lvlN_ prefix) in the level range.
    May have a bug.

    :rtype: (df, List[str]) where the new df includes sum_ and avg_lvl_A_to_B
    :param df: dataframe to pull from and append to
    :param fbase_list: list of feature bases (fnames without lvlN_ prefix)
    :param lvl_range: range of levels to choose. typically range(min_level, max_level+1)
    """
    metadata = []
    metadata.append(f'*arg* lvlfeats = {fbase_list}')
    metadata.append(f'*arg* lvlrange = {lvl_range}')
    if not fbase_list:
        return df, metadata

    # TODO: Add filter for levels we don't want, like the one from lakeland
    # query = f'sessDuration > {(level_time - level_overlap) * (lvl_end) + level_time}'
    # df = df.query(query)
    # metadata.append(
    #     f'Describe Level Feats lvls {lvl_start} to {lvl_end}. Assuming WINDOW_SIZE_SECONDS={level_time} and WINDOW_OVERLAP_SECONDS={level_overlap}, filtered by ({query})')

    fromlvl, tolvl = lvl_range[0], lvl_range[-1]
    sum_prefix = f'sum_lvl_{fromlvl}_to_{tolvl}_'
    avg_prefix = f'avg_lvl_{fromlvl}_to_{tolvl}_'
    for fn in fbase_list:
        tdf = df[[f'lvl{i}_{fn}' for i in lvl_range]].fillna(0)
        df[sum_prefix + fn] = tdf.sum(axis=1)
        df[avg_prefix + fn] = tdf.mean(axis=1)
    return df, metadata

def describe_lvl_feats_waves(df, fbase_list, lvl_range):

    """
    Calculates sum/avg of given level base features (fnames without lvlN_ prefix) in the level range.
    May have a bug.

    :rtype: (df, List[str]) where the new df includes sum_ and avg_lvl_A_to_B
    :param df: dataframe to pull from and append to
    :param fbase_list: list of feature bases (fnames without lvlN_ prefix)
    :param lvl_range: range of levels to choose. typically range(min_level, max_level+1)
    """
    metadata = []
    metadata.append(f'*arg* lvlfeats = {fbase_list}')
    metadata.append(f'*arg* lvlrange = {lvl_range}')
    if not fbase_list:
        return df, metadata
    lvl_start, lvl_end = lvl_range[0], lvl_range[-1]
    # TODO: Add filter for levels we don't want, like the one from lakeland
    # query = f'sessDuration > {(level_time - level_overlap) * (lvl_end) + level_time}'
    # df = df.query(query)
    # metadata.append(
    #     f'Describe Level Feats lvls {lvl_start} to {lvl_end}. Assuming WINDOW_SIZE_SECONDS={level_time} and WINDOW_OVERLAP_SECONDS={level_overlap}, filtered by ({query})')
    fromlvl, tolvl = lvl_range[0], lvl_range[-1]
    sum_prefix = f'sum_lvl_{fromlvl}_to_{tolvl}_'
    avg_prefix = f'avg_lvl_{fromlvl}_to_{tolvl}_'
    for fn in fbase_list:
        tdf = df[[f'lvl{i}_{fn}' for i in lvl_range]].fillna(0)
        df[sum_prefix + fn] = tdf.sum(axis=1)
        df[avg_prefix + fn] = tdf.mean(axis=1)
    return df, metadata



def get_feat_selection_lakeland(df,  max_lvl=9):
    """
    Gets the feature selection widget.
    :param df:
    :param max_lvl:
    :return:
    """


    start_level = widgets.IntSlider(value=0,min=0,max=max_lvl,step=1,description='Start Level:',
                                    disabled=False,continuous_update=False,orientation='horizontal',readout=True,readout_format='d')
    end_level = widgets.IntSlider(value=0,min=0,max=max_lvl,step=1,description='End Level:',
                                  disabled=False,continuous_update=False,orientation='horizontal',readout=True,readout_format='d')
    level_selection = widgets.GridBox([start_level, end_level])

    def change_start_level(change):
        end_level.min = start_level.value
        if end_level.value < start_level.value:
            end_level.value = start_level.value
    start_level.observe(change_start_level, names="value")


    lvl_feats = sorted(set([f[5:] for f in df.columns if f.startswith('lvl')]))
    sess_feats = sorted(set([f[5:] for f in df.columns if f.startswith('sess_')]))
    other_feats = sorted(set([f for f in df.columns if not f.startswith('lvl') and not f.startswith('sess_')]))
    selection_widget = widgets.GridBox([multi_checkbox_widget(lvl_feats,'lvl'),
                                        multi_checkbox_widget(sess_feats,'sess'),
                                        multi_checkbox_widget(other_feats,'other'),
                                        level_selection],
                                       layout=widgets.Layout(grid_template_columns=f"repeat(3, 500px)"))

    return selection_widget

def get_feat_selection_crystal(df,  max_lvl=8):
    """
    Gets the feature selection widget.
    :param df:
    :param max_lvl:
    :return:
    """

    start_level = widgets.IntSlider(value=0, min=0, max=max_lvl, step=1, description='Start Level:',
                                    disabled=False, continuous_update=False, orientation='horizontal', readout=True,
                                    readout_format='d')
    end_level = widgets.IntSlider(value=0, min=0, max=max_lvl, step=1, description='End Level:',
                                  disabled=False, continuous_update=False, orientation='horizontal', readout=True,
                                  readout_format='d')
    level_selection = widgets.GridBox([start_level, end_level])

    def change_start_level(change):
        end_level.min = start_level.value
        if end_level.value < start_level.value:
            end_level.value = start_level.value

    start_level.observe(change_start_level, names="value")

    lvl_feats = sorted(set([f[5:] for f in df.columns if f.startswith('lvl')]))
    sess_feats = sorted(set([f[7:] for f in df.columns if f.startswith('session')]))
    other_feats = sorted(set([f for f in df.columns if not f.startswith('lvl') and not f.startswith('session')]))
    selection_widget = widgets.GridBox([multi_checkbox_widget(lvl_feats, 'lvl'),
                                        multi_checkbox_widget(sess_feats, 'session'),
                                        multi_checkbox_widget(other_feats, 'other'),
                                        level_selection],
                                       layout=widgets.Layout(grid_template_columns=f"repeat(3, 500px)"))

    return selection_widget

def get_feat_selection_waves(df, max_lvl=34):
    """
    Gets the feature selection widget.
    :param df:
    :param max_lvl:
    :return:
    """

    start_level = widgets.IntSlider(value=0, min=0, max=max_lvl, step=1, description='Start Level:',
                                    disabled=False, continuous_update=False, orientation='horizontal', readout=True,
                                    readout_format='d')
    end_level = widgets.IntSlider(value=0, min=0, max=max_lvl, step=1, description='End Level:',
                                  disabled=False, continuous_update=False, orientation='horizontal', readout=True,
                                  readout_format='d')
    level_selection = widgets.GridBox([start_level, end_level])

    def change_start_level(change):
        end_level.min = start_level.value
        if end_level.value < start_level.value:
            end_level.value = start_level.value

    start_level.observe(change_start_level, names="value")

    lvl_feats = sorted(set([''.join(f.split('_')[1:]) for f in df.columns if f.startswith('lvl')]))
    sess_feats = sorted(set([f[7:] for f in df.columns if f.startswith('session')]))
    other_feats = sorted(set([f for f in df.columns if not f.startswith('lvl') and not f.startswith('session')]))
    selection_widget = widgets.GridBox([multi_checkbox_widget(lvl_feats, 'lvl'),
                                        multi_checkbox_widget(sess_feats, 'sess'),
                                        multi_checkbox_widget(other_feats, 'other'),
                                        level_selection],
                                       layout=widgets.Layout(grid_template_columns=f"repeat(3, 500px)"))

    return selection_widget
def get_selected_feature_list_lakeland(selection_widget):
    """

    :param selection_widget:
    :return: list of features selected
    """

    sess_feats = [f'sess_{s.description}' for s in selection_widget.children[1].children[1].children if s.value]
    other_feats = [s.description for s in selection_widget.children[2].children[1].children if s.value]
    lvl_feats, lvl_range = get_level_feats_and_range(selection_widget)
    all_lvl_feats = [f'lvl{i}_{f}' for f in lvl_feats for i in lvl_range]
    return all_lvl_feats + sess_feats + other_feats
def get_selected_feature_list_crystal(selection_widget):
    """

    :param selection_widget:
    :return: list of features selected
    """

    sess_feats = [f'session{s.description}' for s in selection_widget.children[1].children[1].children if s.value]
    other_feats = [s.description for s in selection_widget.children[2].children[1].children if s.value]
    lvl_feats, lvl_range = get_level_feats_and_range(selection_widget)
    all_lvl_feats = [f'lvl{i}_{f}' for f in lvl_feats for i in lvl_range]
    return all_lvl_feats + sess_feats + other_feats
def get_selected_feature_list_waves(selection_widget):
    """

    :param selection_widget:
    :return: list of features selected
    """

    sess_feats = [f'session{s.description}' for s in selection_widget.children[1].children[1].children if s.value]
    other_feats = [s.description for s in selection_widget.children[2].children[1].children if s.value]
    lvl_feats, lvl_range = get_level_feats_and_range(selection_widget)
    all_lvl_feats = [f'lvl{i}_{f}' for f in lvl_feats for i in lvl_range]
    return all_lvl_feats + sess_feats + other_feats


def get_level_feats_and_range(selection_widget) -> (List[str], Iterable):
    """

    :param selection_widget:
    :return: List of fbases from selection_widget and level range
    """
    lvl_start_widget = selection_widget.children[3].children[0]
    lvl_end_widget = selection_widget.children[3].children[1]
    lvl_feats = [s.description for s in selection_widget.children[0].children[1].children if s.value]
    lvl_range = range(lvl_start_widget.value, lvl_end_widget.value + 1)
    # lvl_feat_widget = feat_selection.children[0]
    # lvl_start_widget = feat_selection.children[3].children[0]
    # lvl_end_widget = feat_selection.children[3].children[1]
    # lvl_range = range(lvl_start_widget.value, lvl_end_widget.value+1)
    return lvl_feats, lvl_range


def multi_checkbox_widget(descriptions, category):
    """ Widget with a search field and lots of checkboxes """
    search_widget = widgets.Text(layout={'width': '400px'}, description=f'Search {category}:')
    options_dict = {description: widgets.Checkbox(description=description, value=False,
                                                  layout={'overflow-x': 'scroll', 'width': '400px'}, indent=False) for
                    description in descriptions}
    options = [options_dict[description] for description in descriptions]
    options_widget = widgets.VBox(options, layout={'overflow': 'scroll', 'height': '400px'})
    multi_select = widgets.VBox([search_widget, options_widget])

    # Wire the search field to the checkboxes
    def on_text_change(change):
        search_input = change['new']
        if search_input == '':
            # Reset search field
            for d in descriptions:
                options_dict[d].layout.visibility = 'visible'
                options_dict[d].layout.height = 'auto'
        elif search_input[-1] == '$':
            search_input = search_input[:-1]
            # Filter by search field using difflib.
            for d in descriptions:
                if search_input in d:
                    options_dict[d].layout.visibility = 'visible'
                    options_dict[d].layout.height = 'auto'
                else:
                    options_dict[d].layout.visibility = 'hidden'
                    options_dict[d].layout.height = '0px'
            # close_matches = [d for d in descriptions if search_input in d] #difflib.get_close_matches(search_input, descriptions, cutoff=0.0)
            # new_options = [options_dict[description] for description in close_matches]
        # options_widget.children = new_options

    search_widget.observe(on_text_change, names='value')
    return multi_select


def reduce_feats(df, featlist):
    """
    Takes in a df and outputs only the given columns in featlist
    :param df:
    :param featlist:
    :return:
    """
    return df[featlist].copy(), [f'*arg* finalfeats = {featlist}']


def reduce_outliers(df, z_thresh, show_graphs=True):
    """
    Takes in df and z_thresh, shows box plots, and outputs graph with points of zscore>z_thresh removed.
    Does not always work properly. Does not seem to tolerate NaNs.
    TODO: fix.
    :param df:
    :param z_thresh:
    :param show_graphs:
    :return:
    """
    meta = []
    meta.append(f"Original Num Rows: {len(df)}")
    meta.append(f"*arg* zthresh = {z_thresh}")
    if z_thresh is None:
        return df, meta
    df.plot(kind='box', title=f'Original Data n={len(df)}', figsize=(20, 5))
    z = np.abs(stats.zscore(df))
    no_outlier_df = df[(z < z_thresh).all(axis=1)]
    meta.append(f'Removed points with abs(ZScore) >= {z_thresh}. Reduced num rows: {len(no_outlier_df)}')
    no_outlier_df.plot(kind='box', title=f'ZScore < {z_thresh} n={len(no_outlier_df)}', figsize=(20, 5))
    return no_outlier_df, meta


def full_filter(get_df_func, options) -> (pd.DataFrame, List[str]):
    """
    Takes in a df_getter_function and options group.
    Outputs the filtered df and the meta.
    :param get_df_func:
    :param options:
    :return:
    """
    df, import_meta = get_df_func()
    filtered_df, filter_meta = filter_df(df, **options.filter_args)
    game = options.game.upper()
    if game == 'LAKELAND':
        new_feat_df, new_feat_meta = create_new_base_features_lakeland(filtered_df, **options.new_feat_args)
        aggregate_df, aggregate_meta = describe_lvl_feats_lakeland(new_feat_df, options.lvlfeats, options.lvlrange)
    elif game == 'CRYSTAL':
        new_feat_df, new_feat_meta = create_new_base_features_crystal(filtered_df, **options.new_feat_args)
        aggregate_df, aggregate_meta = describe_lvl_feats_crystal(new_feat_df, options.lvlfeats, options.lvlrange)
    elif game == 'WAVES':
        new_feat_df, new_feat_meta = create_new_base_features_waves(filtered_df, **options.new_feat_args)
        aggregate_df, aggregate_meta = describe_lvl_feats_waves(new_feat_df, options.lvlfeats, options.lvlrange)
    else:
        assert False
    reduced_df, reduced_meta = reduce_feats(aggregate_df, options.finalfeats)
    reduced_df = reduced_df.fillna(0) # hack while NaNs are popping up in aggregate df or newfeatdf TODO: Fix this. It never used to be an issue.
    if options.zthresh is not None:
        final_df, outlier_meta = reduce_outliers(reduced_df, options.zthresh)
    else:
        final_df, outlier_meta = reduced_df, []
    final_meta = import_meta + filter_meta + new_feat_meta + aggregate_meta + reduced_meta + outlier_meta
    return final_df, final_meta


if __name__ == '__main__':
    pass
