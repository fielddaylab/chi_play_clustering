from collections import namedtuple


class Options:
    options = namedtuple('Options',
                         ['game', 'name', 'filter_args', 'new_feat_args', 'lvlfeats', 'lvlrange', 'finalfeats',
                          'zthresh', 'finalfeats_readable'])
    lakeland_actions_lvl0 = options('lakeland',
                                    'actions_lvl0',
                                    {'query_list': ['debug == 0', 'sess_ActiveEventCount >= 10',
                                                    'sessDuration >= 300',
                                                    '_continue == 0',
                                                    'sess_avg_num_tiles_hovered_before_placing_home > 1']},
                                    {'avg_tile_hover_lvl_range': range(0, 1)},
                                    ['count_buy_home', 'count_buy_farm', 'count_buy_livestock', 'count_buys'],
                                    range(0, 1),
                                    ['weighted_avg_lvl_0_to_0_avg_num_tiles_hovered_before_placing_farm',
                                     'sum_lvl_0_to_0_count_buy_home',
                                     'sum_lvl_0_to_0_count_buy_farm',
                                     'sum_lvl_0_to_0_count_buy_livestock',
                                     'sum_lvl_0_to_0_count_buys'],
                                    3,
                                    ['hovers\nbefore\nfarm', 'home', 'farm', 'livestock', 'buys']
                                    )
    lakeland_actions_lvl01 = options('lakeland',
                                     'actions_lvl01',
                                     {'query_list': ['debug == 0', 'sess_ActiveEventCount >= 10',
                                                     f'sessDuration >= {60*5}', f'sessDuration < {60*45}',
                                                     '_continue == 0',
                                                     'sess_avg_num_tiles_hovered_before_placing_home > 1']},
                                     {'avg_tile_hover_lvl_range': range(0, 2)},
                                     ['count_buy_home', 'count_buy_farm', 'count_buy_livestock', 'count_buys'],
                                     range(0, 2),
                                     ['weighted_avg_lvl_0_to_1_avg_num_tiles_hovered_before_placing_farm',
                                      'sum_lvl_0_to_1_count_buy_home',
                                      'sum_lvl_0_to_1_count_buy_farm',
                                      'sum_lvl_0_to_1_count_buy_livestock',
                                      'sum_lvl_0_to_1_count_buys'],
                                     3,
                                     ['hovers\nbefore\nfarm', 'home', 'farm', 'livestock', 'buys']
                                     )
    lakeland_actions_sess = options('lakeland',
                                     'actions_sess',
                                     {'query_list': ['debug == 0', 'sess_ActiveEventCount >= 10',
                                                     f'sessDuration >= {60*5}', f'sessDuration < {60*45}',
                                                     '_continue == 0',
                                                     'sess_avg_num_tiles_hovered_before_placing_home > 1']},
                                     {'avg_tile_hover_lvl_range': range(0, 2)},
                                     ['count_buy_home', 'count_buy_farm', 'count_buy_livestock', 'count_buys'],
                                     range(0, 2),
                                     ['weighted_avg_lvl_0_to_1_avg_num_tiles_hovered_before_placing_farm',
                                      'sess_count_buy_home',
                                      'sess_count_buy_farm',
                                      'sess_count_buy_livestock',
                                      'sess_count_buys'],
                                     3,
                                     ['hovers\nbefore\nfarm', 'home', 'farm', 'livestock', 'buys']
                                     )
    lakeland_actions_lvl0_only_rain = options('lakeland',
                                              'actions_lvl0_only_rain',
                                              {'query_list': ['debug == 0', 'sess_ActiveEventCount >= 10',
                                                              'sessDuration >= 300',
                                                              '_continue == 0',
                                                              'sess_avg_num_tiles_hovered_before_placing_home > 1']},
                                              {'avg_tile_hover_lvl_range': range(0, 1)},
                                              ['count_buy_home', 'count_buy_farm', 'count_buy_livestock', 'count_buys'],
                                              range(0, 1),
                                              ['weighted_avg_lvl_0_to_0_avg_num_tiles_hovered_before_placing_farm',
                                               'sum_lvl_0_to_0_count_buy_home',
                                               'sum_lvl_0_to_0_count_buy_farm',
                                               'sum_lvl_0_to_0_count_buy_livestock',
                                               'sum_lvl_0_to_0_count_buys'],
                                              3,
                                              ['hovers\nbefore\nfarm', 'home', 'farm', 'livestock', 'buys']
                                              )
    lakeland_test_poop_placement_skimmer = options('lakeland',
                                                   'test_poop_placement_skimmer',
                                                   {'query_list': ['debug == 0', 'sess_ActiveEventCount >= 10',
                                                                   'sessDuration >= 300',
                                                                   '_continue == 0']},
                                                   {'avg_tile_hover_lvl_range': range(0, 2), 'verbose': False},
                                                   ['avg_distance_between_poop_placement_and_lake',
                                                    'count_buy_skimmer'],
                                                   range(0, 6),
                                                   ['lvl4_avg_distance_between_poop_placement_and_lake',
                                                    'sum_lvl_0_to_5_avg_distance_between_poop_placement_and_lake'],
                                                   3,
                                                   []
                                                   )
    lakeland_achs_achs_per_sess_second_sessDur = options('lakeland',
                                                         'achs_achs_per_sess_second_sessDur',
                                                         {'query_list': ['debug == 0', 'sess_ActiveEventCount >= 10',
                                                                         'sessDuration >= 300', '_continue == 0']},
                                                         {'avg_tile_hover_lvl_range': None, 'verbose': False},
                                                         [],
                                                         range(0, 1),
                                                         ['bloom_achs_per_sess_second', 'farm_achs_per_sess_second',
                                                          'money_achs_per_sess_second', 'pop_achs_per_sess_second',
                                                          'sessDuration'],
                                                         3,
                                                         ['bloom', 'farm', 'money', 'population', 'session time']
                                                         )
    lakeland_feedback_lv01 = options('lakeland',
                                     'feedback_lv01',
                                     {'query_list': ['debug == 0', 'sess_ActiveEventCount >= 10', 'sessDuration >= 300',
                                                     '_continue == 0']},
                                     {'avg_tile_hover_lvl_range': None, 'verbose': False},
                                     ['count_blooms', 'count_deaths', 'count_farmfails', 'count_food_produced',
                                      'count_milk_produced'],
                                     range(0, 2),
                                     ['avg_lvl_0_to_1_count_deaths', 'avg_lvl_0_to_1_count_farmfails',
                                      'avg_lvl_0_to_1_count_food_produced', 'avg_lvl_0_to_1_count_milk_produced'],
                                     # 'avg_lvl_0_to_1_count_blooms', ,
                                     3,
                                     ['deaths', 'farmfails', 'food', 'milk']
                                     )


    lakeland_feedback_lv01_with_bloom = options('lakeland',
                                     'feedback_lv01_with_bloom',
                                     {'query_list': ['debug == 0', 'sess_ActiveEventCount >= 10', 'sessDuration >= 300',
                                                     '_continue == 0']},
                                     {'avg_tile_hover_lvl_range': None, 'verbose': False},
                                     ['count_blooms', 'count_deaths', 'count_farmfails', 'count_food_produced',
                                      'count_milk_produced'],
                                     range(0, 2),
                                     ['avg_lvl_0_to_1_count_deaths', 'avg_lvl_0_to_1_count_farmfails',
                                      'avg_lvl_0_to_1_count_food_produced', 'avg_lvl_0_to_1_count_milk_produced',
                                     'avg_lvl_0_to_1_count_blooms'],
                                     3,
                                     ['deaths', 'farmfails', 'food', 'milk', 'blooms']
                                     )

    lakeland_feedback_sess_with_bloom = options('lakeland',
                                                'feedback_sess_with_bloom',
                                                {'query_list': ['debug == 0', 'sess_ActiveEventCount >= 10', 'sessDuration >= 300',
                                                                '_continue == 0']},
                                                {'avg_tile_hover_lvl_range': None, 'verbose': False},
                                                ['count_blooms', 'count_deaths', 'count_farmfails', 'count_food_produced',
                                                 'count_milk_produced'],
                                                range(0, 2),
                                                ['sess_count_deaths', 'sess_count_farmfails',
                                                 'sess_count_food_produced', 'sess_count_milk_produced',
                                                 'sess_count_blooms'],
                                                3,
                                                ['deaths', 'farmfails', 'food', 'milk', 'blooms']
                                                )
    crystal_dummy = options('crystal',
                            'dummy',
                            {'query_list': [], 'one_query': False, 'fillna': 0, 'verbose': False},
                            {'verbose': False},
                            ['clearBtnPresses', 'durationInSecs'],
                            range(0, 5),
                            ['lvl0_completesCount', 'lvl0_menuBtnCount', 'sessionMuseumDurationInSecs',
                             'sum_lvl_0_to_4_clearBtnPresses', 'sum_lvl_0_to_4_durationInSecs'],
                            3,
                            []
                            )

    crystal_progression = options('crystal',
            'progression',
            {'query_list': [f'sessionDurationInSecs < {45*60}', 'lvl0_durationInSecs > 0'], 'one_query': False, 'fillna': 0, 'verbose': False},
            {'verbose': False},
            ['completesCount', 'finalScore'],
            range(0, 9),
            ['sessionDurationInSecs', 'sum_lvl_0_to_8_completesCount', 'sum_lvl_0_to_8_finalScore'],
            None,
            ['Session Duration', 'Level Completes', 'Total Score'],
            )

    crystal_actions = options('crystal',
            'actions',
            {'query_list': ['lvl0_durationInSecs > 0', 'lvl1_durationInSecs > 0', 'lvl2_durationInSecs > 0',
                            'lvl3_durationInSecs > 0', 'lvl4_durationInSecs > 0', 'QA1_questionCorrect==QA1_questionCorrect'], 'one_query': False, 'fillna': 0,
             'verbose': False},
            {'verbose': False},
            ['avgMoleculeDragDurationInSecs', 'clearBtnPresses', 'moleculeMoveCount', 'singleRotateCount',
             'stampRotateCount'],
            range(0, 5),
            ['sum_lvl_0_to_4_avgMoleculeDragDurationInSecs', 'sum_lvl_0_to_4_clearBtnPresses',
             'sum_lvl_0_to_4_moleculeMoveCount', 'sum_lvl_0_to_4_singleRotateCount', 'sum_lvl_0_to_4_stampRotateCount'],
            None,
            ['Lv0-4 Avg Drag Duration', 'Lv0-4 Clears', 'Lv0-4 Moves', 'Lv0-4 Single Rotates', 'Lv0-4 Stamp Rotates']
            )

    crystal_actions_sess = options('crystal',
                              'actions_filter_sess',
                              {'query_list': ['lvl0_durationInSecs > 0', 'lvl1_durationInSecs > 0'], 'one_query': False, 'fillna': 0,
                               'verbose': False},
                              {'verbose': False},
                              ['avgMoleculeDragDurationInSecs', 'clearBtnPresses', 'moleculeMoveCount', 'singleRotateCount',
                               'stampRotateCount'],
                              range(0, 9),
                              ['sum_lvl_0_to_8_avgMoleculeDragDurationInSecs', 'sum_lvl_0_to_8_clearBtnPresses',
                               'sum_lvl_0_to_8_moleculeMoveCount', 'sum_lvl_0_to_8_singleRotateCount', 'sum_lvl_0_to_8_stampRotateCount'],
                              None,
                              []
                              )
    crystal_feedback = options('crystal',
	'feedback',
	{'query_list': ['lvl0_durationInSecs > 0', 'lvl1_durationInSecs > 0', 'lvl2_durationInSecs > 0', 'lvl3_durationInSecs > 0', 'lvl4_durationInSecs > 0', 'QA1_questionCorrect==QA1_questionCorrect'], 'one_query': False, 'fillna': 0, 'verbose': False},
	{'verbose': False},
	['finalScore'],
	range(0, 5),
	['QA0_questionCorrect', 'QA1_questionCorrect', 'sum_lvl_0_to_4_finalScore'],
	None,
	['Q0 Correct', 'Q1 Correct', 'Score Lv0-4']
)

    waves_dummy = options('waves',
                          'dummy',
                          {'query_list': [], 'one_query': False, 'fillna': 0, 'verbose': False},
                          {'verbose': False},
                          ['closenessR2', 'rangeIntercept', 'totalFails'],
                          range(0, 12),
                          ['avg_lvl_0_to_11_totalFails', 'overallPercentAmplitudeMoves',
                           'overallPercentWavelengthMoves', 'overallSliderAvgStdDevs'],
                          3,
                          []
                          )
    waves_actions_lv016 = options('waves',
        'actions_lv016',
            {'query_list': ['QA3_questionCorrect==QA3_questionCorrect'], 'one_query': False, 'fillna': 0, # A==A meaning A is not NaN
             'verbose': False},
            {'verbose': False},
            ['menuBtnCount','beginCount', 'totalArrowMoves', 'totalResets', 'totalSkips', 'totalSliderMoves'],
            range(0, 17),
            ['sum_lvl_0_to_16_beginCount', 'sum_lvl_0_to_16_menuBtnCount',
             'sum_lvl_0_to_16_totalArrowMoves', 'sum_lvl_0_to_16_totalResets', 'sum_lvl_0_to_16_totalSkips',
             'sum_lvl_0_to_16_totalSliderMoves'],
            None,
            []
            )
    waves_actions_lv016_no_skips = options('waves',
                                  'actions_lv016_no_skips',
                                  {'query_list': ['QA3_questionCorrect==QA3_questionCorrect'], 'one_query': False, 'fillna': 0,
                                   'verbose': False},
                                  {'verbose': False},
                                  ['menuBtnCount','beginCount', 'totalArrowMoves', 'totalResets', 'totalSkips', 'totalSliderMoves'],
                                  range(0, 17),
                                  ['sum_lvl_0_to_16_beginCount', 'sum_lvl_0_to_16_menuBtnCount',
                                   'sum_lvl_0_to_16_totalArrowMoves', 'sum_lvl_0_to_16_totalResets',
                                   'sum_lvl_0_to_16_totalSliderMoves'],
                                  None,
                                  []
                                  )

    waves_feedback_lv016 = options('waves',
            'feedback_lv016',
            {'fillna': 0, 'one_query': False, 'query_list': ['QA3_questionCorrect==QA3_questionCorrect'],
             'verbose': False},
            {'verbose': False},
            ['closenessIntercept', 'closenessR2', 'closenessSlope', 'succeedCount', 'totalFails'],
            range(0, 17),
            ['sum_lvl_0_to_16_closenessIntercept', 'sum_lvl_0_to_16_closenessR2', 'sum_lvl_0_to_16_closenessSlope',
             'sum_lvl_0_to_16_succeedCount', 'sum_lvl_0_to_16_totalFails', 'sum_random_complete_count'],
            None,
            []
            )

    waves_progression = options('waves',
	'progression',
	{'fillna': 0, 'one_query': False, 'query_list': [], 'verbose': False},
	{'verbose': False},
	['completed', 'totalLevelTime'],
	range(0, 35),
	['sum_lvl_0_to_34_completed', 'sum_lvl_0_to_34_totalLevelTime', 'sum_random_complete_count'],
	None,
	[]
)