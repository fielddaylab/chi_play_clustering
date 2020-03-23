jupyter2

## Clustering for CHI Play Paper

Instructions:

1. Install anaconda from https://www.anaconda.com/
1. Install pycharm community edition from https://www.jetbrains.com/pycharm/download/
1. Open /Notebooks/Clustering in jupyter (Windows: Open anaconda prompt, cd to repo, enter command "jupyter notebook Notebooks\Clustering"
1. click on the set filtering options file you want
1. under "Imports" change BASE_PATH to the path that the repo is in
1. run until the level features to aggrtegate widget appears. Select lvl features and a lvl range to aggregate (calculates sum/avg)
1. run until final features widget appears. select final features to include
1. run until end. copy the options(...) object.
1. open the repo folder in pycharm and paste the options object under src/options in the appropriate place. Change 'GAME' and 'NAME' to the respective game and a descriptive name.
1. (Optional) Change the empty brackets at the end of the options object to a readable format of the final features. For example ['lvl0_avg_num_tiles_hovered_before_placing_food', 'lvl0_avg_num_tiles_hovered_before_placing_road', 'sess_avg_num_tiles_hovered_before_placing_home', 'play_hour', 'play_second'] => ['hovers_before_food', 'hovers_before_road',....]
1. Under clustering_scripts, copy a .py file of the same game and rename it to the new description.
1. Update filter_options = Options.[new option name]
1. press the play button at the left of if \_\_name\_\_ = '\_\_main\_\_'
1. look at outputs and readjust variables if desired

Note: This has not been tested on other computers or macs and there might be (read: probably is) some bugs.
