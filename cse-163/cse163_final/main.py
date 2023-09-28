"""
Ethan Matthew Hsu, Aryan Kartik Shah, Charles Georges Faisandier
CSE 163AA
This program is creates various datasets and analyzes them in various ways in
order to answer the three questions in our project. The q1, q2, q3 found in
our function names refer to these qeustions. This program takes over ten
minutes to run and saves the resulting dataframes so that our testing can be
replicated by the user.
"""

import pandas as pd
import seaborn as sns
import requests
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d
sns.set_theme()


def get_cstandings(start=1950, end=2023) -> pd.DataFrame:
    """ This function takes two parameters, a start date, and an endate, and
    returns a dataframe containing the standings of every formula one
    constructor for every year from the start date (inclusive) to the end date
    (exclusive)
    """
    ssns_cstandings = pd.DataFrame()
    for year in range(start, end):
        time.sleep(0.01)
        response = requests.get('http://ergast.com/api/f1/' + str(year) +
                                '/constructorStandings.json?limit=1000')
        intermediate_json = pd.DataFrame(response.json())
        if len(intermediate_json.loc['StandingsTable', 'MRData']
               .get('StandingsLists')) != 0:
            l_dict_cstandings = intermediate_json\
                .loc['StandingsTable', 'MRData'].get('StandingsLists')[0]\
                .get('ConstructorStandings')
            for d_ind in range(len(l_dict_cstandings)):
                l_dict_cstandings[d_ind]['constructorId'] =\
                    l_dict_cstandings[d_ind].get('Constructor')\
                    .get('constructorId')
                l_dict_cstandings[d_ind]['nationality'] =\
                    l_dict_cstandings[d_ind]\
                    .get('Constructor').get('nationality')
                [l_dict_cstandings[d_ind].pop(key) for key in
                    ['Constructor', 'positionText']]
            year_constr_standings = pd.DataFrame(l_dict_cstandings)
            year_constr_standings.loc[:, 'season'] = year
            ssns_cstandings = pd.concat([ssns_cstandings,
                                         year_constr_standings])
    ssns_cstandings.reset_index(drop=True, inplace=True)
    ssns_cstandings = ssns_cstandings.astype({'position': 'int64',
                                              'points': 'float64',
                                              'wins': 'int64',
                                              'season': 'int64'})
    return ssns_cstandings


def get_dstandings(start=1950, end=2023):
    """ This function takes two parameters, a start date, and an end date, and
    returns a dataframe containing the standings of every formula one driver
    for every year from the start date (inclusive) to the end date (exclusive).
    """
    ssns_dstandings = pd.DataFrame()
    for year in range(start, end):
        time.sleep(0.01)
        response = requests.get('http://ergast.com/api/f1/' + str(year) +
                                '/driverStandings.json?limit=1000')
        intermediate_json = pd.DataFrame(response.json())
        l_dict_dstandings = intermediate_json.loc['StandingsTable', 'MRData']\
            .get('StandingsLists')[0].get('DriverStandings')
        for d_ind in range(len(l_dict_dstandings)):
            l_dict_dstandings[d_ind]['driverId'] = l_dict_dstandings[d_ind]\
                .get('Driver').get('driverId')
            l_dict_dstandings[d_ind]['givenName'] = l_dict_dstandings[d_ind]\
                .get('Driver').get('givenName')
            l_dict_dstandings[d_ind]['familyName'] = l_dict_dstandings[d_ind]\
                .get('Driver').get('familyName')
            l_dict_dstandings[d_ind]['dateOfBirth'] = l_dict_dstandings[d_ind]\
                .get('Driver').get('dateOfBirth')
            l_dict_dstandings[d_ind]['nationality'] = l_dict_dstandings[d_ind]\
                .get('Driver').get('nationality')
            l_dict_dstandings[d_ind]['constructorId'] = \
                l_dict_dstandings[d_ind]\
                .get('Constructors')[0].get('constructorId')
            [l_dict_dstandings[d_ind].pop(key) for key in
                ['Driver', 'Constructors', 'positionText']]
        year_dstandings = pd.DataFrame(l_dict_dstandings)
        year_dstandings.loc[:, 'season'] = year
        ssns_dstandings = pd.concat([ssns_dstandings, year_dstandings])
    ssns_dstandings.reset_index(drop=True, inplace=True)
    ssns_dstandings = ssns_dstandings.astype({'position': 'int64',
                                              'points': 'float64',
                                              'wins': 'int64',
                                              'season': 'int64'})
    return ssns_dstandings


def get_q1_data_ls(cstandings: pd.DataFrame,
                   dstandings: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes the resulting dataframe from get_cstandings and
    get_dstandings and returns a dataframe which has every constructor of every
    season with variables about the constructor's preceeding season.
    """
    q1_data_ls = cstandings\
        .drop(labels=['nationality'], axis=1)
    ssns_dstandings = dstandings
    for row_ind in q1_data_ls.index:
        season = q1_data_ls.loc[row_ind, 'season']
        constructor = q1_data_ls.loc[row_ind, 'constructorId']
        # add ls_position ls_points ls_wins
        ls_position = None
        ls_points = None
        ls_wins = None
        ls_cinfo = q1_data_ls.loc[(q1_data_ls['season'] == season - 1) &
                                  (q1_data_ls['constructorId'] == constructor)]
        if not ls_cinfo.empty:
            ls_position = ls_cinfo['position']
            ls_points = ls_cinfo['points']
            ls_wins = ls_cinfo['wins']
        q1_data_ls.at[row_ind, 'ls_rank'] = ls_position
        q1_data_ls.at[row_ind, 'ls_points'] = ls_points
        q1_data_ls.at[row_ind, 'ls_wins'] = ls_wins
        # add driver ls_position ls_points ls_wins
        drivers = ssns_dstandings\
            .loc[(ssns_dstandings['constructorId'] == constructor) &
                 (ssns_dstandings['season'] == season)].reset_index()
        d1_ls_position = None
        d2_ls_position = None
        d1 = None
        d2 = None
        d3 = None
        d4 = None
        if len(drivers) > 0:
            d1 = drivers.loc[0, 'driverId']
            d1_ls_q1_data_ls = ssns_dstandings\
                .loc[(ssns_dstandings['season'] == season - 1) &
                     (ssns_dstandings['driverId'] == d1)].reset_index()
            if not d1_ls_q1_data_ls.empty:
                d1_ls_position = d1_ls_q1_data_ls.loc[0, 'position']
        if len(drivers) > 1:
            d2 = drivers.loc[1, 'driverId']
            d2_ls_q1_data_ls = ssns_dstandings\
                .loc[(ssns_dstandings['season'] == season - 1) &
                     (ssns_dstandings['driverId'] == d2)].reset_index()
            if not d2_ls_q1_data_ls.empty:
                d2_ls_position = d2_ls_q1_data_ls.loc[0, 'position']
        if len(drivers) > 2:
            d3 = drivers.loc[2, 'driverId']
        if len(drivers) > 3:
            d4 = drivers.loc[3, 'driverId']
        q1_data_ls.at[row_ind, 'd1_id'] = d1
        q1_data_ls.at[row_ind, 'd2_id'] = d2
        q1_data_ls.at[row_ind, 'd3_id'] = d3
        q1_data_ls.at[row_ind, 'd4_id'] = d4
        q1_data_ls.at[row_ind, 'ls_d1_rank'] = d1_ls_position
        q1_data_ls.at[row_ind, 'ls_d2_rank'] = d2_ls_position
    # add ls_av_d_rank
    q1_data_ls['ls_av_d_rank'] = (q1_data_ls['ls_d1_rank'] +
                                  q1_data_ls['ls_d2_rank'])/2
    return q1_data_ls


def get_q1_data_cs(q1_data_ls: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes the resulting dataframe from a1_data_ls and
    returns a dataframe which has every constructor of every
    season with variables about the constructor's current season.
    """
    q1_data_cs = q1_data_ls.drop(labels=['position', 'ls_rank', 'ls_points',
                                         'ls_wins', 'ls_d1_rank', 'ls_d2_rank',
                                         'ls_av_d_rank'], axis=1)
    for season in q1_data_cs['season'].unique():
        time.sleep(0.01)
        response = requests.get('http://ergast.com/api/f1/' + str(season) +
                                '.json?limit=1000')
        q1_data_cs.loc[q1_data_cs['season'] == season, 'no_rounds'] =\
            len(response.json()['MRData']['RaceTable']['Races'])

    all_time = pd.DataFrame()
    for season in q1_data_cs['season'].unique():
        season_df = pd.DataFrame()
        no_rounds = int(q1_data_cs.loc[q1_data_cs['season'] == season]
                        ['no_rounds'].reset_index(drop=True)[0])
        for round in range(1, no_rounds + 1):
            time.sleep(0.01)
            response = requests.get('http://ergast.com/api/f1/' + str(season)
                                    + '/' + str(round) +
                                    '/results.json?limit=1000').json()
            no_drivers = len(response['MRData']['RaceTable']['Races'][0]
                             ['Results'])
            temp_df = pd.DataFrame()
            cols = [[], [], [], [], []]
            for driver in range(no_drivers):
                result = response['MRData']['RaceTable']['Races'][0]
                result = result['Results'][driver]
                cols[0].append(result['Constructor']['constructorId'])
                cols[1].append(result['position'])
                if 'FastestLap' in result.keys():
                    cols[2].append(result['FastestLap']['rank'])
                else:
                    cols[2].append(None)
                cols[3].append(result['status'])
                cols[4].append(result['grid'])

            temp_df['constructorId'] = cols[0]
            temp_df['position'] = cols[1]
            temp_df['fl_rank'] = cols[2]
            temp_df['status'] = cols[3]
            temp_df['grid'] = cols[4]
            temp_df['round'] = round
            temp_df = temp_df.astype({'position': 'int'})
            season_df = pd.concat([season_df, temp_df])
        season_df['season'] = season
        all_time = pd.concat([all_time, season_df])

    # Gets Pit Stop Data
    pitstop_data = pd.DataFrame()
    for season in range(2012, 2023):
        season_df = pd.DataFrame()
        no_rounds = int(q1_data_cs
                        .loc[q1_data_cs['season'] == season]['no_rounds']
                        .reset_index(drop=True)[0])
        for round in range(1, no_rounds + 1):
            round_df = pd.DataFrame()
            time.sleep(0.01)
            response = requests.get('http://ergast.com/api/f1/' + str(season) +
                                    '/' + str(round) +
                                    '/pitstops.json?limit=1000').json()
            if len(response['MRData']['RaceTable']['Races']) > 0:
                pitstops = response['MRData']['RaceTable']['Races'][0]
                pitstops = pitstops['PitStops']
                cols = [[], []]
                for pitstop in pitstops:
                    cols[0].append(pitstop['driverId'])
                    cols[1].append(pitstop['duration'])
                round_df['driverId'] = cols[0]
                round_df['duration'] = cols[1]
            round_df['round'] = round
            season_df = pd.concat([season_df, round_df])
        season_df['season'] = season
        pitstop_data = pd.concat([pitstop_data, season_df])

    # add average pitstop times to q1_data_cs
    pitstop_data = pitstop_data.reset_index(drop=True)
    for ind in pitstop_data.index:
        dur = pitstop_data.at[ind, 'duration']
        if isinstance(dur, str) and ':' in dur:
            inter = dur.split(':')
            mins = int(inter[0])
            secs = float(inter[1])
            pitstop_data.at[ind, 'duration'] = (mins * 60) + secs
    pitstop_data = pitstop_data.astype({'duration': 'double', 'season': 'int'})
    pitstop_data['constructorId'] = None
    for ind in pitstop_data.index:
        driver = pitstop_data.at[ind, 'driverId']
        season = pitstop_data.at[ind, 'season']
        constructor = q1_data_cs.loc[(q1_data_cs['season'] == season) &
                                     ((q1_data_cs['d1_id'] == driver) |
                                     (q1_data_cs['d2_id'] == driver) |
                                     (q1_data_cs['d3_id'] == driver) |
                                     (q1_data_cs['d4_id'] == driver)),
                                     'constructorId'].reset_index(drop=True)[0]
        pitstop_data.loc[ind, 'constructorId'] = constructor

    pitstop_bycon = pitstop_data.groupby(['season', 'constructorId'])\
        .agg(av_dur=('duration', 'mean')).reset_index()
    q1_data_cs = pd.merge(pitstop_bycon, q1_data_cs,  how='right',
                          left_on=['season', 'constructorId'],
                          right_on=['season', 'constructorId'])

    # Add average fastest lap ranking to q1_data_cs
    all_time['fl_rank'] = all_time['fl_rank'].astype('Int64')
    avg_fl_const = all_time.groupby(['season', 'constructorId'])\
        .agg(avg_fl=('fl_rank', 'mean')).reset_index()
    q1_data_cs = pd.merge(q1_data_cs, avg_fl_const,  how='left',
                          left_on=['season', 'constructorId'],
                          right_on=['season', 'constructorId'])

    # Add no_notfin to q1_data_cs
    all_time = all_time.reset_index(drop=True)
    all_time['not_fin'] = 1
    for ind in all_time.index:
        if all_time.at[ind, 'status'] == 'Finished':
            all_time.at[ind, 'not_fin'] = 0
    notfin_by_c = all_time.groupby(['season', 'constructorId'])\
        .agg(no_notfin=('not_fin', 'sum')).reset_index()
    q1_data_cs = pd.merge(q1_data_cs, notfin_by_c,  how='left',
                          left_on=['season', 'constructorId'],
                          right_on=['season', 'constructorId'])

    # Add ave_pos
    all_time['pos'] = all_time['position'].astype('Int64')
    ave_pos_by_const = all_time.groupby(['season', 'constructorId'])\
        .agg(av_pos=('position', 'mean')).reset_index()
    q1_data_cs = pd.merge(q1_data_cs, ave_pos_by_const,  how='left',
                          left_on=['season', 'constructorId'],
                          right_on=['season', 'constructorId'])
    q1_data_cs = q1_data_cs.astype({'avg_fl': 'float64'})
    return q1_data_cs


def q1_analysis(q1_data_ls, q1_data_cs) -> None:
    """
    This function takes the resulting dataframes from get_q1_data_ls and
    and get_q1_data_cs and saves plots (which all start with q1_) which
    answer the first question of our project.
    """
    sns.set_theme()
    for i in range(4):
        # create dataframe of the coefficients per year (coeff)
        coeff = pd.DataFrame()
        analys = ''
        if i == 0 or i == 2:
            analys = 'Last Season Variables'
            for season in q1_data_ls['season'].unique():
                data_season = q1_data_ls.loc[q1_data_ls['season'] == season]
                ls_rank = data_season['ls_rank']\
                    .corr(data_season['points']) ** 2
                ls_points = data_season['ls_points']\
                    .corr(data_season['points']) ** 2
                ls_wins = data_season['ls_wins']\
                    .corr(data_season['points']) ** 2
                ls_d1_rank = data_season['ls_d1_rank']\
                    .corr(data_season['points']) ** 2
                ls_d2_rank = data_season['ls_d2_rank']\
                    .corr(data_season['points']) ** 2
                ls_av_d_rank = data_season['ls_av_d_rank']\
                    .corr(data_season['points']) ** 2
                season_coeff = pd.DataFrame({'last season rank': ls_rank,
                                             'last season points': ls_points,
                                             'last season wins': ls_wins,
                                             'driver 1 last season rank':
                                             ls_d1_rank,
                                             'driver 2 last season rank':
                                             ls_d2_rank,
                                             'last season average driver rank':
                                             ls_av_d_rank}, index=[season])
                coeff = pd.concat([coeff, season_coeff])
        elif i == 1 or i == 3:
            analys = 'Current Season Variables'
            for season in q1_data_cs['season'].unique():
                data_season = q1_data_cs.loc[q1_data_cs['season'] == season]
                no_wins = data_season['wins'].corr(data_season['points']) ** 2
                av_dur = data_season['av_dur'].corr(data_season['points']) ** 2
                avg_fl = data_season['avg_fl'].corr(data_season['points']) ** 2
                no_notfin = data_season['no_notfin']\
                    .corr(data_season['points']) ** 2
                av_pos = data_season['av_pos'].corr(data_season['points']) ** 2
                season_coeff = \
                    pd.DataFrame({'number of wins': no_wins,
                                  'average pit stop duration': av_dur,
                                  'average fastest lap ranking': avg_fl,
                                  'number of failed or slow finishes':
                                  no_notfin,
                                  'average race results': av_pos},
                                 index=[season])
                coeff = pd.concat([coeff, season_coeff])

        if i == 2 or i == 3:
            coeff = coeff.loc[2000:2023]
            analys = analys + ', 2000-2022'
        else:
            analys = analys + ', All Time'

        # get the all time coeffelation averages and their absolute values
        av_coeff = pd.DataFrame({'stats': coeff.columns},
                                index=[i for i in range(len(coeff.columns))])
        for row_ind in range(len(av_coeff)):
            av_coeff.at[row_ind, 'mean'] = coeff[coeff.columns[row_ind]].mean()

        # plot the ranked absolute correlation absolute values
        fig, [ax1, ax2] = plt.subplots(2, figsize=(17, 11))
        fig.suptitle('Coefficient of Determination for Points Scored by '
                     'Constructor: ' + analys, fontsize=16, y=.925)
        sns.barplot(data=av_coeff.sort_values(by='mean', ascending=False),
                    x='mean', y='stats', color='b', ax=ax1)
        ax1.set_title('Coefficients of Determination Ranking')
        ax1.set_xlabel('Coefficient of Determination')
        ax1.set_ylabel('Variables')

        # smoothing the variability of the coeffelation per year data (coeff)
        # creating smooth_coeff
        smooth_coeff = coeff.copy()
        for col in smooth_coeff.columns:
            cond = col != 'average fastest lap ranking' and \
                col != 'average pitstop duration'
            if cond:
                smooth_coeff[col] = smooth_coeff[col].fillna(smooth_coeff[col]
                                                             .mean())
                smooth_coeff[col] = gaussian_filter1d(smooth_coeff[col],
                                                      sigma=6)
            elif col == 'average fastest lap ranking':
                smooth_coeff.loc[2004:2023, col] = \
                    gaussian_filter1d(smooth_coeff.loc[2004:2023, col],
                                      sigma=6)
            else:
                smooth_coeff.loc[2012:2023, col] = \
                    gaussian_filter1d(smooth_coeff.loc[2012:2023, col],
                                      sigma=6)

        # formatting smooth_cor so it can be plotted in as a relplot, creating
        # smooth_coeff_rel
        smooth_coeff_rel = pd.DataFrame()
        for year in smooth_coeff.index:
            vars_year = pd.DataFrame()
            vars_year['coefficient'] = smooth_coeff.loc[year]
            vars_year['year'] = year
            vars_year['Variables'] = smooth_coeff.columns
            smooth_coeff_rel = pd.concat([smooth_coeff_rel, vars_year])
        smooth_coeff_rel.index = [i for i in range(len(smooth_coeff_rel))]

        # plotting smoothe version
        hue_order = av_coeff.sort_values(by='mean', ascending=False)['stats']\
            .tolist()
        sns.lineplot(data=smooth_coeff_rel, x='year', y='coefficient',
                     hue='Variables', hue_order=hue_order, ax=ax2)
        ax2.set_title('Coefficients of Determination vs Time')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Coefficient of Determination')
        plt.savefig('q1_' + str(i) + '.png')


def q2(ssns_dstandings: pd.DataFrame) -> None:
    """
    This function takes the resulting dataframe from get_dstandings and
    and saves plots to the working directory which answer the second question.
    All plots are pngs starting with start with 'q2_'
    """
    for i in range(2):
        year_range = 'All Time'
        if i == 1:
            ssns_dstandings = ssns_dstandings\
                .loc[ssns_dstandings['season'] > 1999]
            year_range = '2000-2022'

        # creating drivers dataset d_regions
        d_regions = pd.DataFrame()
        d_regions['av_rank'] = ssns_dstandings\
            .groupby('driverId')['position'].mean()
        d_regions['av_points'] = ssns_dstandings\
            .groupby('driverId')['points'].mean()
        d_regions['years_raced'] = ssns_dstandings\
            .groupby('driverId')['season'].count()
        d_regions['nationality'] = ssns_dstandings\
            .groupby('driverId')['nationality'].first()
        d_regions['wins'] = ssns_dstandings.groupby('driverId')['wins'].sum()
        for driver in d_regions.index:
            d_1sts = ssns_dstandings\
                .loc[(ssns_dstandings['driverId'] == driver) &
                     (ssns_dstandings['position'] == 1)]
            d_regions.loc[driver, 'n_1st'] = len(d_1sts)
        d_regions['points'] = ssns_dstandings.groupby('driverId')['points']\
            .sum()
        d_regions['driverId'] = d_regions.index
        driver_birthdates = ssns_dstandings\
            .groupby('driverId')['dateOfBirth'].first()
        for driver in d_regions.index:
            d_regions.loc[driver, 'YOB'] = driver_birthdates[driver]\
                .split('-')[0]

        if i == 1:
            ssns_dstandings = ssns_dstandings\
                .loc[ssns_dstandings['season'] > 1999]
            year_range = '2000-2022'
        # creating country dataset country_stat with geography data
        # adding all the cols
        country_stats = pd.DataFrame()
        country_stats['n_drivers'] = d_regions\
            .groupby('nationality')['driverId'].count()
        country_stats['n_wins'] = d_regions\
            .groupby('nationality')['wins'].sum()
        country_stats['n_1st'] = d_regions\
            .groupby('nationality')['n_1st'].sum()
        country_stats['n_points'] = d_regions.\
            groupby('nationality')['points'].sum()
        country_stats['av_rank'] = ssns_dstandings\
            .groupby('nationality')['position'].mean()
        country_stats['av_yraced'] = d_regions\
            .groupby('nationality')['years_raced'].mean()
        if i == 0:
            country_stats.drop(labels=['Liechtensteiner', 'Rhodesian',
                                       'American-Italian', 'Argentine-Italian',
                                       'East German'], axis='index',
                               inplace=True)
            country_stats['country'] = ['United States of America',
                                        'Argentina', 'Australia', 'Austria',
                                        'Belgium', 'Brazil', 'United Kingdom',
                                        'Canada', 'Chile', 'China', 'Colombia',
                                        'Czech Republic', 'Denmark',
                                        'Netherlands', 'Finland', 'France',
                                        'Germany', 'Hungary', 'India',
                                        'Indonesia', 'Ireland', 'Italy',
                                        'Japan', 'Malaysia', 'Mexico',
                                        'Monaco', 'New Zealand', 'Poland',
                                        'Portugal', 'Russia', 'South Africa',
                                        'Spain', 'Sweden', 'Switzerland',
                                        'Thailand', 'Uruguay', 'Venezuela']
        elif i == 1:
            country_stats['country'] = ['United States of America',
                                        'Argentina', 'Australia', 'Austria',
                                        'Belgium', 'Brazil', 'United Kingdom',
                                        'Canada', 'China', 'Colombia',
                                        'Czech Republic', 'Denmark',
                                        'Netherlands', 'Finland', 'France',
                                        'Germany', 'Hungary', 'India',
                                        'Indonesia', 'Ireland', 'Italy',
                                        'Japan', 'Malaysia', 'Mexico',
                                        'Monaco', 'New Zealand', 'Poland',
                                        'Portugal', 'Russia', 'Spain',
                                        'Sweden', 'Switzerland', 'Thailand',
                                        'Venezuela']
        # merging with country geometry dataset
        countries = gpd.read_file('countries.geojson')
        countries.drop(labels='ISO_A3', inplace=True, axis='columns')
        country_stats = countries\
            .merge(country_stats, left_on='ADMIN',
                   right_on='country').drop(labels='ADMIN', axis='columns')
        # creating seperate dataset with only countries that won don't have
        # zero as a statistic
        country_stats_champ = country_stats.loc[country_stats['n_1st'] > 0]
        country_stats_won = country_stats.loc[country_stats['n_wins'] > 0]
        country_stats_points = country_stats.loc[country_stats['n_points'] > 0]

        # plotting
        sns.reset_orig()
        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))
        countries.plot(ax=ax1, color='#EEEEEE')
        countries.plot(ax=ax2, color='#EEEEEE')
        countries.plot(ax=ax3, color='#EEEEEE')
        countries.plot(ax=ax4, color='#EEEEEE')
        country_stats_won.plot(ax=ax1, column='n_wins', legend=True)
        country_stats_champ.plot(ax=ax2, column='n_1st', legend=True)
        country_stats_points.plot(ax=ax3, column='n_points', legend=True)
        country_stats.plot(ax=ax4, column='av_rank', legend=True)
        fig.suptitle('Driver Performance By Country: ' + year_range,
                     fontsize=16, y=.925)
        ax1.set_title('Number of Driver Grand Prix Wins')
        ax2.set_title('Number of Driver Championships Won')
        ax3.set_title('Number of Driver Points Won')
        ax4.set_title('Average Driver Season Rank')
        plt.savefig('q2_1_' + str(i) + '.png')

        # zoom into europe
        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))
        xlim = ([-12, 40])
        ylim = ([35, 75])
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        ax4.set_xlim(xlim)
        ax4.set_ylim(ylim)
        countries.plot(ax=ax1, color='#EEEEEE')
        countries.plot(ax=ax2, color='#EEEEEE')
        countries.plot(ax=ax3, color='#EEEEEE')
        countries.plot(ax=ax4, color='#EEEEEE')
        country_stats_won.plot(ax=ax1, column='n_wins', legend=True)
        country_stats_champ.plot(ax=ax2, column='n_1st', legend=True)
        country_stats_points.plot(ax=ax3, column='n_points', legend=True)
        country_stats.plot(ax=ax4, column='av_rank', legend=True)
        fig.suptitle('Driver Performance By Country: ' + year_range +
                     ' (Europe)', fontsize=16, y=.925)
        ax1.set_title('Number of Driver Grand Prix Wins')
        ax2.set_title('Number of Driver Championships Won')
        ax3.set_title('Number of Driver Points Won')
        ax4.set_title('Average Driver Season Rank')
        plt.savefig('q2_2_' + str(i) + '.png')

        # All countries number of drivers heatmap
        fig, ax = plt.subplots(1, figsize=(20, 10))
        countries.plot(ax=ax, color='#EEEEEE')
        country_stats.plot(ax=ax, column='n_drivers', legend=True)
        ax.set_title('Number of Drivers: ' + year_range)
        plt.savefig('q2_3_' + str(i) + '.png')

        # All countries number of drivers ranked
        sns.set_theme()
        sns.catplot(data=country_stats
                    .sort_values('n_drivers', ascending=False), x='country',
                    y='n_drivers', kind='bar', height=5, aspect=3, color='b')
        plt.title('Country Representation: ' + year_range)
        plt.ylabel('Number of Drivers')
        plt.xlabel('Country')
        plt.xticks(rotation=90)
        plt.savefig('q2_4_' + str(i) + '.png')


def extract_race_data(year) -> pd.DataFrame:
    """
    This function extracts race data for a single given year, returning the
    data as a pandas DataFrame. The resulting DataFrame has the qualifying
    results and race results for every driver of every race of that year, with
    the track information.
    """
    time.sleep(0.01)
    response = requests.get('http://ergast.com/api/f1/' + str(year) +
                            '/results.json?limit=1000')
    json_data = response.json()
    race_data = json_data['MRData']['RaceTable']['Races']
    df = pd.DataFrame()
    for race in race_data:
        course = race['Circuit']['circuitId']
        for result in race['Results']:
            driver = result['Driver']['driverId']
            qualifying = result['grid']
            race_result = result['position']
            df = pd.concat([df, pd.DataFrame({'year': year, 'driver': driver,
                                              'course': course, 'qualifying':
                                              qualifying,
                                              'race_result': race_result},
                                             index=[0])])
    df = df.astype({'qualifying': 'int64', 'year': 'int64',
                                  'race_result': 'int64'})
    return df


def q3_prep() -> tuple(pd.DataFrame()):
    """
    This function creates a tuple with two dataframes, qualifying results and
    race results for every driver of every race, with the track information.
    The first has this information only for 2022, while the second has this
    information for years 2000-2022 (inclusive). These dataframes are the data
    that is then analyzed in q3_analysis to answer the third question of our
    project.
    """
    data_2022 = extract_race_data(2022)
    post_1999 = pd.DataFrame()
    for year in range(2000, 2023):
        one_year = extract_race_data(year)
        post_1999 = pd.concat([post_1999, one_year])
    data_2022.reset_index(drop=True, inplace=True)
    post_1999.reset_index(drop=True, inplace=True)
    return (data_2022, post_1999)


def q3_analysis(data_2022, post_1999) -> None:
    """
    This function takes the dataframes produced by q3_prep and creates a plot
    which helps answers our third question. It also calculates the MSE values
    of the models we are assessing for our third question, which are printed
    in the terminal. The resulting plot is saved to the working directory as
    'q3_1.png'. The MSEs are printed in this order: top-left model on the
    plot, top-right model on the plot, bottom-left model on the plot,
    bottom-right model on the plot.
    """
    results = []
    for i in range(4):
        df_filtered = pd.DataFrame()
        if i == 0:
            df_filtered = data_2022.drop(['year', 'driver', 'course'], axis=1)
        elif i == 1:
            df_filtered = post_1999.drop(['year', 'driver', 'course'], axis=1)
        elif i == 2:
            df_filtered = data_2022
        elif i == 3:
            df_filtered = post_1999

        # Dropping rows with missing values in qualifying column
        df_filtered = df_filtered.dropna()

        # Split the dataset into training and testing sets
        X = df_filtered.drop('race_result', axis=1)
        if i == 2 or i == 3:
            X = pd.get_dummies(df_filtered.drop('race_result', axis=1))
        y = df_filtered['race_result']
        X_train, X_test, y_train, y_test =\
            train_test_split(X, y, test_size=0.15, random_state=42)

        # Training a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Testing the model and storing predictions
        y_pred = model.predict(X_test)

        # Calculating mean squared error
        mse = mean_squared_error(y_test, y_pred)

        # preparing results and predictions for plotting
        y_pred = y_pred.astype(np.int64, copy=False).tolist()
        y_test = y_test.astype("int").tolist()
        plot_df = pd.DataFrame(data={'predictions': y_pred,
                                     'true_values': y_test})
        results.append((plot_df, mse))

    # Plotting
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Model Predicted Race Results vs. Actual Race Results Using '
                 'Various Features and Data', fontsize=16, y=.94)
    sns.regplot(data=results[0][0], x='true_values', y='predictions',
                scatter_kws={'alpha': 1/5}, x_jitter=.2, y_jitter=.2,
                fit_reg=False, ax=ax1)
    sns.regplot(data=results[1][0], x='true_values', y='predictions',
                scatter_kws={'alpha': 1/5}, x_jitter=.2, y_jitter=.2,
                fit_reg=False, ax=ax2)
    sns.regplot(data=results[2][0], x='true_values', y='predictions',
                scatter_kws={'alpha': 1/5}, x_jitter=.2, y_jitter=.2,
                fit_reg=False, ax=ax3)
    sns.regplot(data=results[3][0], x='true_values', y='predictions',
                scatter_kws={'alpha': 1/5}, x_jitter=.2, y_jitter=.2,
                fit_reg=False, ax=ax4)
    ax1.plot([0, max(y_test)], [0, max(y_test)], color='red')
    ax2.plot([0, max(y_test)], [0, max(y_test)], color='red')
    ax3.plot([0, max(y_test)], [0, max(y_test)], color='red')
    ax4.plot([0, max(y_test)], [0, max(y_test)], color='red')
    ax1.set_xlabel('True Race Results')
    ax2.set_xlabel('True Race Results')
    ax3.set_xlabel('True Race Results')
    ax4.set_xlabel('True Race Results')
    ax1.set_ylabel('Predicted Race Results')
    ax2.set_ylabel('Predicted Race Results')
    ax3.set_ylabel('Predicted Race Results')
    ax4.set_ylabel('Predicted Race Results')
    ax1.set_title('Features: Qualifying Results | Data: 2022')
    ax3.set_title('Features: Qualifying Results, '
                  'Course, Driver, Season | Data: 2022')
    ax2.set_title('Features: Qualifying Results | Data: 2000-2022')
    ax4.set_title('Features: Qualifying Results, '
                  'Course, Driver, Season | Data: 2000-2022')
    plt.savefig('q3_1')
    for result in results:
        print('MSE:', result[1])


def main():
    print('making necessary datasets...')
    cstandings = get_cstandings()
    dstandings = get_dstandings()
    print('q1 prep part 1...')
    q1_data_ls = get_q1_data_ls(cstandings, dstandings)
    print('q1 prep part 2...')
    q1_data_cs = get_q1_data_cs(q1_data_ls)
    print('q1 analysis...')
    q1_analysis(q1_data_ls, q1_data_cs)
    print('q2...')
    q2(dstandings)
    print('q3 prep...')
    data_2022, post_1999 = q3_prep()
    print('q3 analysis...')
    q3_analysis(data_2022, post_1999)
    # saving all resulting datasets for testing.
    cstandings.to_csv('cstandings.csv')
    dstandings.to_csv('dstandings.csv')
    q1_data_ls.to_csv('q1_data_ls.csv')
    q1_data_cs.to_csv('q1_data_cs.csv')
    data_2022.to_csv('data_2022.csv')
    post_1999.to_csv('post_1999.csv')


if __name__ == '__main__':
    main()
