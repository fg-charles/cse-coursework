"""
Ethan Matthew Hsu, Aryan Kartik Shah, Charles Georges Faisandier
CSE 163AA
This python scripts tests the resulting datasets created by main.py. main.py
must be run before running these tests, else an error will occur. the dataset
data_2022 is not tested because it is a subset of the dataset post_1999. All
other datasets are tested.
"""

import pandas as pd
from cse163_utils import assert_equals


def test_cstandings(cstandings):
    """
    tests the cstandings dataset.
    """
    assert_equals('mercedes', cstandings.loc[(cstandings['season'] == 2021) &
                                             (cstandings['position'] == 1),
                                             'constructorId']
                                        .reset_index(drop=True)[0])
    assert_equals('renault', cstandings.loc[(cstandings['season'] == 2005) &
                                            (cstandings['position'] == 1),
                                            'constructorId']
                                       .reset_index(drop=True)[0])
    assert_equals(17, cstandings.loc[(cstandings['season'] == 2022) &
                                     (cstandings['position'] == 1), 'wins']
                                .reset_index(drop=True)[0])
    assert_equals(759, cstandings.loc[(cstandings['season'] == 2022) &
                                      (cstandings['position'] == 1), 'points']
                                 .reset_index(drop=True)[0])


def test_dstandings(dstandings):
    """
    tests the dstandings dataset.
    """
    assert_equals('hulme', dstandings.loc[(dstandings['season'] == 1967) &
                                          (dstandings['position'] == 1),
                                          'driverId']
                                     .reset_index(drop=True)[0])
    assert_equals('max_verstappen',
                  dstandings.loc[(dstandings['season'] == 2021) &
                                 (dstandings['position'] == 1), 'driverId']
                            .reset_index(drop=True)[0])
    assert_equals('max_verstappen',
                  dstandings.loc[(dstandings['season'] == 2022) &
                                 (dstandings['points'] == 454), 'driverId']
                            .reset_index(drop=True)[0])
    assert_equals(454,
                  dstandings.loc[(dstandings['season'] == 2022) &
                                 (dstandings['driverId'] == 'max_verstappen'),
                                 'points'].reset_index(drop=True)[0])


def test_q1_data_ls(q1_data_ls):
    """
    tests the q1_data_ls dataset.
    """
    assert_equals(1, q1_data_ls.loc[(q1_data_ls['season'] == 1958) &
                                    (q1_data_ls['wins'] == 6), 'position']
                               .reset_index(drop=True)[0])
    assert_equals('cade',
                  q1_data_ls.loc[(q1_data_ls['season'] == 1959) &
                                 (q1_data_ls['constructorId'] == 'maserati'),
                                 'd4_id'].reset_index(drop=True)[0])
    assert_equals('leclerc',
                  q1_data_ls.loc[(q1_data_ls['season'] == 2020) &
                                 (q1_data_ls['constructorId'] == 'ferrari'),
                                 'd1_id'].reset_index(drop=True)[0])
    assert_equals(759, q1_data_ls.loc[(q1_data_ls['season'] == 2022) &
                                      (q1_data_ls['position'] == 1),
                                      'points'].reset_index(drop=True)[0])


def test_q1_data_cs(q1_data_cs):
    """
    tests the q1_data_cs dataset.
    """
    position = [1, 2, 2, 3, 1, 20, 6, 20, 2, 3, 4, 20, 2, 4, 19, 20, 2, 5, 1,
                4, 1, 18, 5, 19, 4, 6, 3, 6, 3, 8, 2, 4, 2, 3, 3, 19, 3, 20,
                5, 6, 3, 4, 2, 4]
    fl_rank = ['1', '3', '1', '3', '1', None, '2', None, '2', '3', '7', '13',
               '7', '6', '6', '19', '1', '3', '2', '4', '2', '4', '1', '7',
               '4', '2', '7', '2', '6', '7', '2', '4', '4', '3', '3', None,
               '4', None, '6', '10', '3', '6', '10', '3']
    ferrari_2022 = pd.DataFrame({'position': position, 'fl_rank': fl_rank})
    ferrari_2022 = ferrari_2022.astype({'fl_rank': 'Int64'})
    assert_equals(9, q1_data_cs.loc[(q1_data_cs['season'] == 1962) &
                                    (q1_data_cs['constructorId'] == 'ferrari'),
                                    'no_rounds'].reset_index(drop=True)[0])
    assert_equals(18, q1_data_cs.loc[(q1_data_cs['season'] == 1958) &
                                     (q1_data_cs['wins'] == 6), 'no_notfin']
                                .reset_index(drop=True)[0])
    assert_equals(ferrari_2022['position'].mean(),
                  q1_data_cs.loc[(q1_data_cs['season'] == 2022) &
                                 (q1_data_cs['constructorId'] == 'ferrari'),
                                 'av_pos'].reset_index(drop=True)[0])
    assert_equals(ferrari_2022['fl_rank'].mean(),
                  q1_data_cs.loc[(q1_data_cs['season'] == 2022) &
                                 (q1_data_cs['constructorId'] == 'ferrari'),
                                 'avg_fl'].reset_index(drop=True)[0])


def test_post_1999(post_1999):
    """
    tests the post_1999 dataset.
    """
    assert_equals(3, post_1999
                  .loc[(post_1999['year'] == 2000) &
                       (post_1999['course'] == 'albert_park') &
                       (post_1999['driver'] == 'michael_schumacher'),
                       'qualifying'].reset_index(drop=True)[0])
    assert_equals(1, post_1999.loc[(post_1999['year'] == 2000) &
                                   (post_1999['course'] == 'albert_park') &
                                   (post_1999['driver'] ==
                                    'michael_schumacher'), 'race_result']
                              .reset_index(drop=True)[0])
    assert_equals(7, post_1999.loc[(post_1999['year'] == 2015) &
                                   (post_1999['course'] == 'albert_park') &
                                   (post_1999['driver'] == 'sainz'),
                                   'qualifying'].reset_index(drop=True)[0])
    assert_equals(9, post_1999.loc[(post_1999['year'] == 2015) &
                                   (post_1999['course'] == 'albert_park') &
                                   (post_1999['driver'] == 'sainz'),
                                   'race_result'].reset_index(drop=True)[0])


def main():
    cstandings = pd.read_csv('cstandings.csv')
    dstandings = pd.read_csv('dstandings.csv')
    q1_data_ls = pd.read_csv('q1_data_ls.csv')
    q1_data_cs = pd.read_csv('q1_data_cs.csv')
    post_1999 = pd.read_csv('post_1999.csv')
    test_cstandings(cstandings)
    test_dstandings(dstandings)
    test_q1_data_ls(q1_data_ls)
    test_q1_data_cs(q1_data_cs)
    test_post_1999(post_1999)
    print('All tests passed. No errors found.')


if __name__ == '__main__':
    main()
