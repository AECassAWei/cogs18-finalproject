"""Test for my functions."""

import pandas as pd             # dataframe
import numpy as np              # algebra
from scipy import stats         # test statistics

# my own functions
from my_module.functions import read_csv             # select/rename columns
from my_module.functions import pivot_groups         # pivot after group by
from my_module.functions import clean_text           # clean tweets
from my_module.functions import sentiment_analysis   # sentiment analysis
from my_module.functions import stat_test            # statistical test

# my class for print message suppression
from my_module.classes import SuppressPrintStatement # used to suppress print 


### ========================== Preparation Codes ======================= ###

# columns to use, and renames of selected columns
usecols = ['created_at', 'tweet', 'country', 'state']
renames = ['Date', 'Tweet', 'Country', 'State']

# read hashtag-trump & hashtag-biden csv as dataframe
trump = read_csv('./dataset/hashtag_donaldtrump.csv', usecols, renames)
biden = read_csv('./dataset/hashtag_joebiden.csv', usecols, renames)

# add category to each dataframe
trump['Hashtag'] = 'Trump'
biden['Hashtag'] = 'Biden'

# concatenate trump & biden dataframe together
election_tweet = pd.concat([trump, biden])

# get US election tweets only
us_election_tweet = election_tweet[election_tweet['Country'] == 'United States of America']

# just analyzing 50 states + D.C.
isnas = pd.isna(us_election_tweet['State'])                              # find nan entries
deleted = [state in ['Puerto Rico', 'Guam', 'Northern Mariana Islands']  # delete non-50 states
           for state in us_election_tweet['State']]

selected = [not (isna or delete) for isna,delete in zip(isnas, deleted)] # criterion selected
us_election_tweet_selected = (us_election_tweet[selected]                # select states
                              .reset_index(drop=True))                   # reset index

# count the number of tweets based on 'Date', and 'Hashtag'
counts_timeseries = pivot_groups(us_election_tweet_selected, 'Date', 'Hashtag')

# count the number of tweets based on 'State', and 'Hashtag'
counts_by_state = pivot_groups(us_election_tweet_selected, 'State', 'Hashtag', reset = True)

with SuppressPrintStatement():
    # paired t-test to test whether Trump & Biden 'Tweet' counts are significantly different across 'State'
    paired_name = 'Paired T-Test      : Difference between #Trump & #Biden Tweet Counts'
    paired_tstat, paired_pval = stat_test(counts_by_state['Trump'],  # collection 1: Trump 'Tweet' counts
                                          counts_by_state['Biden'],  # collection 2: Biden 'Tweet' counts
                                          paired_name,               # print out name
                                          test = stats.ttest_rel)    # test type: paired t-test

### ========================== Preparation Codes ======================= ###


### ========================== Testing Functions ======================= ###
def test_read_csv():
    
    # is a function
    assert callable(read_csv)
    
    # check shape of dataframe
    assert trump.shape == (970919, 5)
    assert biden.shape == (776886, 5)
    
    # check if renamed column
    assert 'Date' in trump.columns
    assert 'State' in biden.columns
    
    print('==========       read_csv() tests all pass        ==========')


def test_pivot_groups():
    
    # is a function
    assert callable(pivot_groups)
    
    # check shape of dataframe
    assert counts_timeseries.shape == (25, 2)
    assert counts_by_state.shape == (51, 3)
    
    # check values
    check1 = counts_by_state.mean()
    assert check1[0] > 3001 and check1[0] < 3002
    assert check1[1] > 3497 and check1[1] < 3498
    
    check2 = counts_timeseries.mean()
    assert check2[0] > 6123 and check2[0] < 6124
    assert check2[1] > 7134 and check2[1] < 7135
    
    print('==========     pivot_groups() tests all pass      ==========')


def test_clean_text():
    
    # is a function
    assert callable(clean_text)
    
    # check cleaned value
    assert clean_text(' ') == ''
    assert clean_text('I am ALaric?') == 'i am alaric'
    assert clean_text('https://something Haha') == 'haha'
    assert clean_text('\n get rid .of stuff') == 'get rid of stuff'
    
    print('==========      clean_text() tests all pass       ==========')


def test_sentiment_analysis():
    
    # is a function
    assert callable(sentiment_analysis)
    
    # check sentiment output
    assert sentiment_analysis('') == (0.0, 'NEUTRAL')
    assert sentiment_analysis('I am very happy') == (1.0, 'POSITIVE')
    assert sentiment_analysis('I am very sad') == (-0.65, 'NEGATIVE')
    
    print('==========  sentiment_analysis() tests all pass   ==========')

    
def test_stat_test():
    
    # count the number of tweets based on 'State', and 'Hashtag'
    # is a function
    assert callable(stat_test)
    
    # check test values
    assert paired_tstat > 3.4418 and paired_tstat < 3.4419
    assert paired_pval > 0.0011 and paired_pval < 0.0012
    
    print('==========      stat_test() tests all pass        ==========')

### ========================== Testing Functions ======================= ###
