"""A collection of function for my project."""

import string                   # string functions
import pandas as pd             # dataframe
import numpy as np              # algebra

from textblob import TextBlob   # sentiment analysis


def read_csv(path, usecols, renames):
    """
    Read csv files from path by selected columns,
    and rename the columns.
    
    Parameters
    ----------
    path : string
        file path to read csv.
    usecols : list
        a list of columns to select.
    renames : list
        a list of renames of columns.

    Returns
    -------
    df: pd.DataFrame
        a dataframe with selected, renamed columns.
    """
    
    # read in 'path' csv file
    df = pd.read_csv(path, lineterminator = '\n',   # get rid of '\n'
                     parse_dates = ['created_at'],  # parse 'created_at' as date object
                     usecols = usecols)             # select columns
    
    # keep only the date, get rid of time
    df['created_at'] = df['created_at'].dt.date
    
    # rename columns
    df = df.rename(columns=dict(zip(usecols, renames)))
    
    return df


def pivot_groups(df, by_col1, by_col2, select='Tweet', reset=False):
    """
    Group df by col1 and col2, then pivot the output such that
    col1 is the index, col2 is the column, and select is the value.
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe to groupby and pivot.
    by_col1 : string
        column name to groupby and set as index.
    by_col2 : string
        column name to groupby and set as column.
    select : string
        column name to set as value (default is 'Tweet')
    reset : boolean
        whether to reset index or not (default is False)

    Returns
    -------
    output: pd.DataFrame
        a dataframe grouped, pivoted rows and columns;
        if reset=True, index would also be a column.
    """
    # group by col1 and col2
    counts = df.groupby([by_col1, by_col2],
                         as_index = False).count()
    
    # pivot to index=col1, columns=col2, and values=select
    counts_pivot = counts.pivot(index = by_col1,    # pivot, col1 as index
                                columns = by_col2,  # col2 as columns
                                values = select)    # select counts as values
    
    # check if needs to reset index
    if reset:
        output = counts_pivot.reset_index(drop = False)
    else:
        output = counts_pivot
    
    return output


def clean_text(text):
    """
    Clean Tweets by lower cases, and remove punctuations, 
    new line characters ('\n'), and links.
    
    Parameters
    ----------
    text : string
        tweet to clean.

    Returns
    -------
    cleaned: string
        cleaned tweet.
    """
    # convert to lower cases
    cleaned = text.lower()
    
    # remove punctuations
    cleaned = [ch for ch in cleaned if not (ch in string.punctuation)]
    
    # remove '\n' characters
    cleaned = [' ' if ch in ['\n'] else ch for ch in cleaned]
    
    # convert characters back to string
    cleaned = ''.join(cleaned)
    
    # remove links
    cleaned = cleaned.split() # split into grams of words
    cleaned = [grams for grams in cleaned if 'http' not in grams]
    
    # convert splitted words back to string again
    cleaned = ' '.join(cleaned)
    
    return cleaned


def sentiment_analysis(text):
    """
    Analyze the sentiment of text.
    
    Parameters
    ----------
    text : string
        tweet to analyze sentiment.

    Returns
    -------
    polarity : float
        value of polarity measure.
    sentiment : {'POSITIVE', 'NEGATIVE', 'NEUTRAL'}
        classification of sentiments.
    """
    blob = TextBlob(text)       # create TextBolb object
    polarity = blob.polarity    # polarity of text
    
    if polarity > 0:            # positive sentiment
        sentiment = 'POSITIVE'
    elif polarity < 0:          # negative sentiment
        sentiment = 'NEGATIVE'
    else:                       # neutral sentiment
        sentiment = 'NEUTRAL'
    
    return polarity, sentiment


def stat_test(collection1, collection2, name, test):
    """
    Statistical test that outputs T-statistics, and p-values
    based on the type of test specified.
    
    Parameters
    ----------
    collection1 : list, np.array, pd.Series
        first set of datapoints.
    collection2 : list, np.array, pd.Series
        second set of datapoints.
    test : function
        statistical test (default is independent
        two sample t-test : stats.ttest_ind).
    name : string
        print out title.

    Returns
    -------
    tstat : float
        T-statistic associated with the relevant test.
    pval : float
        p-value associated with the relevant test.
    """
    ttest = test(collection1, collection2)  # test for difference
    tstat = ttest.statistic                 # T-statistics
    pval = ttest.pvalue                     # p-value
    
    # print out
    print(name)
    print('T-Statistic        : ' + str(np.round(tstat, 4)))
    print('P-value            : ' + str(np.round(pval, 4)))
    
    return tstat, pval
