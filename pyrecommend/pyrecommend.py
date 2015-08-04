# -*- coding: utf-8 -*-

# simple recommendation engine -- by misterte
# NOT available for 3rd party use under any licence
###################################################
# Methods may return a maximum of 100 predicted items.
MAX_PREDICTIONS_PER_CALL = 100
class SimpleRecommendationEngine(object):
    """
    User & item based recommender engine, based on the pearson correlation similarity. Takes as 
    input a pandas.DataFrame with item preferences for users, as a pref[user, item] = (int|NaN) matrix.
    This is a simple implementation of the example in Chapter 2 of Programming Collective Intelligence, by 
    Toby Segaran (a great read!!).
    The engine will try to use pearson correlation straight over the provided dataframe. If std is zero for 
    a particular row, it will try the same but filling out NaN values with zeroes (i.e., dataframe.fillna(0)).
    
    Use
    ---
    >>> engine = SimpleRecommendationEngine(df)
    >>> simusers = engine.similar_users('Lisa Rose', n=4)    # default is n=5 for every method
    >>> # this will return a pandas series, but you can easily turn that into a tuple list
    >>> zip(simusers.index, simusers.values)
    [('Toby', 0.99124070716193036),
     ('Jack Matthews', 0.74701788083399601),
     ('Mick LaSalle', 0.59408852578600457),
     ('Claudia Puig', 0.56694670951384085)]
    >>> recitems = engine.recommend_items('Toby')
    >>> # again, recitems will be a pandas series but easily turned into a tuple list
    >>> zip(recitems.index, recitems.values)
    # in this case there are only 3 possible recommendations
    [('The Night Listener', 3.3477895267131013),
     ('Lady in the Water', 2.8325499182641618),
     ('Just My Luck', 2.5309807037655645)]
    >>> # you can access equivalent methods for the transposed dataframe
    >>> simitems = engine.similar_items('The Night Listener'); zip(simitems.index, simitems.values)
    [('You, Me and Dupree', 0.65795169495976902),
     ('Lady in the Water', 0.48795003647426655),
     ('Snakes on a Plane', 0.11180339887498948),
     ('The Night Listener', -0.1798471947990542),
     ('Just My Luck', -0.42289003161103106)]
    >>> recusers = engine.recommend_users('The Night Listener'); zip(recusers.index, recusers.values)
    [('Michael Phillips', 4.0), ('Jack Matthews', 3.0)]
    """
    # Default max predictions to return per method.
    MAX_N = MAX_PREDICTIONS_PER_CALL
    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # precompute and cache similarity matrices for dataframe and dataframe.transpose
        self._compute_similarity_matrices()
    
    def get_row_ids(self):
        """
        Returns all row_ids in current dataframe (row_id = index) as a list.
        """
        return list(self.dataframe.index)
    
    def get_column_ids(self):
        """
        Returns all column_ids in current dataframe (column_id = columns) as a list.
        """
        return list(self.dataframe.columns)
    
    def _compute_similarity_matrices(self):
        """
        Computes dataframe|dataframe.transpose pearson|pearson_binary correlation matrices for dataframe and 
        dataframe.fillna(0).
        """
        self._similarity_matrices = {
            # transpose=True is used for column based similarities & row based recommendations
            True: {
                'pearson': self.dataframe.corr(),
                'pearson_binary': self.dataframe.fillna(0).corr()
            },
            False: {
                'pearson': self.dataframe.transpose().corr(),
                'pearson_binary': self.dataframe.fillna(0).transpose().corr()
            }
        }
    
    def _get_similarity_matrix(self, transpose=False, method='pearson'):
        """
        Returns pearson|pearson_binary correlation matrix for transposed|non-transposed dataframe. 
        """
        return self._similarity_matrices[transpose][method]
    
    def _similar_rows(self, transpose, row_id, n, only_positive_corr):
        """
        Returns a descending pd.Series with the `n` most similar rows and it's rank. Note that negative 
        rank could be interpreted as rows (users, for instance) that have very dissimilar tastes. If 
        `row_id` is not in the dataframe, this method will raise a KeyError. The method will first try 
        to build the series using the pearson correlation as the similarity matrix, but if it does not 
        retrieve any suggestions, it will move on to using the pearson_binary approach. If there are no 
        similarities found, it will return an empty series.
        By using `only_positive_corr` we can define if we want negative ranks to be returned too.
        This method is wrapped by self.similar_users and self.similar_items.
        """
        df = self.dataframe.transpose() if transpose else self.dataframe
        for method in ['pearson', 'pearson_binary']:
            sim = self._get_similarity_matrix(transpose=transpose, method=method)
            x = sim.loc[row_id, [i for i in df.index if i != row_id]]
            # Filter out NaN values. If we only found NaN values, then we 
            # proceed to use the pearson_binary method
            x = x[x.notnull()]
            if not x.empty:
                break
        # sort and check if only positive correlations were requested
        x.sort(ascending=False)
        n = min(n, self.MAX_N)
        if only_positive_corr:
            return x[x>0][:n]
        return x[:n]
    
    def _recommend_columns(self, transpose, row_id, n):
        """
        Returns a descending pd.Series with the `n` best ranked column_ids that the `row_id` has not 
        rated--i.e., NaN, as zero is considered a score. It will raise a KeyError if `row_id` isn't found 
        in the dataframe.
        This method is wrapped by self.recommend_items and self.recommend_users.
        """
        df = self.dataframe.transpose() if transpose else self.dataframe
        for method in ['pearson', 'pearson_binary']:
            sim = self._get_similarity_matrix(transpose=transpose, method=method)
            x = sim.loc[row_id, [i for i in df.index if i != row_id]]
            # First get positively correlated row_ids, also filter NaN values. 
            # If we only get NaN values, then we proceed to use the pearson_binary method.
            x = x[(x.notnull()) & (x>0)]
            if not x.empty:
                break
        # Take unrated columns and use them to select rates by row
        unrated = df.loc[row_id,:][df.loc[row_id,:].isnull()].index
        # Now we need to apply two things:
        # 1. multiply each column rating by row's corr with `row_id` (this is the value of `x`)
        # 2. divide each new rating by the sum of row corrs with `row_id` for rows that HAVE rated that column
        # Then we just sum column values and get weighted ratings for recommended columns
        # In order to make this happen, we first need to get all ratings for positively correlated rows and unrated columns
        # so if this df is empty, there can be no recommendations at all, as there is no info to relate this row to others
        if df.loc[x.index, unrated].empty:
            # No possible recommendations => return an empty series
            return pd.Series([])
        # Else we can now process ratings and correlations
        # This part of the process depends on how similarities were computed
        # If we are using pearson, then the similarity matrix is calculated with NaN and int values
        # and the process is pretty straight forward.
        # But if either pearson_binary or euclidean_distance is used, then using the "standard" way
        # will generate a score of 1 for all unrated columns, which is not so useful.
        # 'method' will store the method by which 'sim' was computed in the for loop
        if method == 'pearson':
            y = df.loc[x.index, unrated].apply(lambda col: np.asarray(col)*np.asarray(x) / x[col.notnull()].sum()).sum()
        else:
            # pearson_binary
            # In this case "amount of votes" is what makes a column important, not the value of the vote itself,
            # which will always be 1. So we don't need to "normalize" added scores for the amount of viewers by
            # dividing weighted scores by the sum of weights.
            y = df.loc[x.index, unrated].apply(lambda col: np.asarray(col)*np.asarray(x)).sum()
        # Sort and filter out NaN values
        # These possible NaN values would be product of columns that have not been rated at all, 
        # which makes them elegible for row_id. This could generate a recommendation with NaN as score.
        y = y[y.notnull()]
        y.sort(ascending=False)
        n = min(n, self.MAX_N)
        return y[:n]
    
    # TODO: ADD API METHOD FOR THIS.
    # TODO: ADD THIS METHOD TO CACHE.
    # TODO: ADD RECOMMEND_USERS FOR ITEMS METHOD.
    def similar_users(self, user_id, n=10, only_positive_corr=False):
        """
        This method wraps self._similar_rows so we can use it with users as rows. In this case, we *do not* 
        transpose self.dataframe. Returns a pd.Series of `n` most similar users. By using `only_positive_corr` 
        we can filter out dissimilar users. Method raises KeyError if `user_id` is not found in 
        self.dataframe.index. Defaults to n=10 and only_positive_corr=False.
        """
        return self._similar_rows(transpose=False, row_id=user_id, n=n, only_positive_corr=only_positive_corr)
    
    def recommend_items(self, user_id, n=10):
        """
        This method wraps self._recommend_columns so we can use it with users as rows. In this case, we 
        *do not* transpose self.dataframe. Returns a pd.Series of `n` best ranked unrated items for `user_id`. 
        Raises KeyError if `user_id` is not found in self.dataframe.index. Default n=10.
        """
        return self._recommend_columns(transpose=False, row_id=user_id, n=n)
    
    def similar_items(self, item_id, n=10, only_positive_corr=False):
        """
        This method wraps self._similar_rows so we can use it with items as rows. In this case, we 
        *do* transpose self.dataframe. Returns a pd.Series of `n` most similar items. By using 
        `only_positive_corr` we can filter out dissimilar items. Method raises KeyError if `item_id` is 
        not found in self.dataframe.transpose().index. Defaults to n=10 and only_positive_corr=False.
        """
        return self._similar_rows(transpose=True, row_id=item_id, n=n, only_positive_corr=only_positive_corr)
    
    def recommend_users(self, item_id, n=10):
        """
        This method wraps self._recommend_columns so we can use it with items as rows. In this case, we 
        *do* transpose self.dataframe. Returns a pd.Series of `n` best ranked users that have not rated 
        `item_id` yet. Raises KeyError if `user_id` is not found in self.dataframe.transpose().index. 
        Default n=10.
        """
        return self._recommend_columns(transpose=True, row_id=item_id, n=n)