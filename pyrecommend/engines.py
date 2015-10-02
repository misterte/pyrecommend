import sys
import pandas as pd
import numpy as np
import time
from pyrecommend.cache import memoize, Memoized




# constants
ITEM_BASED, USER_BASED = 'ITEM_BASED', 'USER_BASED'
INDEX_BASED, COLUMN_BASED = 'INDEX_BASED', 'COLUMN_BASED'
BINARY, RATING_ZERO, RATING_NAN = 'BINARY', 'RATING_ZERO', 'RATING_NAN'


class SimplePearsonEngine(Memoized):
    """
    Simple and easy to use implementation of a Pearson similarity based user/item recommendation engine. 
    This package requires numpy, pandas and sqlitedict. Sqlite is used for results cache, but you can 
    provide any backend of your choice in the memoize_backend kwarg (e.g., memoize_backend=dict()).
    
        preferences (required)
        kind (required)
        memoize_backend (optional)
    
    Sample use:
    >>> import pyrecommend as pyr
    >>> engine = pyr.SimplePearsonEngine(preferences=df, kind=pyr.BINARY)
    >>> engine.train()  # may take a while, but will cache what it learns
    >>> engine.similar_items(item_id=23, n=10)  # get the best 10 matches for item 25
    [(0.91234, 44), ...]
    >>> engine.similar_users(user_id=1, only_positive_corr=False)   # all similar & dissimilar users, if any
    >>> engine.recommend_items(user_id=22, n=7)  # make best 7 user based item recommendations for user
    [(1.7889, 123), ...]
    >>> engine.recommend_users(item_id=77)  # all best item based user recommendations for item
    >>> engine.recommend_items(user_id=22, method=pyr.ITEM_BASED) # item based item recs. for user
    >>> engine.recommend_users(item_id=33, method=pyr.USER_BASED) # user based user recs. for item
    
    You should take a look at the differences between item and user based recommendations. I recommend you 
    read chapter 2 of the book "Programming Collective Intelligence", by Toby Segaran.
    """ 
    ITEM_BASED, USER_BASED = ITEM_BASED, USER_BASED
    INDEX_BASED, COLUMN_BASED = INDEX_BASED, COLUMN_BASED
    BINARY, RATING_ZERO, RATING_NAN = BINARY, RATING_ZERO, RATING_NAN
    
    def __init__(self, preferences, kind, *args, **kwargs):
        """
        Setup engine.
            preferences:    pandas dataframe with user/column preferences.
            kind:           either BINARY, with values in {0,1}, RATING_ZERO, with integer 
                            values {0, ..., N} and 0 meaning unreviewed, or RATING_NAN, with 
                            any positive integer value or NaN and NaN meaning unreviewed.
        """
        assert type(preferences) is pd.DataFrame, "Please provide a pd.DataFrame with preferences."
        assert kind in [self.BINARY, self.RATING_ZERO, self.RATING_NAN], \
                "Please specify a valid kind {BINARY, RATING_ZERO, RATING_NONE}."
        self.preferences = preferences
        self.kind = kind
        # we must assert that values in preferences fit with kind
        if kind == self.BINARY:
            # replace NaN with 0
            self.preferences.fillna(0, inplace=True)
            assert ((self.preferences == 0) | 
                    (self.preferences == 1)).all().all(), "Preferences can be in {0,1}."
        if kind == self.RATING_ZERO:
            # replace NaN with 0
            self.preferences.fillna(0, inplace=True)
            assert (self.preferences >= 0).all().all(), "Preferences can be in {0, positive int}."
        if kind == self.RATING_NAN:
            # replace 0 with NaN
            self.preferences.replace(0, np.nan, inplace=True)
            assert ((self.preferences > 0) | 
                    (self.preferences.isnull())).all().all(), "Preferences can be in {NaN, positive int}."
    
    def preload(self, verbose=False):
        """
        Preload recommendations into cache backend for faster response.
        
        TODO: make this thread safe with multithreading.Pool.map
        """
        start = time.time()
        rows, cols = self.preferences.index, self.preferences.columns
        totrows = len(rows)
        totcols = len(cols)
        if verbose:
            print "About to preload %d elements into cache backend..." % (totrows + totcols)
        args_list_1 = [
            # similarities
            # run _similar_ix_inner for ALL rows and columns
            # for rows => row ix and transpose=False
            ("Preloading row (user) similarities...", rows, False, totrows),
            # for cols => col name and transpose=True
            ("\nPreloading column (item) similarities...", cols, True, totcols)]
        for args in args_list_1:
            if verbose:
                print args[0]
            for i, ix in enumerate(args[1]):
                _ = self._similar_ix_inner(ix1=ix, transpose=args[2])
                if verbose:
                    sys.stdout.write('\r%d%%' % (100*(i+1)/float(args[3])))
                    sys.stdout.flush()
        if verbose:
            print
        args_list_2 = [
            # now column based recommendations
            # run similar_ix for ALL rows and columns with only_positive_corr=True, n=None
            # this will create a cache for recommend_col_bycol
            # rows => row ix, transpose=False, only_positive_corr=True, n=None
            ("Preloading column (item) based recommendations for rows (users)...", rows, False, totrows),
            # cols => col name, transpose=True, only_positive_corr=True, n=None
            ("\nPreloading row (user) based recommendations for columns (items)...", cols, True, totcols)]
        for args in args_list_2:
            if verbose:
                print args[0]
            for i, ix in enumerate(args[1]):
                _ = self.similar_ix(ix1=ix, transpose=args[2], n=None, only_positive_corr=True)
                if verbose:
                    sys.stdout.write('\r%d%%' % (100*(i+1)/float(args[3])))
                    sys.stdout.flush()
        if verbose:
            print
            print "Done in %s secs." % str(time.time() - start)        
    
    @memoize
    def similar_users(self, user_id, n=None, only_positive_corr=True):
        """
        Returns a ranked list of similart users.
        """
        # similar_ix: ix1, transpose=False, n=None, only_positive_corr=False
        return self.similar_ix(**{
                'ix1': user_id,
                'n': n,
                'only_positive_corr': only_positive_corr,
                'transpose': False})
    
    @memoize
    def similar_items(self, item_id, n=None, only_positive_corr=True):
        """
        Returns a ranked list of similar items.
        """
        # similar_ix: ix1, transpose=False, n=None, only_positive_corr=False
        return self.similar_ix(**{
            'ix1': item_id,
            'n': n,
            'only_positive_corr': only_positive_corr,
            'transpose': True})
    
    @memoize
    def recommend_items(self, user_id, n=None, only_positive_corr=True, method=USER_BASED):
        """
        Returns a ranked list of recommended items for a user, using a row or column based approach.
        """
        # recommend_cols: ix, transpose=False, only_positive_corr=True, n=None, binary=False
        kwargs = {
            'ix': user_id,
            'transpose': False,
            'n': n,
            'binary': self.kind == self.BINARY,
            'only_positive_corr': only_positive_corr
        }
        if method == self.USER_BASED:
            return self.recommend_cols(**kwargs)
        # else it's item based
        # recommend_cols_bycol: ix, transpose=False, n=None, binary=False
        # we need to pop only_positive_corr
        kwargs.pop('only_positive_corr')
        return self.recommend_cols_bycol(**kwargs)
    
    @memoize
    def recommend_users(self, item_id, n=None, only_positive_corr=True, method=ITEM_BASED):
        """
        Returns a ranked list of recommended users for an item, using a row or column based approach. 
        """
        # recommend_cols: ix, transpose=False, only_positive_corr=True, n=None, binary=False
        kwargs = {
            'ix': item_id,
            'transpose': True,
            'n': n,
            'binary': self.kind == self.BINARY,
            'only_positive_corr': only_positive_corr
        }
        if method == self.ITEM_BASED:
            return self.recommend_cols(**kwargs)
        # recommend_cols_bycol: ix, transpose=False, n=None, binary=False
        # we need to pop only_positive_corr
        kwargs.pop('only_positive_corr')
        return self.recommend_cols_bycol(**kwargs)
    
    # class helpers
    @memoize
    def _similar_ix_inner(self, ix1, transpose=False):
        """
        This method returns every similar_ix to ix1. The idea is to avoid 
        recalculating similar_ix for different values of n and only_positive_corr.
        
        This is the *core method of the engine*, so it needs to be very efficient.
        """
        # transpose dataframe?
        df = self.preferences.transpose() if transpose else self.preferences
        _pcorr = lambda i, j: df.ix[[i, j]].transpose().corr().fillna(0, inplace=False).loc[i, j]
        def _inner(i,j):
            try:
                return _pcorr(i, j)
            except:
                raise Exception(", ".join([str(i), str(j)]))
        # this returns an unsorted list of similarities. there is *no value checking at all*,
        # so use wrapping methods, such as similar_ix or recommend_cols to acces these values
        return [(_inner(ix1, ix2), ix2) for ix2 in df.index if ix1 != ix2]
    
    def similar_ix(self, ix1, transpose=False, n=None, only_positive_corr=False):
        """
        Return a sorted list of pearson-similar indexes, for a given index and dataframe.
        """
        assert type(only_positive_corr) is bool
        similar = self._similar_ix_inner(ix1=ix1, transpose=transpose)
        if only_positive_corr:
            similar = [(score, ix) for (score, ix) in similar if score > 0]
        n = n if (type(n) is int and n > 0) else len(similar)
        return sorted(similar, reverse=True)[:n]
    
    def recommend_cols(self, ix, transpose=False, only_positive_corr=True, n=None, binary=False):
        """
        Return row based recommended columns for specific index.
        """
        assert type(transpose) is bool
        assert type(binary) is bool
        
        # first get similar rows by score and unreviewed cols for ix
        # similar_ix: ix1, transpose, n, only_positive_corr
        sims = self.similar_ix(**{
                'ix1': ix, 
                'transpose': transpose, 
                'only_positive_corr': only_positive_corr, 
                'n': n})
        # if there are no similarities we cannot make recommendations
        if len(sims) == 0:
            return []
        # we have sims, let's move on
        # transpose df?
        df = self.preferences.transpose() if transpose else self.preferences
        # get similarities 
        sim_score, index = map(np.asarray, zip(*sims))
        not_reviewed = df.columns[df.ix[ix].fillna(0) == 0]
        # filter relevant info
        df_sub = df.loc[index, not_reviewed]
        if df_sub.empty:
            return []
        # calculate weighted ratings
        weighted_ratings = df_sub.apply(lambda x: x*sim_score).sum()
        # sum similarity scores in order to normalize weighted 
        # ratings -- we don't want a very popular column to bias recommendations
        # in the case of rating engines, we just need to add anyting that is > 0
        # but in the case of binary engines, rating 0 is the same as "reviewing 0",
        # which means that we have to take into account all similarity scores and 
        # not just the "active" ones
        sum_sim_score = sim_score.sum() if binary else (df_sub > 0).apply(lambda x: x*sim_score).sum()
        # recommended columns are now normalized by sum of sim scores
        recs = weighted_ratings/sum_sim_score
        # are there any np.nan values or zero values in recs?
        # we may still have these, as only_positive_corr only 
        # assures positively correlated rows, but has nothing 
        # to say about which col values can be considered "active"
        recs = recs[recs > 0]
        n = n if (type(n) is int and n > 0) else len(recs)
        return sorted(zip(recs.values, recs.index), reverse=True)[:n]
    
    def recommend_cols_bycol(self, ix, transpose=False, n=None, binary=False):
        """
        Returns a ranked list of recommended columns for a given index, using a column 
        method and pearson similarity distance.
        """
        assert type(transpose) is bool
        assert type(binary) is bool
        
        df = self.preferences.transpose() if transpose else self.preferences
        # we need to get all active columns and respective ratings for this index
        active_ratings = df.ix[ix].replace(0, np.nan).dropna()
        # if there are no active ratings, we cannot make any recommendations
        if active_ratings.empty:
            return []
        # now we need to build a dataframe with all the recommended items and their scores
        # this dataframe will be of the form:
        #
        # similarity | recommended_column | rating
        # -----------|--------------------|-------
        #  0.32144   |  My super movie    |  4.4        # both these items correspond
        #  0.13234   |  Some other movie  |  4.4        # to the same active rating
        #  1.34355   |   A similar movie  |  1.0
        #  ...
        # 
        # let's build our helper
        def _getdf(row):
            # similar_ix will first try to use cache
            # similar_ix: ix1, transpose=False, n=None, only_positive_corr=False
            sims = self.similar_ix(**{
                    'ix1': row[1],
                    # we are getting a list of rating, col
                    # so if we want col similarity we need to use !transpose
                    'transpose': not transpose,
                    'n': None,
                    # it's important to only request positive corr
                    'only_positive_corr': True})
            _df = pd.DataFrame(sims, columns=['similarity', 'recommended_column'])
            _df['rating'] = row[0]
            return _df
        # map over our ratings and drop already rated columns
        _dfc = pd.concat(map(_getdf, zip(active_ratings.values, active_ratings.index)))
        dfconcat = _dfc[~_dfc['recommended_column'].isin(active_ratings.index)].copy()
        del _dfc
        # now we need to do 2 things:
        # 1. add a similarity*rating column
        # 2. group by recommended_column, aggregating by sum simmilarity*rating and simmilarity
        dfconcat['similarity*rating'] = dfconcat['similarity']*dfconcat['rating']
        # dfconcat will have some 0 value rows when working with binary data
        dfagg = dfconcat.groupby(['recommended_column']).agg({'similarity': 'sum', 'similarity*rating': 'sum'})
        # now we need to normalize values, as very popular columns can bias our recommendations.
        # this is a bit tricky when it comes to binary data, as "not reviewing" is a review itself
        # and therefore we should consider the sum of all similarities (or averaged similarity for all columns)
        # when we are dealing with rating engines, we only need the sum of similarities for reviewed columns
        recs = (dfagg['similarity*rating']/dfagg['similarity']) if not binary \
                else (dfagg['similarity*rating']/dfagg['similarity'].sum())
        # are there any np.nan values or zero values in recs?
        # we may still have these, as only_positive_corr only 
        # assures positively correlated rows, but has nothing 
        # to say about which col values can be considered "active"
        recs = recs[recs > 0]
        # return the 'n' top elements
        n = n if (type(n) is int and n > 0) else recs.count()
        return sorted(zip(recs.values, recs.index), reverse=True)[:n]


class SimpleJaccardEngine(SimplePearsonEngine):
    """
    Much like the pearson engine, but based on Jaccard similarity. This engine works better 
    when sparcity is an issue, as it only considers interesection and union of preferences 
    between rows. Also, note that either using binary, rating or using NaN vs 0 has no real 
    effect on the similarities, though it may have effect on recommendations, due to ratings.
    """
    @memoize
    def _similar_ix_inner(self, ix1, transpose=False):
        """
        This method returns every similar_ix to ix1. The idea is to avoid 
        recalculating similar_ix for different values of n and only_positive_corr.
        
        This is the *core method of the engine*, so it needs to be very efficient.
        """
        # transpose dataframe?
        # also note that in this case non-rated columns don't matter
        df = (self.preferences.transpose() if transpose else self.preferences).fillna(0)
        def _inner(i,j):
            try:
                # get columns that are effectively rated
                si, sj = map(lambda value: set(value[1][value[1] != 0].index), df.ix[[i,j]].iterrows())
                # intersection vs union of sets
                num = len(si & sj)
                den = float(len(si | sj))
                # jaccard or 0.0
                return num/den if den else 0.0
            except:
                raise Exception(", ".join([str(i), str(j)]))
        # this returns an *unsorted* list of similarities. there is *no value checking at all*,
        # so use wrapping methods, such as similar_ix or recommend_cols to acces these values
        return [(_inner(ix1, ix2), ix2) for ix2 in df.index if ix1 != ix2]
