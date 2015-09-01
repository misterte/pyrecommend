#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_pyrecommend
----------------------------------

Tests for `pyrecommend` module.
"""
import random
import unittest
import functools
import fixtures
from pyrecommend import SimplePearsonEngine as SPE




class TestPyrecommend(unittest.TestCase):
    def setUp(self):
        """
        Preload an engine, sim_rows and sim_cols for column based recommendations.
        """
        self.engine_nan = SPE(fixtures.DFCRITICS_NAN, SPE.RATING_NAN)
        self.engine_nan.preload()
        self.engine_binary = SPE(fixtures.DFCRITICS_BINARY, SPE.BINARY)
        self.engine_binary.preload()
        
    # tests
    # helper
    def _test_helper(self, ix, func1, func2, binary=False):
        A = [sim for sim in func1(ix) if sim[0] > 0.0]
        B = [sim for sim in func2(ix, binary=binary) if sim[0] > 0.0]
        for a, b in zip(A, B):
            #fname = func1.__name__
            self.assertAlmostEqual(a[0], b[0], places=5)
            #places=5, message="f=%s; ix=%s" % (func1.__name__, str(ix)))
    
    # for rating_nan
    # test similarities
    def test_similar_users_nan(self):
        """
        We can predict similar users using RATING_NAN.
        """
        user = random.choice(fixtures.DFCRITICS_NAN.index)
        self._test_helper(user, self.engine_nan.similar_users, fixtures.similar_users)
    
    def test_similar_items_nan(self):
        """
        We can predict similar items using RATING_NAN.
        """
        item = random.choice(fixtures.DFCRITICS_NAN.transpose().index)
        self._test_helper(item, self.engine_nan.similar_items, fixtures.similar_items)
    
    # test user/item recs by row
    def test_recommend_items(self):
        """
        We can predict recommended items for user, by row.
        """
        user = random.choice(fixtures.DFCRITICS_NAN.index)
        self._test_helper(user, self.engine_nan.recommend_items, fixtures.recommend_items)
    
    def test_recommend_users(self):
        """
        We can predict recommended users for item, by row.
        """
        item = random.choice(fixtures.DFCRITICS_NAN.transpose().index)
        self._test_helper(item, self.engine_nan.recommend_users,  fixtures.recommend_users)
    
    # test column based recs
    def test_recommend_items_bycol(self):
        """
        We can predict recommended items for user, by column.
        """
        user = random.choice(fixtures.DFCRITICS_NAN.index)
        self._test_helper(user, 
            functools.partial(self.engine_nan.recommend_items, method=self.engine_nan.ITEM_BASED),
            fixtures.recommend_items_bycol)
    
    def test_recommend_users_bycol(self):
        """
        We can predict recommended users for item, by column.
        """
        item = random.choice(fixtures.DFCRITICS_NAN.transpose().index)
        self._test_helper(item,
            functools.partial(self.engine_nan.recommend_users, method=self.engine_nan.USER_BASED),
            fixtures.recommend_users_bycol)
    
    # same for binary
    
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
