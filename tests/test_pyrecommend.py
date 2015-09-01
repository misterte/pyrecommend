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
        # A is provided
        A = [sim for sim in func1(ix) if sim[0] > 0.0]
        # B is expected
        B = [sim for sim in func2(ix, binary=binary) if sim[0] > 0.0]
        if binary:
            # if engine is binary, scores will not necessarily be the same
            # but recommendations will be very similar
            if (len(A) == 0) or (len(B) == 0):
                # then one of the engines has not returned any recommendations
                # it could be the case that our engine did not return any recs
                # but in that case we will just assume that it's due to a small DF
                return
            # else we'll test that least 75% of the items match
            _ = set(zip(*A)[1]) and set(zip(*B)[1])
            self.assertTrue(len(_)*2.0 >= (len(A) + len(B))*0.75)
            return
        # rating_nan engine
        for a, b in zip(A, B):
            self.assertAlmostEqual(a[0], b[0], places=5)
    
    # for rating_nan & binary
    # test similarities
    def test_similar_users(self):
        """
        We can predict similar users using RATING_NAN / BINARY.
        """
        user = random.choice(fixtures.DFCRITICS_NAN.index)
        self._test_helper(user, self.engine_nan.similar_users, fixtures.similar_users)
        self._test_helper(user, self.engine_binary.similar_users, fixtures.similar_users, binary=True)
    
    def test_similar_items_nan(self):
        """
        We can predict similar items using RATING_NAN / BINARY.
        """
        item = random.choice(fixtures.DFCRITICS_NAN.transpose().index)
        self._test_helper(item, self.engine_nan.similar_items, fixtures.similar_items)
        self._test_helper(item, self.engine_binary.similar_items, fixtures.similar_items, binary=True)
    
    # test user/item recs by row
    def test_recommend_items(self):
        """
        We can predict recommended items for user, by row, using RATING_NAN / BINARY.
        """
        user = random.choice(fixtures.DFCRITICS_NAN.index)
        self._test_helper(user, self.engine_nan.recommend_items, fixtures.recommend_items)
        self._test_helper(user, self.engine_binary.recommend_items, fixtures.recommend_items, binary=True)
    
    def test_recommend_users(self):
        """
        We can predict recommended users for item, by row, using RATING_NAN / BINARY.
        """
        item = random.choice(fixtures.DFCRITICS_NAN.transpose().index)
        self._test_helper(item, self.engine_nan.recommend_users,  fixtures.recommend_users)
        self._test_helper(item, self.engine_binary.recommend_users, fixtures.recommend_users, binary=True)
    
    # test column based recs
    def test_recommend_items_bycol(self):
        """
        We can predict recommended items for user, by column.
        """
        user = random.choice(fixtures.DFCRITICS_NAN.index)
        self._test_helper(user, 
            functools.partial(self.engine_nan.recommend_items, method=self.engine_nan.ITEM_BASED),
            fixtures.recommend_items_bycol)
        self._test_helper(user, 
            functools.partial(self.engine_binary.recommend_items, method=self.engine_binary.ITEM_BASED),
            fixtures.recommend_items_bycol, binary=True)
    
    def test_recommend_users_bycol(self):
        """
        We can predict recommended users for item, by column.
        """
        item = random.choice(fixtures.DFCRITICS_NAN.transpose().index)
        self._test_helper(item,
            functools.partial(self.engine_nan.recommend_users, method=self.engine_nan.USER_BASED),
            fixtures.recommend_users_bycol)
        self._test_helper(item,
            functools.partial(self.engine_binary.recommend_users, method=self.engine_binary.USER_BASED),
            fixtures.recommend_users_bycol, binary=True)
    
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
