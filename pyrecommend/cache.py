import sqlitedict
import tempfile
from functools import wraps
import inspect
import weakref


class MySqliteDict(sqlitedict.SqliteDict):
    """
    Avoids calling __del__ on an already deleted reference.
    """
    def __del__(self):
        try:
            super(MySqliteDict, self).__del__
        except:
            pass


class SQLiteCacheBackend(object):
    """
    A memoize-like cache backend for SPE recommendation calls.
    """
    def __init__(self):
        # create tempfile
        _, self.fname = tempfile.mkstemp(suffix='.db')
        # create cache
        self.cache = MySqliteDict(self.fname, autocommit=True)
    
    def __del__(self):
        # try to remove cache file
        import os
        os.remove(self.fname)
    
    def __setitem__(self, key, value):
        self.cache[key] = value
    
    def __getitem__(self, key):
        return self.cache[key]


def memoize(func):
    """
    This decorator wraps func so it checks memoized cache before returning result. 
    Note func must recieve self as first argument, as __memoize will look for the 
    cache backend in self.memoize_backend, which is a dict like object, supporting 
    __getitem__ and __setitem__.
    
    Usage:
    
    class Foo(MyParent, Memoized):
        @memoize
        def squared(self, x):
            return x**2
    
    >>> foo = Foo(memoize_backend=MyBackend()) # default backend is SQLiteCacheBackend
    >>> foo.squared(10)    # will store result under hash considering {'x': 10}
    >>> foo.squared(10)    # will return result from cache
    >>> foo.squared(x=10)  # will also return result from cache
    >>> foo2 = Foo(memoize_backend=dict()) # it can be a dict too
    
    If you do use a dict as backend, remember python passes dicts and lists by reference!
    
    class Foo(MyParent, Memoized):
        def __init__(self, memoize_backend=dict(), *args, **kwargs):
            self.memoize_backend = memoize_backend  # wrong! all foo objects will 
                                                    # have the same dict as backend
    
    class Foo(MyParent, Memoized):
        def __init__(self, *args, **kwargs):
            # ...
    
    >> foo = Foo(memoize_backend=dict())  # correct
    """
    @wraps(func)
    def _inner(self, *args, **kwargs):
        # we need to build a hashed key from func name, args and kwargs
        # note that @memoize will only work with instance methods
        d  = inspect.getcallargs(func, self, *args, **kwargs)
        d['func'] = func.__name__
        hsh = ":".join([ ":".join([str(k), str(v)]) for k,v in sorted(d.items())])
        key = hash(hsh)
        # first try to get result from backend
        # use KeyError except, as stored result could be None
        try:
            result = self.memoize_backend[key]
        except KeyError:
            result = func(self, *args, **kwargs)
            # backend should be defined in your memoized class.__init__
            self.memoize_backend[key] = result
        return result
    return _inner


class Memoized(object):
    def __new__(cls, *args, **kwargs):
        cls.memoize_backend = kwargs.get('memoize_backend', SQLiteCacheBackend())
        return super(Memoized, cls).__new__(cls)

