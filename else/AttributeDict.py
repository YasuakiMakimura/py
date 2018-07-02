# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class DotAccessible(object):
    """オブジェクトグラフ内の辞書要素をプロパティ風にアクセスすることを可能にするラッパー。
        DotAccessible( { 'foo' : 42 } ).foo==42

    メンバーを帰納的にワップすることによりこの挙動を下層オブジェクトにも与える。
        DotAccessible( { 'lst' : [ { 'foo' : 42 } ] } ).lst[0].foo==42
    """

    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return "DotAccessible(%s)" % repr(self.obj)

    def __getitem__(self, i):
        """リストメンバーをラップ"""
        return self.wrap(self.obj[i])

    def __getslice__(self, i, j):
        """リストメンバーをラップ"""

        return map(self.wrap, self.obj.__getslice__(i, j))

    def __getattr__(self, key):
        """辞書メンバーをプロパティとしてアクセス可能にする。
        辞書キーと同じ名のプロパティはアクセス不可になる。
        """

        if isinstance(self.obj, dict):
            try:
                v = self.obj[key]
            except KeyError:

                v = self.obj.__getattribute__(key)
        else:
            v = self.obj.__getattribute__(key)

        return self.wrap(v)

    def wrap(self, v):
        """要素をラップするためのヘルパー"""

        if isinstance(v, (dict, list, tuple)):  # xx add set
            return self.__class__(v)
        return v


class AttributeDict:
    def __init__(self, obj):
        self.obj = obj

    # pickle化されるときに呼び出される
    def __getstate__(self):
        return self.obj.items()

    # 非pickle化されるときに呼び出される
    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def fields(self):
        return self.obj

    def keys(self):
        return self.obj.keys()


if __name__ == "__main__":
    # d = DotAccessible(dict(foo=1))
    d = AttributeDict({"foo": 1})
    # d = Map({"foo": 1}, last_name='Pool', age=24, sports=['Soccer'])
    d.new_key = 'hel'
    # print(d.foo)
    # print(d.new_key)
    # print(d.fields())
