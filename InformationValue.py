import math
import numpy
from scipy import stats
from sklearn.utils.multiclass import type_of_target


class InformationValue:
    def __init__(self,min_woe=-20,max_woe=20):
        self.min_woe = min_woe
        self.max_woe = max_woe

    def check_binary(self,y):
        if type_of_target(y) != 'binary': raise ValueError('Class must be binary')

    def discrete_continuous_feature(self,x,v):
        m = len(x)
        _x = numpy.zeros(m)
        n = len(v)-1
        for i in range(n):
            p = v[i]
            q = v[i+1]
            _x[numpy.where((x >= p) & (x < q))] = i
        return _x

    def discrete_feature(self,x,v):
        n,m = x.shape
        _x = numpy.zeros((n,m))
        for i in range(m):
            xx = x[:,i]
            if type_of_target(xx) == 'continuous': _x[:,i] = self.discrete_continuous_feature(xx,v[i])
            else: _x[:,i] = xx
        return _x

    def count_value(self,y):
        return len(y[y == 0]),len(y[y == 1])

    def get_woe_iv(self,x,y,v):
        self.check_binary(y)
        n0,n1 = self.count_value(y)
        _x = self.discrete_feature(x,v)
        n,m = x.shape
        woe = []
        iv = []
        for i in range(m):
            _woe = {}
            _iv = 0
            xx = _x[:,i]
            ux = numpy.unique(xx)
            for v in ux:
                _n0,_n1 = self.count_value(y[numpy.where(xx == v)])
                r0 = float(_n0)/n0
                r1 = float(_n1)/n1
                if r0 == 0: _woe[v] = self.max_woe
                elif r1 == 0: _woe[v] = self.min_woe
                else: _woe[v] = math.log(r1/r0)
                _iv += (r1-r0)*_woe[v]
            woe.append(_woe)
            iv.append(_iv)
        return woe,iv


if __name__ == '__main__':
    def get_feature_range(x):
        n,m = x.shape
        v = {}
        for i in range(m):
            xx = x[:,i]
            if type_of_target(xx) == 'continuous':
                v[i] = [
                    -numpy.inf,
                    stats.scoreatpercentile(xx, 20),
                    stats.scoreatpercentile(xx, 40),
                    stats.scoreatpercentile(xx, 60),
                    stats.scoreatpercentile(xx, 80),
                    numpy.inf
                ]
        return v


    import pandas
    data = pandas.read_csv('cs-training.csv')
    y = data['SeriousDlqin2yrs'].values
    x = data.drop(['ID','SeriousDlqin2yrs'],axis=1).values
    from sklearn.preprocessing import Imputer
    imputer = Imputer()
    x = imputer.fit_transform(x)
    woe,iv = InformationValue().get_woe_iv(x,y,get_feature_range(x))
    print(numpy.array(iv))