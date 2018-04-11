from sklearn.utils.multiclass import type_of_target
import numpy
from scipy import stats
import math


class InformationValue:
    def __init__(self,min_woe=-20,max_woe=20):
        self.min_woe = min_woe
        self.max_woe = max_woe

    def check_binary(self,y):
        if type_of_target(y) != 'binary': raise ValueError('Class must be binary')

    def discrete_feature(self,x,discrete_count=None):
        n,m = x.shape
        _x = numpy.zeros((n,m))
        for i in range(m):
            xx = x[:,i]
            if type_of_target(xx) == 'continuous':
                if discrete_count is None:
                    _x[:,i] = self.discrete_continuous_feature(xx)
                else:
                    _x[:,i] = self.discrete_continuous_feature(xx,discrete_count.get(i,5))
            else: _x[:,i] = xx
        return _x

    def discrete_continuous_feature(self,x,n=5):
        m = len(x)
        _x = numpy.zeros(m)
        for i in range(n):
            p1 = stats.scoreatpercentile(x,100/float(n)*i)
            p2 = stats.scoreatpercentile(x,100/float(n)*(i+1))
            _x[numpy.where((x >= p1) & (x <= p2))] = i
        return _x

    def get_woe_iv(self,x,y):
        self.check_binary(y)
        total_n0,total_n1 = self.value_count(y)

        _x = self.discrete_feature(x)
        n,m = x.shape
        woe = []
        iv = []

        for i in range(m):
            xx = _x[:,i]
            _xx = numpy.unique(xx)
            _woe = {}
            _iv = 0
            for v in _xx:
                vn0,vn1 = self.value_count(y[numpy.where(xx == v)])
                r0 = float(vn0)/total_n0
                r1 = float(vn1)/total_n1
                if r0 == 0: _woe[v] = self.max_woe
                elif r1 == 0: _woe[v] = self.min_woe
                else: _woe[v] = math.log(r1/r0)
                _iv += (r1-r0)*_woe[v]
            woe.append(_woe)
            iv.append(_iv)
        return woe,iv

    def value_count(self,y):
        n0 = len(y[y == 0])
        n1 = len(y[y == 1])
        return n0,n1


if __name__ == '__main__':
    import pandas
    from sklearn.preprocessing import Imputer
    data = pandas.read_csv('cs-training.csv')
    y = data['SeriousDlqin2yrs'].values
    x = data.drop(['ID', 'SeriousDlqin2yrs'], axis=1).values
    imputer = Imputer()
    x = imputer.fit_transform(x)
    woe, iv = InformationValue().get_woe_iv(x,y)
    print(numpy.array(iv))