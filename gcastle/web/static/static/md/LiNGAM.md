# LiNGAM
lingam python code(lingam.py)
You can find causal structure of variables using ICA-LiNGAM.

Reference: [LiNGAM paper](http://www.jmlr.org/papers/volume7/shimizu06a/shimizu06a.pdf), [Qiita(Japanese)](https://qiita.com/ragAgar/items/131b17171b93f63d6475)

### 1. Ready data that you want to know causal structure.(LiNGAM only can use Continuous variables.)

```python3
import numpy as np
import pandas as pd

size = 10000
np.random.seed(2017)
x = np.random.uniform(size=size)

np.random.seed(1028)
y = 3*x + np.random.uniform(size=size)

X = pd.DataFrame(np.asarray([x,y]).T,columns=["x","y"])
```

### 2. Use LiNGAM. Select ICA method kurtosis(default) or negentropy(sklearn).

use kurtosis-based ICA
```python3
lingam = LiNGAM()
lingam.fit(X)
```

or 

use negentropy-based ICA
```python3
lingam = LiNGAM()
lingam.fit(X,use_sklearn=True)
```

And, get result(same result).

```result
x ---|2.991|---> y

array([[ 0.        ,  0.        ],
       [ 2.99149033,  0.        ]])
```

This means correct result.

       x =    0*x + 0*y + e1 = e1
      
       y = 2.99*x + 0*y + e2 = 2.99x + e2