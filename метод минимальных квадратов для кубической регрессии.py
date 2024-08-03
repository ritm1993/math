import pandas as pd

# create DataFrame
df = pd.DataFrame({'x': [1, 2, 3, 5, 7, 8, 11, 14, 15],
                   'y': [-3, 3, 5, 4, 1, -5, 6, 3, 2]})
# view DataFrame
print(df)
x
y
0
1 - 3
1
2
3
2
3
5
3
5
4
4
7
1
5
8 - 5
6
11
6
7
14
3
8
15
2
import matplotlib.pyplot as plt

# create scatterplot
plt.scatter(df.x, df.y)
< matplotlib.collections.PathCollectionat0x7facf070ab00>

import numpy as np

# fit cubic regression model
model = np.poly1d(np.polyfit(df.x, df.y, 3))

# add fitted cubic regression line to scatterplot
polyline = np.linspace(1, 60, 50)
plt.scatter(df.x, df.y)
plt.plot(polyline, model(polyline))

# add axis labels
plt.xlabel('x')
plt.ylabel('y')

# display plot
plt.show()
