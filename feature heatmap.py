import seaborn as sns
import pandas, numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# import data: 20 features or 3 PCs
texture = pandas.read_csv('training 7patients round2 4k pca 1014.csv')
array = texture.values
# x = array[:, 0:20]
x = array[:, 0:3]
scaler = MinMaxScaler().fit(x)
rescaledX = scaler.transform(x)


# y = array[:, 23]
y = array[:, 3]
# randomly select 0.01% of the data for reducing computational time
number_of_rows = x.shape[0]
random_size = round(number_of_rows*0.0001)
random_indices = numpy.random.choice(number_of_rows, size=random_size, replace=False)
random_matrix = numpy.append(rescaledX[random_indices, :], y[random_indices].reshape(-1, 1), 1)
# sort the data by the cluster number
sorted_matrix = random_matrix[numpy.argsort(random_matrix[:, 3])]
x_random_rows = sorted_matrix[:, 0:3]
y_random_rows = sorted_matrix[:, 3]
# plot the heatmap
palette = sns.color_palette()
color_dict = {}
for col in range(0, random_size):
    if y_random_rows[col] == 2:
        color_dict[col] = palette[2]  # green
    elif y_random_rows[col] == 1:
        color_dict[col] = palette[1]  # yellow
    else:
        color_dict[col] = palette[3]  # red
# Convert the dictionary into a Series
color_rows = pandas.Series(color_dict)
matrix = pandas.DataFrame(x_random_rows)
g = sns.clustermap(matrix, row_cluster=False, col_cluster=False, row_colors=color_rows)
plt.show()
