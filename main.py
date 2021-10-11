import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r"DATASET_LOCATION")
df_to_numpy = df.to_numpy()


# Cosine Similarity function


def numerator(x, y):
    return np.dot(x, y)


def denominator(x, y):
    list_x = list()
    list_y = list()

    for i in x:
        list_x.append(np.square(i))
    x_sqroot = np.sqrt(sum(list_x))

    for k in y:
        list_y.append(np.square(k))
    y_sqroot = np.sqrt(sum(list_y))

    return x_sqroot * y_sqroot


def cosine_similarity_my_func(x, y):
    num = numerator(x, y)
    den = denominator(x, y)

    return num / den


sim_0 = list()

# Or we can use cosine_similarity from sklearn too instead of cosine_similarity_my_func :)

for i in range(0, len(df_to_numpy)):
    sim_0.append(cosine_similarity_my_func(df_to_numpy[0], df_to_numpy[i]))

for ind, val in enumerate(sim_0):
    if val > 0.999:
        print(
            f"This index {ind} is similar to sample 0 with similariy value of {(sim_0[ind])}"
        )


print(cosine_similarity_my_func(df_to_numpy[0], df_to_numpy[18385]))
