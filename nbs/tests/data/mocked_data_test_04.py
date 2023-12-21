import pandas as pd
import numpy as np

# Data for tests
# See Der, G., &amp; Everitt, B. S. (2009). A handbook
# of statistical analyses using SAS, from Display 11.1

# to remove the array wrapping behaviour of black
# fmt: off
group = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
first = [20, 14, 7, 6, 9, 9, 7, 18, 6, 10, 5, 11, 10, 17, 16, 7, 5, 16, 2, 7, 9, 2, 7, 19,
         7, 9, 6, 13, 9, 6, 11, 7, 8, 3, 4, 11, 1, 6, 0, 18, 15, 10,  6,  9,  4,  4, 10]
second = [15, 12, 5, 10, 7, 9, 3, 17, 9, 15, 9, 11, 2, 12, 15, 10, 0, 7, 1, 11, 16,
        5, 3, 13, 5, 12, 7, 18, 10, 7, 11, 10, 18, 3, 10, 10, 3, 7, 3, 18, 15, 14, 6, 9, 3, 13, 11]
third = [14, 12, 5, 9, 9, 9, 7, 16, 9, 12, 7, 8, 9, 14, 12, 4, 5, 7, 1, 7, 14, 6, 5, 14, 8, 16, 10,
         14, 12, 8, 12, 11, 19, 3, 11, 10, 2, 7, 3, 19, 15, 16, 7, 13, 4, 13, 13]
fourth = [13, 10, 6, 8, 5, 11, 6, 14, 9, 12, 3, 8, 3, 10, 7, 7, 0, 6, 2, 5, 10, 7, 5, 12, 8, 17, 15,
          21, 14, 9, 14, 12, 19, 7, 17, 15, 4, 9, 4, 22, 18, 17, 9, 16, 7, 16, 17]
fifth = [13, 10, 5, 7, 4, 8, 5, 12, 9, 11, 5, 9, 5, 9, 9, 5, 0, 4, 2, 8, 6, 6, 5, 10, 6, 18, 16, 21,
         15, 12, 16, 14, 22, 8, 18, 16, 5, 10, 6, 22, 19, 19, 10, 20, 9, 19, 21]
# fmt: on

df = pd.DataFrame(
    {
        "Group": group,
        "First": first,
        "Second": second,
        "Third": third,
        "Fourth": fourth,
        "Fifth": fifth,
        "ID": np.arange(0, 47),
    }
)