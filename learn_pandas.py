import numpy as np
import pandas as pd

my_data = np.array([[0,3], [10,7], [20,9], [30,14], [40,15]])

my_column_names = ["temperature", "activity"]

# create dataFrame
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

print(my_dataframe)

# add column to DataFrame
my_dataframe["adjusted"] = my_dataframe["activity"] + 2
print(my_dataframe)

#specifying a Subnet of a DataFrame
print("Rows 0, 1 and 2:")
print(my_dataframe.head(3))

print("Row #2:")
print(my_dataframe.iloc[[2]])

print("Rows #1, #2 and#3:")
print(my_dataframe[1:4])

print("Column 'Temperature': ")
print(my_dataframe["temperature"][1:4])