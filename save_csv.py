# A code snippet to help you save your results into a kaggle accepted csv
import pandas as pd
import numpy as np

def results_to_csv(y_test, file_name):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1
    df.to_csv(file_name, index_label='Id')
