""" This script is used for loading and preliminarily processing the data. """
import csv
import numpy as np

DATA_PATH = "bid_data.csv"

def load_data(data_path):
    """ Given a data_path, this function reads a csv file into a np.array. """
    with open(data_path) as csv_file: # Reads the csv file into a list.
        csv_reader = csv.reader(csv_file, delimiter=",")
        rows = list(csv_reader)
    
    del rows[0] # Remove the header, which are "value" and "bid".
    
    rows = remove_empty_rows(rows) # Remove empty rows.
    # Convert data from string to float format
    rows = [[float(a) for a in item] for item in rows]
    return np.array(rows)
    

def remove_empty_rows(array_x):
    """ Given an array_x, removes all rows that contain an empty data. """
    output = [x for x in array_x if all(elem != " " for elem in x)]
    return output