import sys
from os import listdir
from os.path import isfile, isdir, dirname
from operator import add

from pyspark import SparkContext

def calculate(context, directory, label):
    path_root = '/home/DATA/NOAA_weather/'

    all_files = []
    output = None
    for f in directory:
        lines = context.textFile(path_root + f)
        all_files.append((f, lines))

    for year, data in all_files:
        output = data.map(lambda word: (f, word[66:70])).reduceByKey(lambda a, b: a + b).collectAsMap()

    maximum_wind = max(output.items(), key=(lambda key: output[key]))

    return 'Max for the {}: {}'.format(label, maximum_wind)

def main():
    if len(sys.argv) != 2:
        print("usage: climate_change data_path", file=sys.stderr)
        sys.exit(1)

    data_path = sys.argv[1]
    dir1980 = []
    dir2000 = []
    for f in listdir(data_path):
        year = None
        try:
            year = int(f)
        except ValueError as ve:
            print('not a number, skipping: {}'.format(f))

        if type(year) is int and year >= 1980 and year <= 1989:
            dir1980.append(f)
        elif type(year) is int and year >= 2000 and year <= 2009:
            dir2000.append(f)
        else:
            print('out of range, skipping: {}'.format(f))

    sc = SparkContext(appName="CleanCoal")
    data = calculate(sc, dir1980, '80\'s')
    print(data)
    sc.stop()

main()
