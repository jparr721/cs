import sys
from os import listdir
from os.path import isfile, join
from operator import add

from pyspark import SparkContext

def calculate(context, dir, label):
    all_files = []
    for f in dir:
        lines = sc.textFile(f)
        all_files.append(lines)

    for data in all_files:
        data.map(lambda word: word = (word[61:63]))
            .reduceByKey(lambda a, b: a + b).collectAsMap()

    maximum_wind = ""
    return 'Max for the {}: {}'.format(label, maximum_wind)

def main():
    if len(sys.argv) != 2:
        print("usage: climate_change data_path", file=sys.stderr)
        sys.exit(1)

    data_path = sys.argv[1]
    80dirs = [f for f in listdir(data_path) if isdir(f) and int(f) > 1980 and int(f) <= 1989]
    20dirs = [f for f in listdir(data_path) if isdir(f) and int(f) > 2000 and int(f) <= 2009]

    sc = SparkContext(appName="CleanCoal")
    data = calculate(sc, 80dirs, '80\'s')
    print(data)
    sc.stop()

main()
