import sys
from collections import defaultdict
from os import listdir
from os.path import isfile, isdir, dirname
from operator import add

from pyspark import SparkContext

def calculate(context, directory, label):
    path_root = '/home/DATA/NOAA_weather/'
    max_speed = defaultdict(int)
    min_speed = defaultdict(int)
    max_temp = defaultdict(int)
    min_temp = defaultdict(int)
    outfile = open('output', 'a+')

    for year in directory:
        for subfile in listdir(path_root + year):
            print('YEAR------------------------------------------------')
            print('YEAR------------------------------------------------')
            print(year)
            print('YEAR------------------------------------------------')
            print('YEAR------------------------------------------------')
            print('SUBFILE------------------------------------------------')
            print('SUBFILE------------------------------------------------')
            print(subfile)
            print('SUBFILE------------------------------------------------')
            print('SUBFILE------------------------------------------------')
            lines = context.textFile(path_root + year + '/' + subfile)
            speed = lines.map(lambda word: word[66:70]).collect()
            temp = lines.map(lambda word: word[88:93]).collect()
            speed_max = max(speed)
            speed_min = min(speed)
            temp_max = max(temp)
            temp_min = min(temp)

            max_speed[year] = max(int(speed_max), max_speed[year])
            min_speed[year] = min(int(speed_min), min_speed[year])
            max_temp[year] = max(int(temp_max), max_temp[year])
            min_temp[year] = min(int(temp_min), min_temp[year])

        outfile.write('{} {}'.format(year, max_speed[year]))
        outfile.write('{} {}'.format(year, min_speed[year]))
        outfile.write('{} {}'.format(year, max_temp[year]))
        outfile.write('{} {}'.format(year, min_temp[year]))


    return (max_speed, min_speed, max_temp, min_temp)

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
    data1980 = calculate(sc, dir1980, '80\'s')
    data2000 = calculate(sc, dir2000, '2000\'s')
    sc.stop()

main()
