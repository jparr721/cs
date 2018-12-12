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
        lines = context.textFile(path_root + year + '/*')
        speed = lines.map(lambda word: float(word[65:69]) / 10).collect()
        temp = lines.map(lambda word: float(word[87:92]) / 10).collect()
        speed_list = [s for s in speed if s < 300]
        temp_list = [t for t in temp if t <= 300]
        speed_max = max(speed_list)
        speed_min = min(speed_list)
        temp_max = max(temp_list)
        temp_min = min(temp_list)
        temp_avg = sum(temp_list)/float(len(temp_list))

        max_speed[year] = max(speed_max, max_speed[year])
        min_speed[year] = min(speed_min, min_speed[year])
        max_temp[year] = max(temp_max, max_temp[year])
        min_temp[year] = min(temp_min, min_temp[year])

        outfile.write('{} Max Speed: {}\n'.format(year, max_speed[year]))
        outfile.write('{} Min Speed: {}\n'.format(year, min_speed[year]))
        outfile.write('{} Max Temp: {}\n'.format(year, max_temp[year]))
        outfile.write('{} Min Temp: {}\n'.format(year, min_temp[year]))
        outfile.write('{} Average Temp: {}\n'.format(year, temp_avg))


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
        except ValueError:
            print('not a number, skipping: {}'.format(f))

        if type(year) is int and year >= 1980 and year <= 1989:
            dir1980.append(f)
        elif type(year) is int and year >= 2000 \
                and year <= 2009 and year != 2004:
            dir2000.append(f)
        else:
            print('out of range, skipping: {}'.format(f))

    sc = SparkContext(appName="CleanCoal")
    calculate(sc, dir1980, '80\'s')
    calculate(sc, dir2000, '2000\'s')
    sc.stop()


main()
