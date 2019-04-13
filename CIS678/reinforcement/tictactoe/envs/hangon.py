#!/usr/bin/env python3

if __name__ == '__main__':
    def func(thing, value):
        if value in thing:
            thing[value] += 1
        else:
            thing[value] = 1

    thing = {}
    func(thing, 'oh')
    print(thing)
    func(thing, 'ohyeah')
    print(thing)
