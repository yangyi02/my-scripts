from __future__ import division
import math


def main():
    n = 100
    x, y = 0, 1
    for i in range(n - 1):
        mid_x = (x + 1) / 2
        mid_y = y / 2
        x = 1 / ((1 + (mid_y / mid_x) ** 2) ** 0.5)
        y = mid_y / mid_x * x
    # length = ((x - 1) ** 2 + y ** 2) ** 0.5
    length = y
    pi = (2 ** n) * length
    print('pi is %.20f' % pi)
    print('pi is %.20f in math pi' % math.pi)

if __name__ == '__main__':
    main()
