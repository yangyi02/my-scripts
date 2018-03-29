import argparse
import re
import matplotlib.pyplot as plt


def parse_log(log_file, x_coord, y_coord):
    lines = open(log_file).readlines()

    regex_x_coord = re.compile(x_coord + r': (\d+)')
    regex_y_coord = re.compile(y_coord + r': (\d+)')

    x_coords = []
    y_coords = []
    for line in lines:
        x_coord_match = regex_x_coord.search(line)
        if x_coord_match:
            x_coord = float(x_coord_match.group(1))

        y_coord_match = regex_y_coord.search(line)
        if y_coord_match:
            y_coord = float(y_coord_match.group(1))

        if x_coord_match and y_coord_match:
            print x_coord, y_coord
            x_coords.append(x_coord)
            y_coords.append(y_coord)

    return x_coords, y_coords

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='dqn_log_2016-08-29:20:15')
    parser.add_argument('--x_coord', default='frames')
    parser.add_argument('--y_coord', default='reward')
    parser.add_argument('--save_file', default='curve.jpg')

    args = parser.parse_args()
    log_file = args.log_file
    x_coord = args.x_coord
    y_coord = args.y_coord

    x_coords, y_coords = parse_log(log_file, x_coord, y_coord)
    plt.plot(x_coords, y_coords)
    plt.xlabel(x_coord)
    plt.ylabel(y_coord)
    plt.savefig(args.save_file)
