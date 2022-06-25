import math


def get_compactness(area, perimeter):

    info_compactness = (4 * math.pi * area[1:]) / ((perimeter[1:]) ** 2)

    return info_compactness


def get_rectangularity(major_axis_length, minor_axis_length, area):

    info_rectangularity = area[1:] / (major_axis_length[1:] * minor_axis_length[1:])

    return info_rectangularity


def get_eccentricity(major_axis_length, minor_axis_length):

    info_eccentricity = minor_axis_length[1:] / major_axis_length[1:]

    return info_eccentricity


def get_elongation(bounding_box_width, bounding_box_height):

    info_elongation = bounding_box_width[1:] / bounding_box_height[1:]

    return info_elongation


def get_roundness(area, convex_hull_perimeter):

    info_roundness = (4 * math.pi * area[1:]) / (convex_hull_perimeter[1:]) ** 2

    return info_roundness


def get_convexity(perimeter, convex_hull_perimeter):

    info_convexity = convex_hull_perimeter[1:] / perimeter[1:]

    return info_convexity


def get_solidity(area, convex_hull_area):

    info_solidity = area[1:] / convex_hull_area[1:]

    return info_solidity


def get_curl(major_axis_length, perimeter, area):

    fibre_length = (perimeter[1:] - (perimeter[1:] ** 2 - 16 * area[1:]) ** (1 / 2)) / 4

    info_curl = major_axis_length[1:] / fibre_length[1:]

    return info_curl, fibre_length


def get_sphericity(r_inscribing, r_circumscribing):

    info_sphericity = r_inscribing[1:] / r_circumscribing[1:]

    return info_sphericity


if __name__ == "__main__":

    print('Hello home')