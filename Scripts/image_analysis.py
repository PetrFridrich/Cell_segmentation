from skimage.morphology import convex_hull_image
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
import mahotas as mh
import pandas as pd
import numpy as np

import math
import cv2

import shape_descriptors as sd
import export_worker as ew


def analysis_process(img, img_labeled,output_path,descriptor_mask):

    #  Rozměry segmentovaného snímku
    width = img_labeled.shape[1]
    height = img_labeled.shape[0]

    #  Velikosti buněk
    area = mh.labeled.labeled_size(img_labeled)
    area[0] = 0  # Pozadí nastavený na 0

    #  Počet buněk + 1 (0-tý index je pozadí)
    number_of_cells = area.shape[0]

    #  Pokrytí snímku v %
    coverage = (np.sum(img_labeled != 0) / (width * height)) * 100

    #  Snímek hranic buněk a jejich obvod
    img_labeled_boundary = get_boundary_4_connected(img_labeled,width,height)
    perimeter = mh.labeled.labeled_size(img_labeled_boundary)
    perimeter[0] = 0  # Pozadí nastavený na 0

    #  Matice souřadnic hranic jednotlivých buněk
    coordinates_boundary = get_coordinates_of_pixels(img_labeled_boundary,perimeter,number_of_cells,width,height)

    #  Informace o hlavní poloose
    major_axis_vector,img_of_vectors = get_major_axis_vector(coordinates_boundary,perimeter,number_of_cells,width,height)
    major_axis_angle = get_major_axis_angle(major_axis_vector,number_of_cells)
    major_axis_length = get_major_axis_length(major_axis_vector)

    #  Pro snažší zjištění vedlejší poloosy buňky rotujeme dle hlavní poloosy
    coordinates_rotated_boundary,img_rotated_cells = get_coordinates_of_rotated_cells(coordinates_boundary,major_axis_angle,perimeter,number_of_cells)

    #  Délka vedlejší poloosy
    minor_axis_length = get_minor_axis_length(coordinates_rotated_boundary,perimeter,number_of_cells)

    #  Convex information
    convex_hull_area, convex_hull_perimeter, img_boundary_convex = get_convex_hull_info(img_labeled,number_of_cells)

    #  Těžiště buněk
    centroids = get_centroids(coordinates_boundary,perimeter,number_of_cells)

    #  Poloměr kružnice opsané a vepsané
    R_inscribing, R_circumscribing = get_R_inner_and_outer(coordinates_boundary, perimeter, centroids, number_of_cells)

    #  DataFrame obecné charakteristiky
    id_cell = {'Cell id': np.linspace(1, number_of_cells-1, number_of_cells-1)}
    df_info = pd.DataFrame(id_cell)

    df_info['Area'] = area[1:]
    df_info['Perimeter'] = perimeter[1:]
    df_info['R_inscribing'] = R_inscribing[1:]
    df_info['R_circumscribing'] = R_circumscribing[1:]
    df_info['Convex hull area'] = convex_hull_area[1:]
    df_info['Convex hull perimeter'] = convex_hull_perimeter[1:]
    df_info['Major axis length'] = major_axis_length[1:]
    df_info['Minor axis length'] = minor_axis_length[1:]
    df_info['Centroids'] = ew.convert_2D_array_to_1D_list(centroids)[1:]


    #  DataFrame shape descriptors
    df_shape_des = pd.DataFrame(id_cell)

    if descriptor_mask[0]:
        compactness = sd.get_compactness(area, perimeter)
        df_shape_des['Compactness'] = compactness
        ew.histogram(compactness,'Compactness', np.linspace(0,1,6),'Compactness',output_path)

    if descriptor_mask[1]:
        pass
        #rectangularity = sd.get_rectangularity(major_axis_length, minor_axis_length, area)
        #df_shape_des['Rectangularity'] = rectangularity
        #ew.histogram(rectangularity, 'Rectangularity', np.linspace(0, 1, 6), 'Rectangularity', output_path)

    if descriptor_mask[2]:
        eccentricity = sd.get_eccentricity(major_axis_length, minor_axis_length)
        df_shape_des['Eccentricity'] = eccentricity
        ew.histogram(eccentricity, 'Eccentricity', np.linspace(0, 1, 6), 'Eccentricity', output_path)

    if descriptor_mask[3]:
        pass
        #elongation = sd.get_elongation()
        #df_shape_des['Elongation'] = elongation
        #ew.histogram(elongation, 'Elongation', np.linspace(0, 1, 6), 'Elongation', output_path)

    if descriptor_mask[4]:
        roundness = sd.get_roundness(area, convex_hull_perimeter)
        df_shape_des['Roundness'] = roundness
        ew.histogram(roundness, 'Roundness', np.linspace(0, 1, 6), 'Roundness', output_path)

    if descriptor_mask[5]:
        convexity = sd.get_convexity(perimeter, convex_hull_perimeter)
        df_shape_des['Convexity'] = convexity
        ew.histogram(convexity, 'Convexity', np.linspace(0, 1, 6), 'Convexity', output_path)

    if descriptor_mask[6]:
        solidity = sd.get_solidity(area, convex_hull_area)
        df_shape_des['Solidity'] = solidity
        ew.histogram(solidity, 'Solidity', np.linspace(0, 1, 6), 'Solidity', output_path)

    if descriptor_mask[7]:
        curl, fibre_length = sd.get_curl(major_axis_length, perimeter, area)
        df_shape_des['Curl'] = curl
        ew.histogram(curl, 'Curl', np.linspace(0, 1, 6), 'Curl', output_path)

    if descriptor_mask[8]:
        sphericity = sd.get_sphericity(R_inscribing, R_circumscribing)
        df_shape_des['Sphericity'] = sphericity
        ew.histogram(sphericity, 'Sphericity', np.linspace(0, 1, 6), 'Sphericity', output_path)


    #  -----------  Export dat  --------------------

    #  DataFrame
    df_info.to_csv(f'{output_path}CSV_TXT/Obecné charakteristiky.csv', index=False, header=True)
    df_shape_des.to_csv(f'{output_path}CSV_TXT/Tvarové deskriptory.csv', index=False, header=True)

    #  Vztah tvarových deskriptorů
    ew.plot_descriptors(df_shape_des['Roundness'],df_shape_des['Sphericity'],'Shape descriptor','roundness','sphericity','11_roundness-sphericity',output_path)
    ew.plot_descriptors(df_shape_des['Compactness'], df_shape_des['Eccentricity'], 'Shape descriptor', 'compactness', 'eccentricity', '12_compactness-eccentricity', output_path)

    #  Spojení hranic a těžišť v jeden snímek
    img_boundary_with_centroids = ew.boundary_with_centroids(img_labeled_boundary, centroids)
    plt.imsave(f'{output_path}IMG/50_Boundary_centroids.png', img_boundary_with_centroids, cmap='jet')

    #  Snímek rotovaných buněk
    plt.imsave(f'{output_path}IMG/51_Rotated_cells.png', img_rotated_cells, cmap='jet')

    #  Snímek major axis
    img_of_vectors.save(f'{output_path}IMG/52_Major_axis.png')

    #  Convexní hranice buněk
    plt.imsave(f'{output_path}IMG/53_Convex_boundary.png', img_boundary_convex, cmap='jet')

    #  Pokrytí a počet buněk
    ew.basic_info(number_of_cells,coverage,output_path)

    #  Hranice v originálním obrázku
    img_original_with_boundary = ew.boundary_to_original_image(img,img_labeled_boundary,width,height)
    plt.imsave(f'{output_path}IMG/54_Boundary_in_original_image.png', img_original_with_boundary)

    #  Histogramy RGB
    ew.histogram_image(img[:, :, 0], 'Red_channel', 'Value', 'Frequency', '01_Red_channel_histogram', output_path)
    ew.histogram_image(img[:, :, 1], 'Green_channel', 'Value', 'Frequency', '02_Green_channel_histogram', output_path)
    ew.histogram_image(img[:, :, 2], 'Blue_channel', 'Value', 'Frequency', '03_Blue_channel_histogram', output_path)

    #  Histogram Velikostí buněk
    ew.histogram(area[1:], 'Velikosti buněk', 20, '21_Area_of_cells', output_path)

    #  Histogram obvodů buněk
    ew.histogram(perimeter[1:], 'Velikosti obvodů buněk', 20, '22_Perimeter_of_cells', output_path)


def get_major_axis_vector(coordinates_of_boundary_pixels, boundary_sizes, number_of_cells, width, height):

    major_axis_vector = np.zeros((number_of_cells, 2), dtype=int)

    # Taková kontrola, budu si ukládat počáteční a koncový body
    points_of_major_axis = np.zeros((number_of_cells, 4), dtype=int)

    for i in range(1, number_of_cells):

        distance_max = 0

        for k in range(0, int(boundary_sizes[i] * 2), 2):
            for l in range(k + 2, int(boundary_sizes[i] * 2), 2):

                x1 = coordinates_of_boundary_pixels[i][k]
                y1 = coordinates_of_boundary_pixels[i][k + 1]

                x2 = coordinates_of_boundary_pixels[i][l]
                y2 = coordinates_of_boundary_pixels[i][l + 1]

                distance = (x2 - x1) ** 2 + (y2 - y1) ** 2

                if distance > distance_max:
                    distance_max = distance
                    major_axis_vector[i][0] = x2 - x1
                    major_axis_vector[i][1] = y1 - y2

                    points_of_major_axis[i][0] = x1
                    points_of_major_axis[i][1] = y1
                    points_of_major_axis[i][2] = x2
                    points_of_major_axis[i][3] = y2

        if major_axis_vector[i][0] < 0:
            major_axis_vector[i][0] = major_axis_vector[i][0] * (-1)
            major_axis_vector[i][1] = major_axis_vector[i][1] * (-1)

    # '''
    # ----------------------------------------------------------------------- #
    img_major_axis_vector = Image.new('L', (width, height), 'black')
    d_bmp = ImageDraw.Draw(img_major_axis_vector)

    for i in range(1, number_of_cells):
        x1 = points_of_major_axis[i][0]
        y1 = points_of_major_axis[i][1]

        x2 = points_of_major_axis[i][2]
        y2 = points_of_major_axis[i][3]

        d_bmp.line((x1, y1, x2, y2), fill='white')
    # ----------------------------------------------------------------------- #
    # '''

    return major_axis_vector, img_major_axis_vector


def get_major_axis_angle(major_axis_vector, number_of_cells):

    major_axis_angle = np.zeros(number_of_cells)

    for i in range(1, number_of_cells):

        if major_axis_vector[i][1] == 0:
            continue

        if major_axis_vector[i][0] == 0:
            if major_axis_vector[i][1] < 0:
                major_axis_angle = - (math.pi / 2)
            else:
                major_axis_angle = math.pi / 2

            continue

        ratio = major_axis_vector[i][1] / major_axis_vector[i][0]

        major_axis_angle[i] = np.arctan(ratio)

    return major_axis_angle


def get_major_axis_length(major_axis_vector):

    major_axis_length = (major_axis_vector[:,0] ** 2 + major_axis_vector[:,1] ** 2) ** (1 / 2)
    major_axis_length = np.round(major_axis_length,1)

    return major_axis_length


def get_coordinates_of_rotated_cells(coordinates_of_boundary_pixels, major_axis_angle, boundary_sizes, number_of_cells):

    rotated_coordinates = np.zeros(coordinates_of_boundary_pixels.shape, dtype=int)

    for i in range(1, number_of_cells):
        for j in range(0, int(boundary_sizes[i] * 2), 2):

            x = coordinates_of_boundary_pixels[i][j]
            y = coordinates_of_boundary_pixels[i][j + 1]

            alpha = major_axis_angle[i]

            rotated_coordinates[i][j] = int(x * math.cos(alpha) - y * math.sin(alpha))
            rotated_coordinates[i][j + 1] = int(x * math.sin(alpha) + y * math.cos(alpha))

    # '''
    # ---------------------------------------------------------------------------------------------------------------- #
    minimum = np.amin(rotated_coordinates)
    maximum = np.amax(rotated_coordinates)

    shape_of_new_img = maximum - minimum + 1
    shift = 0 - minimum

    img_rotated_cells = np.zeros((shape_of_new_img, shape_of_new_img))

    for i in range(1, number_of_cells):
        for j in range(0, int(boundary_sizes[i] * 2), 2):
            x = rotated_coordinates[i][j] + shift
            y = rotated_coordinates[i][j + 1] + shift

            img_rotated_cells[y][x] = i

    # ---------------------------------------------------------------------------------------------------------------- #
    # '''

    return rotated_coordinates, img_rotated_cells


def get_minor_axis_length(rotated_coordinates_of_boundary_pixels, boundary_sizes, number_of_cells):

    minor_axis_length = np.zeros(number_of_cells)  # jen  vzdálenost, nic víc pro analýzu tvarů nepotřebuji

    for i in range(1, number_of_cells):

        distance_max = 0

        for k in range(0, int(boundary_sizes[i] * 2), 2):
            for l in range(0, int(boundary_sizes[i] * 2), 2):

                if rotated_coordinates_of_boundary_pixels[i][k] == rotated_coordinates_of_boundary_pixels[i][l]:
                    current_distance = abs(rotated_coordinates_of_boundary_pixels[i][k + 1] - rotated_coordinates_of_boundary_pixels[i][l + 1])

                    if current_distance > distance_max:
                        distance_max = current_distance
                        minor_axis_length[i] = current_distance

    return minor_axis_length


def get_convex_hull_info(img_labeled, number_of_cells):

    convex_hull_area = np.zeros(number_of_cells, dtype=int)
    convex_hull_perimeter = np.zeros(number_of_cells, dtype=int)
    img_boundary = np.zeros_like(img_labeled)

    for i in range(1, number_of_cells):

        single_cell = (img_labeled == i).astype(int)
        img_convex_hull_boundary = np.zeros(img_labeled.shape)

        img_convex_hull_area = convex_hull_image(single_cell).astype(np.uint8)
        convex_hull_area[i] = np.sum(img_convex_hull_area)

        contours, hierarchy = cv2.findContours(img_convex_hull_area, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_convex_boundary = cv2.drawContours(img_convex_hull_boundary, contours, -1, 1, 1,lineType=cv2.LINE_4)
        convex_hull_perimeter[i] = np.sum(img_convex_boundary)

        # ---------------------------------------------------------------------#
        img_boundary = img_boundary + (img_convex_hull_boundary * i)
        img_boundary[img_boundary > i] = i
        # ---------------------------------------------------------------------#

    return convex_hull_area, convex_hull_perimeter, img_boundary


def get_boundary_4_connected(img_labeled, width, height):

    img_boundary = np.zeros((height, width), dtype=int)

    for i in range(height):
        for j in range(width):

            if img_labeled[i][j] != 0:
                value = img_labeled[i][j]

                if i < height - 1 and j < width - 1:
                    if img_labeled[i + 1][j + 1] != value:
                        img_boundary[i][j] = value
                        continue

                if i > 0 and j < width - 1:
                    if img_labeled[i - 1][j + 1] != value:
                        img_boundary[i][j] = value
                        continue

                if i < height - 1 and j > 0:
                    if img_labeled[i + 1][j - 1] != value:
                        img_boundary[i][j] = value
                        continue

                if i > 0 and j > 0:
                    if img_labeled[i - 1][j - 1] != value:
                        img_boundary[i][j] = value
                        continue

                if j < width - 1:
                    if img_labeled[i][j + 1] != value:
                        img_boundary[i][j] = value
                        continue

                if j > 0:
                    if img_labeled[i][j - 1] != value:
                        img_boundary[i][j] = value
                        continue

                if i < height - 1:
                    if img_labeled[i + 1][j] != value:
                        img_boundary[i][j] = value
                        continue

                if i > 0:
                    if img_labeled[i - 1][j] != value:
                        img_boundary[i][j] = value
                        continue


    return img_boundary


def get_centroids(coordinates_of_boundary_pixels, perimeter, number_of_cells):

    centroids = np.zeros((number_of_cells, 2))

    for i in range(1, number_of_cells):

        for j in range(0, int(perimeter[i] * 2), 2):
            centroids[i][0] += coordinates_of_boundary_pixels[i][j]

        for j in range(1, int(perimeter[i] * 2), 2):
            centroids[i][1] += coordinates_of_boundary_pixels[i][j]

        centroids[i][0] = int(centroids[i][0] / perimeter[i])
        centroids[i][1] = int(centroids[i][1] / perimeter[i])

    return centroids


def get_coordinates_of_pixels(img_labeled, sizes, number_of_cells, width, height):

    # Informace o potřebných rozměrech matice
    matrix_height = number_of_cells
    matrix_width = (int(np.amax(sizes) * 2))

    # Matice souřadnic pixelů jenotlivých buněk
    matrix_coordinates = np.zeros((matrix_height, matrix_width), dtype=int)

    # Matice pro uložení počtu souřadnic které jsem již použil
    matrix_shifts = np.zeros(number_of_cells)

    for i in range(height):
        for j in range(width):

            if img_labeled[i][j] == 0: continue

            cell_index = int(img_labeled[i][j])
            shift = int(matrix_shifts[cell_index])

            matrix_coordinates[cell_index][shift] = j
            matrix_coordinates[cell_index][shift + 1] = i

            matrix_shifts[cell_index] += 2

    return matrix_coordinates


def get_R_inner_and_outer(coordinates_of_boundary_pixels, perimeter, centroids, number_of_cells):

    r_inner = np.full(number_of_cells, math.inf)
    r_outer = np.zeros(number_of_cells)

    for i in range(1, number_of_cells):

        for k in range(0, int(perimeter[i] * 2), 2):

            r = ((centroids[i][0] - coordinates_of_boundary_pixels[i][k]) ** 2 + (centroids[i][1] - coordinates_of_boundary_pixels[i][k + 1]) ** 2) ** (1 / 2)

            if r > r_outer[i]:
                r_outer[i] = r

            if r < r_inner[i]:
                r_inner[i] = r

    return r_inner, r_outer


if __name__ == "__main__":

    print('Hello home')