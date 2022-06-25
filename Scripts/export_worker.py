import matplotlib.pyplot as plt
import numpy as np


__BINS__ = np.linspace(0,256,256,dtype=int)


def plot_descriptors(x,y,title,x_label,y_label,name,output_path):

    plt.plot(x, y, '+', color='red')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()

    plt.savefig(f'{output_path}/GRAPHS/{name}.png')

    plt.clf()
    plt.close()


def histogram_image(data,title,x_label,y_label,name,output_path):

    data = list(data.ravel())

    plt.hist(data, bins=__BINS__, facecolor='green')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()

    plt.savefig(f'{output_path}/GRAPHS/{name}.png')

    plt.clf()
    plt.close()


def histogram(data,title,bins,name,output_path):

    plt.hist(data, bins=bins, facecolor='blue', rwidth=0.5, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid()

    plt.savefig(f'{output_path}/GRAPHS/{name}.png')

    plt.clf()
    plt.close()


def basic_info(number_of_cells, coverage, output_path):

    file = open(f'{output_path}/CSV_TXT/Information.txt', 'w')

    file.write(f'Number of cells : {int(number_of_cells - 1)}\n')
    file.write(f'Surface coverage : {np.round(coverage,2)} %')

    file.close()


def convert_2D_array_to_1D_list(array):

    list_result = []

    for i in range(array.shape[0]):
        line = f'({array[i][0]} , {array[i][1]})'
        list_result.append(line)

    return list_result


def boundary_with_centroids(img_boundary,centroids):

    img = np.copy(img_boundary)

    for i in range(1,centroids.shape[0]):

        x = int(centroids[i, 0])
        y = int(centroids[i, 1])

        img[y,x] = i

    return img


def boundary_to_original_image(img,img_boundary,width,height, color=None):

    if color is None:
        color = [255, 255, 255]

    img_original_with_boundary = np.copy(img)

    for i in range(height):
        for j in range(width):
            if img_boundary[i][j] != 0:
                img_original_with_boundary[i][j][0] = color[0]
                img_original_with_boundary[i][j][1] = color[1]
                img_original_with_boundary[i][j][2] = color[2]

    return img_original_with_boundary


if __name__ == "__main__":

    print('Hello, home!')