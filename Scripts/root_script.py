import datetime
import cv2
import os

import image_manipulation as im
import image_analysis as ia


def get_names_from_directory(base_path):

    images = []

    for entry in os.listdir(base_path):
        if os.path.isfile(os.path.join(base_path, entry)):
            images.append(entry)

    return images


def create_directories_for_results(path, list_of_input_data, note):

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if note is None:
        path = f'{path}A_{time}/'
    else :
        path = f'{path}A_{time}_{note}/'

    #  vytvoření složky pro výstup zpracování obrazu
    os.mkdir(path)

    for index, name in enumerate(list_of_input_data):

        #  vytvoření složky pro snímek
        n_path = f'{path}{name}/'
        os.mkdir(n_path)

        #  vytvoření složek pro jednotlivé výstupy snímku
        os.mkdir(f'{n_path}IMG/')
        os.mkdir(f'{n_path}GRAPHS/')
        os.mkdir(f'{n_path}CSV_TXT/')

    return path


def image_processing(data_path, output_path, note=None):

    print('Processing of all images just started')

    #  Načtu do listu jména snímků ve složce
    try:
        list_of_input_data = get_names_from_directory(data_path)
    except:
        print('Something wrong with input path')
        return

    #  Vytvoření složek, kde budou výsledky
    try:
        default_output_path = create_directories_for_results(output_path,list_of_input_data,note)
    except:
        print('Something wrong with output path')
        return

    #  Procházím všechny snímky a zpracovávám je
    for index, name in enumerate(list_of_input_data):

        output_path = default_output_path + f'{name}/'

        try:
            img = cv2.imread(data_path + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print('Something wrong with input data')
            continue

        print(f'\tImage manipulation - {index+1}')

        #  Manipulace se snímkem, očekávám segmentovaný snímek
        img_1, img_2 = im.segmentation_process(img,output_path)

        print(f'\tImage analysis - {index+1}')

        #  Analýza segmentovaného snímku
        descriptor_mask = [True, False, True, False, True, True, True, False, True]
        ia.analysis_process(img, img_2,output_path,descriptor_mask)

    print('Processing of all images just finished')


if __name__ == "__main__":

    folder_name = 'Images'

    t0 = datetime.datetime.now()

    DATA_PATH = f'../Data/{folder_name}/'
    OUTPUT_PATH = f'../Results/'
    NOTE = f'{folder_name}'

    image_processing(DATA_PATH, OUTPUT_PATH, NOTE)

    t1 = datetime.datetime.now()
    print(str(t1 - t0))