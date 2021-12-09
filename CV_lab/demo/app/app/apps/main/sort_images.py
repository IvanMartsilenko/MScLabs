import os
from shutil import copyfile


PATH = r'media/images/'
DATE_PATH = r'media/result/date/'
LOCATION_PATH = r'media/result/location/'
WITHOUT_GROUP_PATH = r'media/result/without/'
CLASS_PATH = r'media/result/classes'

def sort_images(data):
    for image in data.iterrows():
        print(image[1][4], image[1][5])

        if (image[1][4] != -1):
            try:
                copyfile(image[1][0], DATE_PATH + str(image[1][4]) + "/" + image[1][0][12:])
            except IOError as io_err:
                os.makedirs(os.path.dirname(DATE_PATH + str(image[1][4]) + "/" + image[1][0][12:]))
                copyfile(image[1][0], DATE_PATH + str(image[1][4]) + "/" + image[1][0][12:])
        if (image[1][5] != -1):
            try:
                copyfile(image[1][0], LOCATION_PATH + str(image[1][5]) + "/" + image[1][0][12:])
            except IOError as io_err:
                os.makedirs(os.path.dirname(LOCATION_PATH + str(image[1][5]) + "/" + image[1][0][12:]))
                copyfile(image[1][0], LOCATION_PATH + str(image[1][5]) + "/" + image[1][0][12:])
        #if (image[1][6] != -1):
        #    try:
         #       copyfile(image[1][0], CLASS_PATH + str(image[1][6]) + "/" + image[1][0][12:])
         #   except IOError as io_err:
          #      os.makedirs(os.path.dirname(CLASS_PATH + str(image[1][6]) + "/" + image[1][0][12:]))
            #    copyfile(image[1][0], CLASS_PATH + str(image[1][6]) + "/" + image[1][0][12:])
        if (image[1][5] == -1 and image[1][4] == -1): #   and image[1][6] == -1):
            try:
                copyfile(image[1][0], WITHOUT_GROUP_PATH + image[1][0][12:])
            except IOError as io_err:
                os.makedirs(os.path.dirname(WITHOUT_GROUP_PATH + image[1][0][12:]))
                copyfile(image[1][0], WITHOUT_GROUP_PATH + image[1][0][12:])


