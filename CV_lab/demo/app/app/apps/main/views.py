from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.shortcuts import render, redirect
from .forms import *
from django.urls import reverse
from .exif_sorter import get_images_paths, get_meta, add_clusters_by_date, add_clusters_by_gps, add_class
from .clarity import clusters_by_clarity
from .sort_images import sort_images
from .duplicate import compare_images
from .landmark import sort_landmark
import shutil

def main(request):
    return render(request, 'main/main.html')


def add_images(request):
    if request.method == 'POST':
        images = request.FILES.getlist('images')

        for image in images:
            new_photo = Photo(image=image)
            new_photo.name = new_photo.image.name
            new_photo.save()

        return HttpResponseRedirect(reverse('main:display_images'))

    return render(request, 'main/add_images.html')


def display_images(request):
    if request.method == 'GET':
        images = Photo.objects.all()
        return render(request, 'main/display_images.html', {'imgs': images})

    if request.method == 'POST':
        try:
            shutil.rmtree(r'media/result')
        except: print("not such dir")
        #sort_images(add_clusters_by_gps((add_clusters_by_date(get_meta(get_images_paths())))))
        #clusters_by_clarity()
        #compare_images()
        sort_landmark()
        
        directory = r'media/result'
        directories, files = get_paths(directory=directory)

        directories = remove_extra_paths(directory=directory)

        print(directories)
        print(files)
        zip_name = zip_sorted_directories(directory=directory)

        return render(request, 'main/display_sorted_images.html', {'directories': directories,
                                                                   'files': files,
                                                                   'zip_name': zip_name})


def delete_image(request, img_id):
    cur_img = None

    try:
        cur_img = Photo.objects.get(id=img_id)
    except:
        raise Http404('ERROR item_delete')

    cur_img.delete()

    return HttpResponseRedirect(reverse('main:display_images'))


def download(request):
    zipFileName = 'sorted_photos.zip'
    content_type = "application/octet-stream"
    response = HttpResponse(zipFileName, content_type=content_type)

    response["Content-Disposition"] = "attachment; filename=" + str(zipFileName)
    return response


def get_paths(directory):
    import os, sys

    directories_m = []
    files_m = []

    for root, subdirectories, files in os.walk(directory):
        for subdirectory in subdirectories:
            directories_m.append(os.path.join(root, subdirectory))
        for file in files:
            files_m.append(os.path.join(root, file))

    return directories_m, files_m


def remove_extra_paths(directory):
    import os, sys

    tmp = []
    new_d = []

    for root, subdirectories, files in os.walk(directory):
        for subdirectory in subdirectories:
            tmp.append(os.path.join(root, subdirectory))

    for el in tmp:
        print(f'now see --> {el}')
        if not check_if_sub_path(el, tmp):
            new_d.append(el)

    return new_d


def check_if_sub_path(el, arr):
    for a in arr:
        if el in a:
            print(f'      {el} was found as subdirectory')
            return True
        return False


def zip_sorted_directories(directory):
    from zipfile import ZipFile
    import os
    from os.path import basename

    zipFileName = 'sorted_photos.zip'
    with ZipFile(zipFileName, 'w') as zipObj:
        for folderName, subfolders, filenames in os.walk(directory):
            for filename in filenames:
                filePath = os.path.join(folderName, filename)
                zipObj.write(filePath, basename(filePath))

    return zipFileName
