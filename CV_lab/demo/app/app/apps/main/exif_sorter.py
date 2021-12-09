import datetime
import pandas as pd
import warnings
import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
from PIL import Image, ExifTags
import re
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


warnings.filterwarnings('ignore')

ONE_DAY_IN_SEC = (datetime.datetime(1970, 1, 2) - datetime.datetime(1970, 1, 1)).total_seconds()
GPS_DISTANCE = 0.2

PATH = r'media/images/'
COMPILED_PATTERN_FILE = re.compile(r"\w*.(png|jpg|JPG|jpeg)")
MODEL = ResNet50(weights='imagenet')


def get_images_paths():
	files_in_folder = []
	for file in os.listdir(PATH):
		if COMPILED_PATTERN_FILE.fullmatch(file) is not None:
			files_in_folder.append(os.path.join(PATH, file))
	return files_in_folder


def get_meta(images_paths):
	"""
	Read images from list, return this paths with meta data (exif)

	:param images_paths: list with str paths to images
	:return: dataframe with columns: path, date, 		longitude, latitude
							types:   str,  datetime,	float, 	   float
	"""
	pil_images = _get_pil_images(images_paths)
	path_with_meta = _get_image_meta(pil_images)
	_close_pil_images(pil_images)

	return pd.DataFrame(
		data=path_with_meta,
		columns=["path", "date", "longitude", "latitude"])


def add_clusters_by_gps(data):
	"""
	Clusters images by geolocation (longitude and latitude).
	Samples without geolocation and anomalies are marked as -1

	:param data: dataframe with columns longitude and latitude that contains floating point objects
	:return: copy of this dataframe with column (int) clusters_by_gps
	"""
	without_na = data.dropna(subset=["longitude", "latitude"])
	input_data = without_na[["longitude", "latitude"]]

	dbscan = DBSCAN(eps=GPS_DISTANCE * 3, min_samples=2)
	dbscan.fit(input_data)
	without_na["clusters_by_gps"] = dbscan.labels_

	with_na = data[data["longitude"].isna() | data["latitude"].isna()]
	with_na["clusters_by_gps"] = -1

	output_data = pd.concat([with_na, without_na])
	return output_data


def add_clusters_by_date(data):
	"""
	Clusters images by date.
	Samples without date and anomalies are marked as -1

	:param data: dataframe with column date, that contains datetime objects
	:return: copy of this dataframe with column (int) clusters_by_date
	"""
	without_na = data.dropna(subset=["date"])

	timestamps = [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in without_na["date"]]
	without_na["timestamp"] = timestamps
	without_na["temp"] = 0
	input_data = without_na[["timestamp", "temp"]]

	dbscan = DBSCAN(eps=ONE_DAY_IN_SEC * 3, min_samples=3)
	dbscan.fit(input_data)
	without_na["clusters_by_date"] = dbscan.labels_
	without_na.drop(["temp", "timestamp"], axis=1, inplace=True)

	with_na = data[data["date"].isna()]
	with_na["clusters_by_date"] = -1

	output_data = pd.concat([with_na, without_na])

	return output_data


def _get_image_meta(bin_images):
	res = []
	for image in bin_images:
		raw_exif = image._getexif()

		if raw_exif is None:
			continue

		exif = {
			ExifTags.TAGS[k]: v
			for k, v in raw_exif.items()
			if k in ExifTags.TAGS
		}

		date = _get_exif_with_preparing(exif, "DateTime", _prepare_date)
		lonlat = _get_exif_with_preparing(exif, "GPSInfo", _prepare_gps)

		if lonlat is not None:
			longitude, latitude = lonlat[0], lonlat[1]
		else:
			longitude, latitude = None, None

		res.append((image.filename, date, longitude, latitude))

	return res


def _get_exif_with_preparing(exif, tag, preparing_function):
	data = exif.get(tag, None)

	if data is not None:
		data = preparing_function(data)

	return data


def _prepare_date(str_date):
	return datetime.datetime.strptime(str_date, "%Y:%m:%d %H:%M:%S")


def _prepare_gps(data):
	lon = data.get(2)
	lat = data.get(4)

	if lon is None or lat is None:
		return None, None

	to_degree = lambda coords: float(coords[0]) + (coords[1] / 60) + (coords[2] / 3600)
	return to_degree(lon), to_degree(lat)


def _get_pil_images(images_paths) :
	bin_images = []
	for path in images_paths:
		bin_images.append(Image.open(path))

	return bin_images


def _close_pil_images(bin_images):
	for image in bin_images:
		image.close()

def add_class(data):
	res = []
	for path in data["path"]:
		x = cv2.imread(path, cv2.COLOR_BGR2RGB)
		x = cv2.resize(x, (224, 224))
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		preds = MODEL.predict(x)
		res.append(decode_predictions(preds, top=1)[0][0][1])

	res_data = data.copy()
	res_data["class"] = res
	return res_data