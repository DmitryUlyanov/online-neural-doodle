from sklearn.cluster import KMeans
import scipy
import numpy as np
import h5py
import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--colors', help='Array with color mapping.')
parser.add_argument('--model', help='Path to model.t7.')
parser.add_argument('--target_mask', help='Path to target mask.')
parser.add_argument(
    '--out_path', default='out.png', help='Where to store rendered image.')
parser.add_argument(
    '--temp_hdf5', default='temp.hdf5', help='Where to temporary hdf5 file.')

args = parser.parse_args()


def convert_image(target_mask_path, centroids):
    num_colors = len(centroids)
    kmeans = KMeans(n_clusters=num_colors, random_state=0)
    kmeans.cluster_centers_ = centroids

    mask_target = scipy.misc.imread(target_mask_path)[:, :, :3]
    target_shape = mask_target.shape

    target_flatten = mask_target.reshape(target_shape[0] * target_shape[1], -1)

    labels_target = kmeans.predict(target_flatten.astype(float))
    target_kval = labels_target.reshape(target_shape[0], target_shape[1])

    result_mask = np.zeros(
        (num_colors, mask_target.shape[0], mask_target.shape[1]))
    for i in range(num_colors):
        result_mask[i] = (target_kval == i).astype(float)

    return result_mask

centroids = np.load(args.colors)
mask = convert_image(args.target_mask, centroids)

f = h5py.File(args.temp_hdf5, 'w')
f['mask'] = mask
f.close()

cmd = 'th apply.lua -model %s -mask_hdf5 %s -out_path %s' % (
    args.model, args.temp_hdf5, args.out_path)

subprocess.call(cmd, shell=True)
