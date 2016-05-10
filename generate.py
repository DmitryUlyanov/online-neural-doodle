import numpy as np
import h5py
import argparse
import scipy.misc
from joblib import Parallel, delayed

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

from skimage.transform import resize
from skimage.morphology import disk
from skimage.filters.rank import median

import diamond as DS

parser = argparse.ArgumentParser()

parser.add_argument(
    '--n_colors', type=int, help='How many distinct colors does mask have.')
parser.add_argument('--style_image', help='Path to style image.')
parser.add_argument('--style_mask', help='Path to mask for style.')
parser.add_argument(
    '--out_hdf5', default='gen_doodles.hdf5', help='Where to store hdf5 file.')
parser.add_argument(
    '--n_jobs', type=int, default=4, help='Number of worker threads.')

args = parser.parse_args()

style_img = args.style_image
style_mask_img = args.style_mask
n_colors = args.n_colors

# get shape
im = scipy.misc.imread(style_img)
dims = (im.shape[0], im.shape[1])


def generate():
    np.random.seed(None)
    ohe = OneHotEncoder(sparse=False)

    hmap = np.array(DS.diamond_square((200, 200), -1, 1, 0.35))
    + np.array(DS.diamond_square((200, 200), -1, 1, 0.55))
    + np.array(DS.diamond_square((200, 200), -1, 1, 0.75))

    hmap_flatten = np.array(hmap).ravel()[:, None]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(hmap_flatten)
    labels_hmap = kmeans.predict(hmap_flatten)[:, None]

    # Back to rectangular
    labels_hmap = labels_hmap.reshape([hmap.shape[0], hmap.shape[1]])
    labels_hmap = median(labels_hmap.astype(np.uint8), disk(5))
    labels_hmap = resize(labels_hmap, dims, order=0, preserve_range=True)

    labels_hmap = ohe.fit_transform(labels_hmap.ravel()[:, None])

    # Reshape
    hmap_masks = labels_hmap.reshape([dims[0], dims[1], n_colors])
    hmap_masks = hmap_masks.transpose([2, 0, 1])

    return hmap_masks


# Generate doodles
f = h5py.File(args.out_hdf5, 'w')
gen_masks = Parallel(n_jobs=args.n_jobs)(delayed(generate)()
                                         for i in range(1000))

# Save
for i, mask in enumerate(gen_masks):
    f['train_mask_%d' % i] = mask

f['n_train'] = np.array([len(gen_masks)])  # yeah, 1x1 array..

ohe = OneHotEncoder(sparse=False)


# get style mask
# IMAGE
img_style = scipy.misc.imread(style_img)[..., :3]
img_style = scipy.misc.imresize(img_style, dims)
f['style_img'] = img_style.transpose(2, 0, 1).astype(float) / 255.

# MASK
img_style_mask = scipy.misc.imread(style_mask_img)[..., :3]
img_style_mask = scipy.misc.imresize(img_style_mask, dims)

style_mask_flatten = img_style_mask.reshape(
    img_style_mask.shape[0] * img_style_mask.shape[1], -1)
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(style_mask_flatten)

# Get labels
labels_target = kmeans.predict(style_mask_flatten)[:, None]
labels_target = ohe.fit_transform(labels_target)

# Get the right shape
masks = labels_target.reshape(
    [img_style_mask.shape[0], img_style_mask.shape[1], n_colors])
masks = masks.transpose([2, 0, 1])

f['style_mask'] = masks
f.close()

np.save(args.out_hdf5 + '_colors.npy', kmeans.cluster_centers_)
