"""
Module containing processing code to load TCGA or CAMELYON datasets
from preextracted feature embeddings & preprocess data for ProtoMixer
"""

import h5py
import itertools


def load_features(data, labels, FEATURE_DIR, experiment="tcga"):
    features = []
    coords = []
    filenames = []
    if experiment == "tcga":
        tcga_files = os.listdir(FEATURE_DIR)
        data = [[x for x in tcga_files if file in x] for file in data]
        data = list(set(itertools.chain(*data)))
        tcga_labels = labels
        labels = list(range(len(data)))  # mock labels
        new_labels = []

    for file, label in zip(data[:], labels[:]):
        try:
            if experiment == "inhouse":
                f = h5py.File(f"{FEATURE_DIR}/{file}.h5", "r")
            elif experiment == "tcga":
                f = h5py.File(f"{FEATURE_DIR}/{file}", "r")
                handle = file.split(".")[0]
                new_labels.append(tcga_labels[handle])
            wsi_features = np.asarray(f["features"])
            wsi_coords = np.asarray(f["coords"])
            features.append(wsi_features)
            coords.append(wsi_coords)
            filenames.append(str(file))
            f.close()
        except:
            # can't load features so delete element in data and labels
            data.remove(file)
            labels.remove(label)
    if experiment == "tcga":
        labels = new_labels
    return features, labels, coords, filenames


def load_tcga_data(subtypes=SUBTYPES, path_to_slide_info=SLIDE_INFO_DIR):
    import pandas as pd

    data = []
    labels = {}
    label = 0

    f = pd.read_csv(path_to_slide_info)
    for subtype in subtypes:
        lines = f[f["project_id"] == "TCGA-" + subtype]
        data.extend(list(lines.PATIENT))
        for line in list(lines.FILENAME):
            labels[str(line)] = label
        label += 1

    return data, labels


# Load CAMELYON data - Special case that loads C16/C17 seperately and does
# not use load_features function


#
#
def load_camelyon_data(
    path_to_slide_info=SLIDE_INFO_DIR, path_to_features=FEATURE_DIR, mode="16"
):
    import pandas as pd

    path = path_to_slide_info + "CAMELYON" + mode + "/"
    path_to_features = path + path_to_features
    path_to_annotations = path + "annotations/"
    if mode == "16":
        annotations = sorted(os.listdir(path_to_annotations))
        annotations = [os.path.splitext(x)[0] for x in annotations]
    if mode == "17":
        df = pd.read_csv(path_to_annotations + "stages.csv")
        df["patient"] = df.patient.apply(lambda x: os.path.splitext(x)[0])

    features = []
    coords = []
    labels = []
    filenames = []

    for i in os.listdir(path_to_features):
        handle = os.path.splitext(i)[0]
        if mode == "16":
            if handle in annotations:
                labels.append(1)  # tumor positive label
            else:
                labels.append(0)  # negative label

        elif mode == "17":
            diagnosis = df.loc[df["patient"] == handle, ["stage"]].values[0][0]
            if diagnosis in ["micro", "macro"]:  # metastasis
                labels.append(1)
            elif diagnosis in ["negative", "itc"]:  # no metastasis
                labels.append(0)
        # try:
        f = h5py.File(f"{path_to_features}/{i}", "r")
        wsi_features = np.asarray(f["features"])
        wsi_coords = np.asarray(f["coords"])
        features.append(wsi_features)
        coords.append(wsi_coords)
        filenames.append(str(handle))
        f.close()
        # except:
        # pass

    return features, coords, labels, filenames


def preprocess_features(npdata, pca=-1):
    """Preprocess an array of features for kmeans clustering.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    assert npdata.dtype == np.float32

    if np.any(np.isnan(npdata)):
        raise Exception("nan occurs")

    if pca != -1:
        import faiss

        print("\nPCA from dim {} to dim {}".format(ndim, pca))
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)
    if np.any(np.isnan(npdata)):
        percent = np.isnan(npdata).sum().item() / float(np.size(npdata)) * 100
        if percent > 0.1:
            raise Exception(
                "More than 0.1% nan occurs after pca, percent: {}%".format(percent)
            )
        else:
            npdata[np.isnan(npdata)] = 0.0
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)

    npdata = npdata / (row_sums[:, np.newaxis] + 1e-10)

    return npdata


def cluster_dataset(features, k_clusters: int = 8, method="kmeans"):
    prototype_list = []
    cluster_labels_list = []
    if method == "kmeans":
        from sklearn.cluster import KMeans

        for wsi_features in features:
            kmeans = KMeans(n_clusters=k_clusters, random_state=1337, n_init=10)
            kmeans.fit_predict(wsi_features)
            prototype_list.append(kmeans.cluster_centers_)
            cluster_labels_list.append(kmeans.labels_)
        return np.stack(prototype_list), cluster_labels_list, kmeans

    elif method == "dbscan":
        from sklearn.cluster import DBSCAN

        for wsi_features in features:
            dbscan = DBSCAN(eps=2, min_samples=5)
            dbscan.fit_predict(wsi_features)
            prototype_list.append(
                np.array(
                    [
                        np.mean(wsi_features[dbscan.labels_ == i], axis=0)
                        for i in range(k_clusters)
                    ]
                )
            )
            cluster_labels_list.append(dbscan.labels_)

        return prototype_list, cluster_labels_list, dbscan

    elif method == "gmm":
        from sklearn.mixture import GaussianMixture

        for wsi_features in features:
            gmm = GaussianMixture(n_components=k_clusters)
            cluster_labels = gmm.fit_predict(wsi_features)
            prototype_list.append(
                np.array(
                    [
                        np.mean(wsi_features[cluster_labels == i], axis=0)
                        for i in range(k_clusters)
                    ]
                )
            )
            cluster_labels_list.append(cluster_labels)
        return prototype_list, cluster_labels_list, gmm
