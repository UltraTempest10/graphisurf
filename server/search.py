########################################################################################################################
# This function implements the image search/retrieval .
# inputs: Input location of uploaded image, extracted vectors
# 
########################################################################################################################
import tensorflow._api.v2.compat.v1 as tf
import numpy as np
import imageio.v2 as imageio

from scipy.spatial.distance import cosine
import pickle
import os
from tensorflow.python.platform import gfile

imsave = imageio.imsave
imread = imageio.imread

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

CLASS = {0: 'people', 1: 'animals', 2: 'plants', 3: 'objects', 4: 'scenes', 5: 'other'}


def get_top_k_similar(image_data, pred, pred_final, k):
    print("total data", len(pred))
    # print(image_data.shape)

    if not gfile.Exists('static/result'):
        os.mkdir('static/result')
    if not gfile.Exists('static/classified'):
        os.mkdir('static/classified')
    for i in CLASS.values():
        if not gfile.Exists('static/classified/' + i):
            os.mkdir('static/classified/' + i)

    tag = [np.loadtxt('database/tags/people.txt', dtype=int), np.loadtxt('database/tags/animals.txt', dtype=int),
           np.loadtxt('database/tags/plants.txt', dtype=int), np.loadtxt('database/tags/objects.txt', dtype=int),
           np.loadtxt('database/tags/scenes.txt', dtype=int), np.loadtxt('database/tags/other.txt', dtype=int)]

    # cosine calculates the cosine distance, not similarity, hence no need to reverse list
    distances = [cosine(image_data, pred_row) for ith_row, pred_row in enumerate(pred)]
    top_k_ind = np.argsort(distances)[:k]
    # filter out the ones that are not similar enough
    threshold = 0.32
    filtered_ind = [top_k_ind[i] for i in range(len(top_k_ind)) if distances[top_k_ind[i]] < threshold]

    # for i in filtered_ind:
    #     print(distances[i])

    print("recommended images ", len(filtered_ind))
    for i, neighbor in enumerate(filtered_ind):
        image = imread(pred_final[neighbor])
        name = pred_final[neighbor]
        tokens = name.split("\\")
        img_name = tokens[-1]

        # classify image
        img_no = img_name.split(".")[0][2:]
        for j in range(6):
            if int(img_no) in tag[j]:
                name = 'static/classified/' + CLASS[j] + '/' + img_name
                imsave(name, image)

        print(img_no)
        name = 'static/result/' + img_name
        imsave(name, image)


def create_inception_graph():
    """Creates a graph from saved GraphDef file and returns a Graph object.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
    with tf.Session() as sess:
        model_filename = os.path.join(
            'imagenet', 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def recommend(image_path, extracted_features):
    tf.reset_default_graph()

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    sess = tf.Session(config=config)
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())
    image_data = gfile.FastGFile(image_path, 'rb').read()
    features = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

    with open('neighbor_list_recom.pickle', 'rb') as f:
        neighbor_list = pickle.load(f)
    print("loaded images")
    get_top_k_similar(features, extracted_features, neighbor_list, k=12)
