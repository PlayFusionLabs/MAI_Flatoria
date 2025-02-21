import tensorflow as tf
# TODO

IMAGE_SIZE = 64
IMAGE_CHANNELS = 3

def get_image_label_list(image_label_file):
    filenames = []
    labels = []
    for line in open(image_label_file, "r"):
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))

    return filenames, labels

def read_image_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    rgb_image = tf.image.decode_jpeg(file_contents, channels=IMAGE_CHANNELS,
        name="decode_jpeg")
    rgb_image = tf.image.resize_images(rgb_image, [IMAGE_SIZE, IMAGE_SIZE])
    return rgb_image, label