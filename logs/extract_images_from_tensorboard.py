# Example:
# python3 extract_images_from_tensorboard.py events.out.tfevents.1636059534.ws-06.46674.0 "Render, validation" images_extracted/

# Source: https://stackoverflow.com/questions/47232779/how-to-extract-and-save-images-from-tensorboard-event-summary

import os
import imageio
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def save_images_from_event(fn, tag, output_dir='./'):
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    imageio.imwrite(output_fn, im)
                    count += 1

if __name__ == '__main__':
    import sys
    event_file_path, tag_name, output_dir_path = sys.argv[1:]
    save_images_from_event(event_file_path, tag_name, output_dir_path)

