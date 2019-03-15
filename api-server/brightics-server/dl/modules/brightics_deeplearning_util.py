import os
from datetime import datetime


def make_checkpoint_dir(checkpoint_dir, user, checkpoint):
    user_path = checkpoint_dir + user
    ckpt_path = user_path + '/' + checkpoint

    if not os.path.exists(user_path):
        os.mkdir(user_path)
    elif os.path.exists(ckpt_path):
        ckpt_path += '_'
        ckpt_path += '{:%Y%m%d_%H%M%S}'.format(datetime.now())

    os.mkdir(ckpt_path)
    return ckpt_path


def write_summary(filepath, summary=[]):
    layers_str = reformat_layers_log(summary[1:-5])
    summary = summary[-4:-1]

    with open(filepath, 'a') as file:
        file.write('\n'.join(summary + layers_str))
        file.write('\n')


def reformat_layers_log(layers=[]):
    def get_bucket_list(layer):
        layer_type_index = layer.find('Layer')
        output_shape_index = layer.find('Output')
        param_index = layer.find('Param')
        connected_index = layer.find('Connected')

        if connected_index != -1:
            index_list = [layer_type_index, output_shape_index, param_index, connected_index, len(layer)]
        else:
            index_list = [layer_type_index, output_shape_index, param_index, len(layer)]

        return list(zip(index_list[:-1], index_list[1:]))

    def is_layer_data(layer):
        return len(layer.replace('=', '').replace('_', '')) != 0

    bucket_list = get_bucket_list(layers[0])

    layers_str = ['|'.join([l[bucket[0]:bucket[1]].strip() for bucket in bucket_list]) for l in layers if is_layer_data(l)]

    return layers_str

