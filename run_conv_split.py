import heapq
import math


def de_inplace(net):
    inplace = False
    for layer in net.layer[1:-1]:
        if not inplace:
            if layer.bottom[0] == layer.top[0]:
                inplace = True
                lasttop = layer.top[0] + "_" + layer.type[:min(len(layer.type), 4)].lower()
                layer.top[0] = lasttop
        else:
            if layer.bottom[0] == layer.top[0]:
                layer.bottom[0] = lasttop
                layer.top[0] = lasttop + "_" + layer.type[:min(len(layer.type), 4)].lower()
                lasttop = layer.top[0]
            else:
                layer.bottom[0] = lasttop
                inplace = False


def remove_blobs(net):
    for layer in net.layer:
        del layer.blobs[:]


def split_convlayers(net):
    flag_layer = net.layer.add()
    flag_layer.name = "stop_flag"  # a helper layer to indicate the end of processing

    layers = net.layer
    while True:
        if layers[0].name == "stop_flag":
            del layers[0]  # delete the stop flag layer
            break

        elif layers[0].type == "Convolution" and layers[1].type == "BN":
            conv_layer = layers[0]
            bn_layer = layers[1]
            fmap_num = conv_layer.convolution_param.num_output

            print("delete conv layer: " + layers[0].name)
            del layers[0]  # del conv layer
            print("delete bn layer: " + layers[0].name)
            del layers[0]  # del BN layer

            for i in range(fmap_num):
                # add conv split layer
                conv_split_layer = layers.add()
                conv_split_layer.MergeFrom(conv_layer)
                conv_split_layer.name = "%s_split_%02d" % (conv_layer.name, i)
                del conv_split_layer.top[:]
                conv_split_layer.top.append("%s_split_%02d" % (conv_layer.name, i))
                conv_split_layer.convolution_param.num_output = 1
                for blob in conv_split_layer.blobs:
                    blob.shape.dim[0] = 1
                    fmap_size = int(len(blob.data) / fmap_num)
                    del blob.data[(i + 1) * fmap_size:]
                    del blob.data[:i * fmap_size]

                # add bn split layer
                bn_split_layer = layers.add()
                bn_split_layer.MergeFrom(bn_layer)
                bn_split_layer.name = "%s_split_%02d" % (bn_layer.name, i)
                del bn_split_layer.bottom[:]
                bn_split_layer.bottom.append("%s_split_%02d" % (conv_layer.name, i))
                del bn_split_layer.top[:]
                bn_split_layer.top.append("%s_split_%02d" % (bn_layer.name, i))
                for blob in bn_split_layer.blobs:
                    blob.shape.dim[1] = 1
                    fmap_size = int(len(blob.data) / fmap_num)
                    del blob.data[(i + 1) * fmap_size:]
                    del blob.data[:i * fmap_size]
            print("append conv split layers: %s_split_{00-%d}" % (conv_layer.name, fmap_num - 1))
            print("append bn split layers: %s_split_{00-%d}" % (bn_layer.name, fmap_num - 1))

            # add concat layer
            concat_layer = layers.add()
            concat_layer.name = "%s_split_concat" % bn_layer.name
            print("append concat layer: " + concat_layer.name)
            concat_layer.top.append(bn_layer.top[0])
            concat_layer.type = "Concat"
            for i in range(fmap_num):
                concat_layer.bottom.append("%s_split_%02d" % (bn_layer.name, i))
            concat_layer.concat_param.axis = 1

        else:
            print("move layer to the end: " + layers[0].name)
            new_layer = layers.add()
            new_layer.MergeFrom(layers[0])
            del layers[0]


def merge_conv_bn(net):
    conv_layers = []
    for i in range(len(net.layer) - 1):
        if net.layer[i].type == "Convolution" and net.layer[i + 1].type == "BN":
            conv_layers.append(i)

    for i in reversed(conv_layers):
        conv_layer = net.layer[i]
        bn_layer = net.layer[i + 1]
        fmap_num = conv_layer.convolution_param.num_output
        fmap_size = int(len(conv_layer.blobs[0].data) / fmap_num)

        for fi in range(fmap_num):
            gamma = bn_layer.blobs[0].data[fi]
            beta = bn_layer.blobs[1].data[fi]

            for di in range(fi * fmap_size, (fi + 1) * fmap_size):
                conv_layer.blobs[0].data[di] *= gamma

            conv_layer.blobs[1].data[fi] *= gamma
            conv_layer.blobs[1].data[fi] += beta

        conv_layer.top[0] = bn_layer.top[0]
        del net.layer[i + 1]  # del bn layer


def get_frac(data, ratio):
    if not 0 <= ratio <= 1:
        raise ValueError("Check failed: 0<=ratio<=1")

    heap_size = 1 if ratio == 1 else int(len(data) * (1 - ratio))

    heap = []
    for i in range(len(data)):
        d = abs(data[i])
        if i < heap_size:
            heapq.heappush(heap, d)
        elif heap[0] < d:
            heapq.heapreplace(heap, d)

    max_abs_value = heap[0]
    frac = -int(math.ceil(math.log(max_abs_value) / math.log(2)))
    return frac


def split_convlayers_by_conv_frac_w(net):
    # merge
    merge_conv_bn(net)

    flag_layer = net.layer.add()
    flag_layer.name = "stop_flag"  # a helper layer to indicate the end of processing

    layers = net.layer
    while True:
        if layers[0].name == "stop_flag":
            del layers[0]  # delete the stop flag layer
            break

        elif layers[0].type == "Convolution":
            conv_layer = layers[0]
            fmap_num = conv_layer.convolution_param.num_output

            del layers[0]  # del conv layer

            # get frac_w
            w_size = int(len(conv_layer.blobs[0].data) / fmap_num)
            ratio = 0.95
            frac_w = []
            for i in range(fmap_num):
                data = conv_layer.blobs[0].data[i * w_size:(i + 1) * w_size]
                frac_w.append(get_frac(data, ratio))

            frac_index_dict = {}
            for i, frac in enumerate(frac_w):
                if frac not in frac_index_dict:
                    frac_index_dict[frac] = [i]
                else:
                    frac_index_dict[frac].append(i)
            # print(frac_index_dict)

            print("%s fracw" % (conv_layer.name))

            for frac in sorted(frac_index_dict.keys()):
                indexes = frac_index_dict[frac]
                print("%s %s" % (frac, indexes))

                # add conv split layer
                conv_split_layer = layers.add()
                conv_split_layer.MergeFrom(conv_layer)
                conv_split_layer.name = "%s_fracw_%02d" % (conv_layer.name, frac)
                del conv_split_layer.top[:]
                conv_split_layer.top.append(conv_split_layer.name)
                conv_split_layer.convolution_param.num_output = len(indexes)
                for bi in range(len(conv_split_layer.blobs)):
                    blob = conv_split_layer.blobs[bi]
                    blob.shape.dim[0] = len(indexes)
                    del blob.data[:]
                    ori_blob = conv_layer.blobs[bi]
                    fmap_size = int(len(ori_blob.data) / fmap_num)
                    for i in indexes:
                        blob.data.extend(ori_blob.data[i * fmap_size:(i + 1) * fmap_size])

            # add concat layer
            concat_layer = layers.add()
            concat_layer.name = "%s_fracw_concat" % conv_layer.name
            concat_layer.type = "Concat"
            concat_layer.top.append(concat_layer.name)
            for frac in sorted(frac_index_dict.keys()):
                concat_layer.bottom.append("%s_fracw_%02d" % (conv_layer.name, frac))
            concat_layer.concat_param.axis = 1

            # add switch order layer
            switch_layer = layers.add()
            switch_layer.name = "%s_switch" % concat_layer.name
            switch_layer.type = "Convolution"
            switch_layer.bottom.append(concat_layer.name)
            switch_layer.top.append(conv_layer.top[0])
            switch_layer.convolution_param.num_output = fmap_num
            switch_layer.convolution_param.kernel_size.append(1)
            switch_layer.convolution_param.pad.append(0)

            blob0 = switch_layer.blobs.add()
            blob0.shape.dim.extend([fmap_num, fmap_num, 1, 1])
            blob0.data.extend([0] * fmap_num * fmap_num)
            blob1 = switch_layer.blobs.add()
            blob1.shape.dim.extend([fmap_num])
            blob1.data.extend([0] * fmap_num)

            param = switch_layer.param.add()
            param.lr_mult = 0

            indexes = []
            for frac in sorted(frac_index_dict.keys()):
                indexes += frac_index_dict[frac]

            for i, ind in enumerate(indexes):
                blob0.data[ind * fmap_num + i] = 1

            print(switch_layer.name)
            print(indexes)
            print("")

        else:
            new_layer = layers.add()
            new_layer.MergeFrom(layers[0])
            del layers[0]


if __name__ == "__main__":
    import pyrwproto
    input_caffemodel = "segnet_original.caffemodel"
    input_prototxt = "segnet_original.prototxt"
    output_caffemodel = "out.caffemodel"
    output_prototxt = "out.prototxt"

    net = pyrwproto.caffe_pb2.NetParameter()
    with open(input_caffemodel, "rb") as f:
        pyrwproto.ParseFromBinary(f.read(), net)

    de_inplace(net)

    split_convlayers_by_conv_frac_w(net)

    with open(output_caffemodel, "wb") as f:
        print("writing into %s ..." % output_caffemodel)
        f.write(pyrwproto.SerializeToBinary(net))

    remove_blobs(net)

    with open(output_prototxt, "w") as f:
        print("writing into %s ..." % output_prototxt)
        f.write(pyrwproto.SerializeToText(net))
