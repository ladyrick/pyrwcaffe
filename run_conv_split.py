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
                    fmap_size = len(blob.data) / fmap_num
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
                    fmap_size = len(blob.data) / fmap_num
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

    split_convlayers(net)

    with open(output_caffemodel, "wb") as f:
        f.write(pyrwproto.SerializeToBinary(net))

    remove_blobs(net)

    with open(output_prototxt, "w") as f:
        f.write(pyrwproto.SerializeToText(net))
