from google.protobuf import text_format

try:
    import caffe_pb2
except:
    import os
    import sys
    sys.path.append("proto")
    os.system("protoc proto/caffe.proto --python_out=./")
    import caffe_pb2

__all__ = ["caffe_pb2",
           "ParseFromBinary",
           "ParseFromString",
           "SerializeToBinary",
           "SerializeToString"]


def ParseFromBinary(bytestr, message):
    message.ParseFromString(bytestr)


def ParseFromText(string, message):
    text_format.Merge(string, message)


def SerializeToBinary(message):
    return message.SerializeToString()


def SerializeToText(message):
    return text_format.MessageToString(message)
