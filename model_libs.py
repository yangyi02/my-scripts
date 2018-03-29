import os

from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def UnpackVariable(var, num):
    assert len > 0
    if type(var) is list and len(var) == num:
        return var
    else:
        ret = []
        if type(var) is list:
            assert len(var) == 1
            for i in xrange(0, num):
                ret.append(var[0])
        else:
            for i in xrange(0, num):
                ret.append(var)
        return ret

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
                kernel_size, pad, stride, use_scale=True, conv_prefix='', conv_postfix='',
                bn_prefix='', bn_postfix='_bn', scale_prefix='', scale_postfix='_scale',
                bias_prefix='', bias_postfix='_bias'):
    if use_bn:
        # parameters for convolution layer with batchnorm.
        kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1)],
            'weight_filler': dict(type='msra'),
            'bias_term': False,
        }
        # parameters for batchnorm layer.
        bn_kwargs = {
            'use_global_stats': False,
            'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        }
        # parameters for scale bias layer after batchnorm.
        if use_scale:
            sb_kwargs = {
                'bias_term': True,
                'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                'filler': dict(type='constant', value=1.0),
                'bias_filler': dict(type='constant', value=0.0),
            }
        else:
            bias_kwargs = {
                'param': [dict(lr_mult=1, decay_mult=0)],
                'filler': dict(type='constant', value=0.0),
            }
    else:
        kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='msra'),
            'bias_filler': dict(type='constant', value=0)
        }

    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
    [pad_h, pad_w] = UnpackVariable(pad, 2)
    [stride_h, stride_w] = UnpackVariable(stride, 2)
    if kernel_h == kernel_w:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                                       kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
    else:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                                       kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                                       stride_h=stride_h, stride_w=stride_w, **kwargs)
    if use_bn:
        bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
        net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
        if use_scale:
            sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
            net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
        else:
            bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
            net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
    if use_relu:
        relu_name = '{}_relu'.format(conv_name)
        net[relu_name] = L.ReLU(net[conv_name], in_place=True)


def ConvBNLayer_mult10(net, from_layer, out_layer, use_bn, use_relu, num_output,
                kernel_size, pad, stride, use_scale=True, conv_prefix='', conv_postfix='',
                bn_prefix='', bn_postfix='_bn', scale_prefix='', scale_postfix='_scale',
                bias_prefix='', bias_postfix='_bias'):
    if use_bn:
        # parameters for convolution layer with batchnorm.
        kwargs = {
            'param': [dict(lr_mult=10, decay_mult=1)],
            'weight_filler': dict(type='msra'),
            'bias_term': False,
        }
        # parameters for batchnorm layer.
        bn_kwargs = {
            'use_global_stats': False,
            'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        }
        # parameters for scale bias layer after batchnorm.
        if use_scale:
            sb_kwargs = {
                'bias_term': True,
                'param': [dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
                'filler': dict(type='constant', value=1.0),
                'bias_filler': dict(type='constant', value=0.0),
            }
        else:
            bias_kwargs = {
                'param': [dict(lr_mult=10, decay_mult=0)],
                'filler': dict(type='constant', value=0.0),
            }
    else:
        kwargs = {
            'param': [dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
            'weight_filler': dict(type='msra'),
            'bias_filler': dict(type='constant', value=0)
        }

    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
    [pad_h, pad_w] = UnpackVariable(pad, 2)
    [stride_h, stride_w] = UnpackVariable(stride, 2)
    if kernel_h == kernel_w:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                                       kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
    else:
        net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
                                       kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                                       stride_h=stride_h, stride_w=stride_w, **kwargs)
    if use_bn:
        bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
        net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
        if use_scale:
            sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
            net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
        else:
            bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
            net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
    if use_relu:
        relu_name = '{}_relu'.format(conv_name)
        net[relu_name] = L.ReLU(net[conv_name], in_place=True)


def ConvBNLayerOfIdenMap(net, from_layer, out_layer, use_bn, use_relu, num_output,
                kernel_size, pad, stride, use_scale=True, conv_prefix='', conv_postfix='',
                bn_prefix='', bn_postfix='_bn', scale_prefix='', scale_postfix='_scale',
                bias_prefix='', bias_postfix='_bias'):
    if use_bn:
        # parameters for convolution layer with batchnorm.
        kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1)],
            'weight_filler': dict(type='msra'),
            'bias_term': False,
        }
        # parameters for batchnorm layer.
        bn_kwargs = {
            'use_global_stats': False,
            'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],
        }
        # parameters for scale bias layer after batchnorm.
        if use_scale:
            sb_kwargs = {
                'bias_term': True,
                'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                'filler': dict(type='constant', value=1.0),
                'bias_filler': dict(type='constant', value=0.0),
            }
        else:
            bias_kwargs = {
                'param': [dict(lr_mult=1, decay_mult=0)],
                'filler': dict(type='constant', value=0.0),
            }
    else:
        kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='msra'),
            'bias_filler': dict(type='constant', value=0)
        }
    conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
    if use_bn:
        bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
        net[bn_name] = L.BatchNorm(net[from_layer], in_place=False, **bn_kwargs)
        if use_scale:
            sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
            net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
        else:
            bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
            net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
    if use_relu:
        relu_name = '{}_relu'.format(conv_name)
        net[relu_name] = L.ReLU(net[sb_name], in_place=True)
    else:
        relu_name = from_layer

    [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
    [pad_h, pad_w] = UnpackVariable(pad, 2)
    [stride_h, stride_w] = UnpackVariable(stride, 2)
    if kernel_h == kernel_w:
        net[conv_name] = L.Convolution(net[relu_name], num_output=num_output,
                                       kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
    else:
        net[conv_name] = L.Convolution(net[relu_name], num_output=num_output,
                                       kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
                                       stride_h=stride_h, stride_w=stride_w, **kwargs)


def ResBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1):
    # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

    conv_prefix = 'res{}_'.format(block_name)
    conv_postfix = ''
    bn_prefix = 'bn{}_'.format(block_name)
    bn_postfix = ''
    scale_prefix = 'scale{}_'.format(block_name)
    scale_postfix = ''
    use_scale = True

    if use_branch1:
        branch_name = 'branch1'
        ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
                    num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix)
        branch1 = '{}{}'.format(conv_prefix, branch_name)
    else:
        branch1 = from_layer

    branch_name = 'branch2a'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
                num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2b'
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=True,
                num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2c'
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
                num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch2 = '{}{}'.format(conv_prefix, branch_name)

    res_name = 'res{}'.format(block_name)
    net[res_name] = L.Eltwise(net[branch1], net[branch2])
    relu_name = '{}_relu'.format(res_name)
    net[relu_name] = L.ReLU(net[res_name], in_place=True)

def ResBody_mult10(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1):
    # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

    conv_prefix = 'res{}_'.format(block_name)
    conv_postfix = ''
    bn_prefix = 'bn{}_'.format(block_name)
    bn_postfix = ''
    scale_prefix = 'scale{}_'.format(block_name)
    scale_postfix = ''
    use_scale = True

    if use_branch1:
        branch_name = 'branch1'
        ConvBNLayer_mult10(net, from_layer, branch_name, use_bn=True, use_relu=False,
                    num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix)
        branch1 = '{}{}'.format(conv_prefix, branch_name)
    else:
        branch1 = from_layer

    branch_name = 'branch2a'
    ConvBNLayer_mult10(net, from_layer, branch_name, use_bn=True, use_relu=True,
                num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2b'
    ConvBNLayer_mult10(net, out_name, branch_name, use_bn=True, use_relu=True,
                num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2c'
    ConvBNLayer_mult10(net, out_name, branch_name, use_bn=True, use_relu=False,
                num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch2 = '{}{}'.format(conv_prefix, branch_name)

    res_name = 'res{}'.format(block_name)
    net[res_name] = L.Eltwise(net[branch1], net[branch2])
    relu_name = '{}_relu'.format(res_name)
    net[relu_name] = L.ReLU(net[res_name], in_place=True)

def ResBodyCifar(net, from_layer, block_name, out2a, out2b, stride, use_branch1):
    # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

    conv_prefix = 'res{}_'.format(block_name)
    conv_postfix = ''
    bn_prefix = 'bn{}_'.format(block_name)
    bn_postfix = ''
    scale_prefix = 'scale{}_'.format(block_name)
    scale_postfix = ''
    use_scale = True

    if use_branch1:
        branch_name = 'branch1'
        ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
                    num_output=out2b, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix)
        branch1 = '{}{}'.format(conv_prefix, branch_name)
    else:
        branch1 = from_layer

    branch_name = 'branch2a'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
                num_output=out2a, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2b'
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
                num_output=out2b, kernel_size=3, pad=1, stride=stride, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    # branch_name = 'branch2c'
    # ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
    #             num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
    #             conv_prefix=conv_prefix, conv_postfix=conv_postfix,
    #             bn_prefix=bn_prefix, bn_postfix=bn_postfix,
    #             scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    # branch2 = '{}{}'.format(conv_prefix, branch_name)

    res_name = 'res{}'.format(block_name)
    net[res_name] = L.Eltwise(net[branch1], net[out_name])
    relu_name = '{}_relu'.format(res_name)
    net[relu_name] = L.ReLU(net[res_name], in_place=True)


def ResBodyWithPrefix(net, from_layer, block_name, out2a, out2b, stride, use_branch1, hg_prefix):
    # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

    conv_prefix = '{}_res{}_'.format(hg_prefix, block_name)
    conv_postfix = ''
    bn_prefix = '{}_bn{}_'.format(hg_prefix, block_name)
    bn_postfix = ''
    scale_prefix = '{}_scale{}_'.format(hg_prefix, block_name)
    scale_postfix = ''
    use_scale = True

    if use_branch1:
        branch_name = 'branch1'
        ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
                    num_output=out2b, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix)
        branch1 = '{}{}'.format(conv_prefix, branch_name)
    else:
        branch1 = from_layer

    branch_name = 'branch2a'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
                num_output=out2a, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2b'
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
                num_output=out2b, kernel_size=3, pad=1, stride=stride, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    # branch_name = 'branch2c'
    # ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
    #             num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
    #             conv_prefix=conv_prefix, conv_postfix=conv_postfix,
    #             bn_prefix=bn_prefix, bn_postfix=bn_postfix,
    #             scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    # branch2 = '{}{}'.format(conv_prefix, branch_name)
    branch2 = out_name
    res_name = '{}_res{}'.format(hg_prefix, block_name)
    net[res_name] = L.Eltwise(net[branch1], net[branch2])
    relu_name = '{}_relu'.format(res_name)
    net[relu_name] = L.ReLU(net[res_name], in_place=True)


def IdenMapBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1):
    # IdenMapBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

    conv_prefix = 'res{}_'.format(block_name)
    conv_postfix = ''
    bn_prefix = 'bn{}_'.format(block_name)
    bn_postfix = ''
    scale_prefix = 'scale{}_'.format(block_name)
    scale_postfix = ''
    use_scale = True

    if use_branch1:
        branch_name = 'branch1'
        ConvBNLayerOfIdenMap(net, from_layer, branch_name, use_bn=False, use_relu=False,
                    num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix)
        branch1 = '{}{}'.format(conv_prefix, branch_name)
    else:
        branch1 = from_layer

    branch_name = 'branch2a'
    ConvBNLayerOfIdenMap(net, from_layer, branch_name, use_bn=True, use_relu=True,
                num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2b'
    ConvBNLayerOfIdenMap(net, out_name, branch_name, use_bn=True, use_relu=True,
                num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2c'
    ConvBNLayerOfIdenMap(net, out_name, branch_name, use_bn=True, use_relu=True,
                num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch2 = '{}{}'.format(conv_prefix, branch_name)

    res_name = 'res{}'.format(block_name)
    net[res_name] = L.Eltwise(net[branch1], net[branch2])

def IdenMapBody2(net, from_layer, block_name, out2a, out2b, stride, use_branch1):
    # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

    conv_prefix = 'res{}_'.format(block_name)
    conv_postfix = ''
    bn_prefix = 'bn{}_'.format(block_name)
    bn_postfix = ''
    scale_prefix = 'scale{}_'.format(block_name)
    scale_postfix = ''
    use_scale = True

    if use_branch1:
        branch_name = 'branch1'
        ConvBNLayerOfIdenMap(net, from_layer, branch_name, use_bn=False, use_relu=False,
                    num_output=out2b, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix)
        branch1 = '{}{}'.format(conv_prefix, branch_name)
    else:
        branch1 = from_layer

    branch_name = 'branch2a'
    ConvBNLayerOfIdenMap(net, from_layer, branch_name, use_bn=True, use_relu=True,
                num_output=out2a, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2b'
    ConvBNLayerOfIdenMap(net, out_name, branch_name, use_bn=True, use_relu=True,
                num_output=out2b, kernel_size=3, pad=1, stride=stride, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    # branch_name = 'branch2c'
    # ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
    #             num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
    #             conv_prefix=conv_prefix, conv_postfix=conv_postfix,
    #             bn_prefix=bn_prefix, bn_postfix=bn_postfix,
    #             scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    # branch2 = '{}{}'.format(conv_prefix, branch_name)

    res_name = 'res{}'.format(block_name)
    net[res_name] = L.Eltwise(net[branch1], net[out_name])
#    relu_name = '{}_relu'.format(res_name)
#    net[relu_name] = L.ReLU(net[res_name], in_place=True)

def UpsampleDeepLab(net, from_layer, n):
    ##### RessBody before upsampling 1 #####
    ResBody_mult10(net, from_layer, '_up1', out2a=2048, out2b=512, out2c=512, stride=1, use_branch1=True)
    
    ##### upsampling x2 #####
    from_layer = 'res_up1'
    interp_name = 'up1'
    net[interp_name] = L.Interp(net[from_layer], interp_param=dict(zoom_factor=2, pad_beg=0, pad_end=0))

    ##### RessBody of new hourglass #####
    ResBody_mult10(net, 'up1', '6a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True)
    from_layer = 'res6a'
    for i in xrange(1, n):
        block_name = '6b{}'.format(i)
        ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
        from_layer = 'res{}'.format(block_name)

    ##### RessBody before upsampling 2#####
    ResBody_mult10(net, from_layer, '_up2', out2a=512, out2b=128, out2c=128, stride=1, use_branch1=True)
    
    ##### upsampling x2 #####
    from_layer = 'res_up2'
    interp_name = 'up2'
    net[interp_name] = L.Interp(net[from_layer], interp_param=dict(zoom_factor=2, pad_beg=0, pad_end=0))    

    return net

def IdenMapOutHourglass(net, from_layer, block_name, numIn, numOut, stride):

    conv_prefix = 'hg_res{}_'.format(block_name)
    conv_postfix = ''
    bn_prefix = 'hg_bn{}_'.format(block_name)
    bn_postfix = ''
    scale_prefix = 'hg_scale{}_'.format(block_name)
    scale_postfix = ''
    use_scale = True

    out2a = numOut / 2
    out2b = numOut / 2
    out2c = numOut

    if numIn != numOut:
        branch_name = 'branch1'
        ConvBNLayerOfIdenMap(net, from_layer, branch_name, use_bn=False, use_relu=False,
                    num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix)
        branch1 = '{}{}'.format(conv_prefix, branch_name)
    else:
        branch1 = from_layer

    branch_name = 'branch2a'
    ConvBNLayerOfIdenMap(net, from_layer, branch_name, use_bn=True, use_relu=True,
                num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2b'
    ConvBNLayerOfIdenMap(net, out_name, branch_name, use_bn=True, use_relu=True,
                num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2c'
    ConvBNLayerOfIdenMap(net, out_name, branch_name, use_bn=True, use_relu=True,
                num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch2 = '{}{}'.format(conv_prefix, branch_name)

    res_name = 'hg_res{}'.format(block_name)
    net[res_name] = L.Eltwise(net[branch1], net[branch2])
    # relu_name = '{}_relu'.format(res_name)
    # net[relu_name] = L.ReLU(net[res_name], in_place=True)


def IdenMapInHourglass(net, from_layer, block_name, numIn, numOut, stride, hg_prefix, up_low_prefix, size_prefix):
    # ResBodyInHourglass(net, 'pool1', '2a', 64, 64, 256, 1, True)

    conv_prefix = '{}_{}_{}_res{}_'.format(hg_prefix, size_prefix, up_low_prefix, block_name)
    conv_postfix = ''
    bn_prefix = '{}_{}_{}_bn{}_'.format(hg_prefix, size_prefix, up_low_prefix, block_name)
    bn_postfix = ''
    scale_prefix = '{}_{}_{}_scale{}_'.format(hg_prefix, size_prefix, up_low_prefix, block_name)
    scale_postfix = ''
    use_scale = True

    out2a = numOut / 2
    out2b = numOut / 2
    out2c = numOut

    if numIn != numOut:
        branch_name = 'branch1'
        ConvBNLayerOfIdenMap(net, from_layer, branch_name, use_bn=False, use_relu=False,
                    num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix)
        branch1 = '{}{}'.format(conv_prefix, branch_name)
    else:
        branch1 = from_layer

    branch_name = 'branch2a'
    ConvBNLayerOfIdenMap(net, from_layer, branch_name, use_bn=True, use_relu=True,
                num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2b'
    ConvBNLayerOfIdenMap(net, out_name, branch_name, use_bn=True, use_relu=True,
                num_output=out2b, kernel_size=3, pad=1, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2c'
    ConvBNLayerOfIdenMap(net, out_name, branch_name, use_bn=True, use_relu=True,
                num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch2 = '{}{}'.format(conv_prefix, branch_name)

    res_name = '{}_{}_{}_res{}'.format(hg_prefix, size_prefix, up_low_prefix, block_name)
    net[res_name] = L.Eltwise(net[branch1], net[branch2])
    # relu_name = '{}_relu'.format(res_name)
    # net[relu_name] = L.ReLU(net[res_name], in_place=True)

def Hourglass(net, from_layer, numIn, numOut, stride, hg_prefix, stage):
	######### imagenet: 256 cifar10: 32
    size = 'size{}'.format(stage)
    from_layer_in = from_layer
    #####Upper branch#####
    up_low = 'up'
    IdenMapInHourglass(net, from_layer, '1', numIn, 256, stride, hg_prefix, up_low, size)
    from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '1')
    print from_layer
    IdenMapInHourglass(net, from_layer, '2', 256, 256, stride, hg_prefix, up_low, size)
    from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '2')
    print from_layer
    IdenMapInHourglass(net, from_layer, '3', 256, numOut, stride, hg_prefix, up_low, size)
    up_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '3')
    print up_layer

    ######Lower branch#####
    pool_name = '{}_{}_pool'.format(hg_prefix, size)
    net[pool_name] = L.Pooling(net[from_layer_in], pool=P.Pooling.MAX, kernel_size=2, stride=2)
    up_low = 'low'
    IdenMapInHourglass(net, pool_name, '1', numIn, 256, stride, hg_prefix, up_low, size)
    from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '1')
    print from_layer
    IdenMapInHourglass(net, from_layer, '2', 256, 256, stride, hg_prefix, up_low, size)
    from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '2')
    print from_layer
    IdenMapInHourglass(net, from_layer, '3', 256, 256, stride, hg_prefix, up_low, size)
    from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '3')
    print from_layer

    if stage > 1:
        Hourglass(net, from_layer, 256, numOut, stride, hg_prefix, stage - 1)
        temp_size = 'size{}'.format(stage - 1)
        from_layer = '{}_{}_{}_add'.format(hg_prefix, temp_size, up_low)
        IdenMapInHourglass(net, from_layer, '4', numOut, numOut, stride, hg_prefix, up_low, size)
        from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '4')  
        print from_layer
    else:
        IdenMapInHourglass(net, from_layer, '4', 256, numOut, stride, hg_prefix, up_low, size)
        from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '4')    
        print from_layer        
        IdenMapInHourglass(net, from_layer, '5', numOut, numOut, stride, hg_prefix, up_low, size)
        from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '5')
        print from_layer 

    interp_name = '{}_{}_{}_interp'.format(hg_prefix, size, up_low)
    print interp_name
    net[interp_name] = L.Interp(net[from_layer], interp_param=dict(zoom_factor=2, pad_beg=0, pad_end=0))
    # deconv_name = '{}_{}_{}_deconv'.format(hg_prefix, size, up_low)
    # print deconv_name
    # net[deconv_name] = L.Deconvolution(net[from_layer], convolution_param=dict(num_output=numOut, group=numOut, kernel_size=4, stride=2, pad=1, bias_term=False, weight_filler=dict(type='bilinear')), param=[dict(lr_mult=0, decay_mult=0)])
    add_name = '{}_{}_{}_add'.format(hg_prefix, size, up_low)
    print add_name
    net[add_name] = L.Eltwise(net[interp_name], net[up_layer])

    return net

def Hourglass2(net, from_layer, numIn, numOut, stride, hg_prefix, stage):
	######### imagenet:256 cifar10:32
    size = 'size{}'.format(stage)
    from_layer_in = from_layer
    #####Upper branch#####
    up_low = 'up'
    IdenMapInHourglass(net, from_layer, '1', numIn, numOut, stride, hg_prefix, up_low, size)
    up_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '1')
    print up_layer
    # IdenMapInHourglass(net, from_layer, '2', 256, numOut, stride, hg_prefix, up_low, size)
    # up_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '2')
    # print up_layer

    ######Lower branch#####
    pool_name = '{}_{}_pool'.format(hg_prefix, size)
    net[pool_name] = L.Pooling(net[from_layer_in], pool=P.Pooling.MAX, kernel_size=2, stride=2)
    up_low = 'low'
    IdenMapInHourglass(net, pool_name, '1', numIn, 256, stride, hg_prefix, up_low, size)
    from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '1')
    print from_layer
    # IdenMapInHourglass(net, from_layer, '2', 256, 256, stride, hg_prefix, up_low, size)
    # from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '2')
    # print from_layer

    if stage > 1:
        Hourglass2(net, from_layer, 256, numOut, stride, hg_prefix, stage - 1)
        temp_size = 'size{}'.format(stage - 1)
        from_layer = '{}_{}_{}_add'.format(hg_prefix, temp_size, up_low)
        IdenMapInHourglass(net, from_layer, '3', numOut, numOut, stride, hg_prefix, up_low, size)
        from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '3')  
        print from_layer
    else:
        IdenMapInHourglass(net, from_layer, '3', 256, numOut, stride, hg_prefix, up_low, size)
        from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '3')    
        print from_layer        
        # IdenMapInHourglass(net, from_layer, '4', numOut, numOut, stride, hg_prefix, up_low, size)
        # from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '4')
        # print from_layer 

    interp_name = '{}_{}_{}_interp'.format(hg_prefix, size, up_low)
    print interp_name
    net[interp_name] = L.Interp(net[from_layer], interp_param=dict(zoom_factor=2, pad_beg=0, pad_end=0))
    # deconv_name = '{}_{}_{}_deconv'.format(hg_prefix, size, up_low)
    # print deconv_name
    # net[deconv_name] = L.Deconvolution(net[from_layer], convolution_param=dict(num_output=numOut, group=numOut, kernel_size=4, stride=2, pad=1, bias_term=False, weight_filler=dict(type='bilinear')), param=[dict(lr_mult=0, decay_mult=0)])
    add_name = '{}_{}_{}_add'.format(hg_prefix, size, up_low)
    print add_name
    net[add_name] = L.Eltwise(net[interp_name], net[up_layer])

    return net

def HalfHourglass(net, from_layer, numIn, numOut, stride, hg_prefix, stage):
    ######Lower branch##### imagenet:256 cifar10:32
    up_low = 'low'
    for i in xrange(stage, 0, -1):
        size = 'size{}'.format(i)
        pool_name = '{}_{}_pool'.format(hg_prefix, size)
        net[pool_name] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2)
        if i == stage:
            numIn = 384
        else:
            numIn = 256
        IdenMapInHourglass(net, pool_name, '1', numIn, 256, stride, hg_prefix, up_low, size)
        from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '1')
        print from_layer
        IdenMapInHourglass(net, from_layer, '2', 256, 256, stride, hg_prefix, up_low, size)
        from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '2')
        print from_layer
        IdenMapInHourglass(net, from_layer, '3', 256, 256, stride, hg_prefix, up_low, size)
        from_layer = '{}_{}_{}_res{}'.format(hg_prefix, size, up_low, '3')
        print from_layer

    return net


def Hourglass_1_5(net, from_layer_data, from_layer_label, stride, stage):
    ConvBNLayer(net, from_layer_data, 'conv1', use_bn=True, use_relu=True, num_output=64, kernel_size=7, pad=3, stride=2)
    IdenMapOutHourglass(net, 'conv1', '1', 64, 128, stride) 
    net['pool1'] = L.Pooling(net['res1'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  
    IdenMapOutHourglass(net, 'pool1', '2', 128, 128, stride) 
    IdenMapOutHourglass(net, 'res2', '3', 128, 128, stride) 
    IdenMapOutHourglass(net, 'res3', '4', 128, 256, stride)

    ######1 Hourglass network#####
    Hourglass(net, 'res4', 256, 512, stride, 'hg1', stage)

    ######linear layer######
    ConvBNLayer(net, 'hg1_size4_low_add', 'hg1_linear1', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg1_linear1', 'hg1_linear2', use_bn=True, use_relu=True, num_output=256, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg1_linear2', 'hg1_proj1', use_bn=False, use_relu=False, num_output=256, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg1_proj1', 'hg1_proj2', use_bn=False, use_relu=False, num_output=384, kernel_size=1, pad=0, stride=1)

    ######loss1#####
    ConvBNLayer(net, 'hg1_size1_low_res3', 'loss1/linear1', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'loss1/linear1', 'loss1/linear2', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)
    
    net['loss1/dropout'] = L.Dropout(net['loss1/linear2'], dropout_ratio=0.5, in_place=True)
    net['loss1/classifier'] = L.InnerProduct(net['loss1/dropout'], num_output=200, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss1/loss'] = L.SoftmaxWithLoss(net['loss1/classifier'], net[from_layer_label])
    net['loss1/top-1'] = L.Accuracy(net['loss1/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net['loss1/top-5'] = L.Accuracy(net['loss1/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')), top_k=5)

    #####Concat and Eltwise layer#####
    net['concat'] = L.Concat(net['pool1'], net['hg1_linear2'])
    ConvBNLayer(net, 'concat', 'concat_proj', use_bn=False, use_relu=False, num_output=384, kernel_size=1, pad=0, stride=1)
    net['concat_add'] = L.Eltwise(net['hg1_proj2'], net['concat_proj'])

    #####1.5 Hourglass#####
    HalfHourglass(net, 'concat_add', 384, 512, stride, 'hg2', stage)

    ######linear layer######
    ConvBNLayer(net, 'hg2_size1_low_res3', 'hg2_linear1', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg2_linear1', 'hg2_linear2', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)

    ######loss2#####
    net['loss2/dropout'] = L.Dropout(net['hg2_linear2'], dropout_ratio=0.5, in_place=True)
    net['loss2/classifier'] = L.InnerProduct(net['loss2/dropout'], num_output=200, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss2/loss'] = L.SoftmaxWithLoss(net['loss2/classifier'], net[from_layer_label])
    net['loss2/top-1'] = L.Accuracy(net['loss2/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net['loss2/top-5'] = L.Accuracy(net['loss2/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')), top_k=5)

    return net

def add_hg_deeplab(net, from_layer, stride, stage):
    # upsample feature map
    # ConvBNLayer(net, from_layer, 'hg_conv1', use_bn=True, use_relu=True, num_output=64, kernel_size=7, pad=3, stride=2)
    # IdenMapOutHourglass(net, 'hg_conv1', '1', 64, 128, stride) 
    # net['hg_pool1'] = L.Pooling(net['hg_res1'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  
    # IdenMapOutHourglass(net, 'hg_pool1', '2', 64, 128, stride)  
    # IdenMapOutHourglass(net, 'hg_res2', '3', 128, 128, stride) 
    # IdenMapOutHourglass(net, 'hg_res3', '4', 128, 256, stride)
    # downsample data
    ConvBNLayer(net, from_layer, 'hg_conv1', use_bn=True, use_relu=True, num_output=64, kernel_size=7, pad=3, stride=1)
    IdenMapOutHourglass(net, 'hg_conv1', '1', 64, 128, stride)  
    IdenMapOutHourglass(net, 'hg_res1', '2', 128, 128, stride) 
    IdenMapOutHourglass(net, 'hg_res2', '3', 128, 256, stride)

    ######1 Hourglass network#####
    Hourglass(net, 'hg_res3', 256, 512, stride, 'hg', stage)

    ConvBNLayer(net, 'hg_size2_low_add', 'hg_linear1', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg_linear1', 'hg_linear2', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)

    return net

def Hourglass_05(net, from_layer_data, from_layer_label, stride, stage):
    ConvBNLayer(net, from_layer_data, 'conv1', use_bn=True, use_relu=True, num_output=64, kernel_size=7, pad=3, stride=2)
    IdenMapOutHourglass(net, 'conv1', '1', 64, 128, stride) 
    net['pool1'] = L.Pooling(net['res1'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  

    #####1.5 Hourglass#####
    HalfHourglass(net, 'pool1', 384, 512, stride, 'hg2', stage)

    ######linear layer######
    ConvBNLayer(net, 'hg2_size1_low_res3', 'hg2_linear1', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg2_linear1', 'hg2_linear2', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)

    ######loss2#####
    net['loss/dropout'] = L.Dropout(net['hg2_linear2'], dropout_ratio=0.5, in_place=True)
    net['loss/classifier'] = L.InnerProduct(net['loss/dropout'], num_output=200, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss/loss'] = L.SoftmaxWithLoss(net['loss/classifier'], net[from_layer_label])
    net['loss/top-1'] = L.Accuracy(net['loss/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net['loss/top-5'] = L.Accuracy(net['loss/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')), top_k=5)


    return net

def Hourglass_1_5OfCifar10(net, from_layer_data, from_layer_label, stride, stage):
    ConvBNLayer(net, from_layer_data, 'conv1', use_bn=True, use_relu=True, num_output=32, kernel_size=3, pad=1, stride=1)

    ######1 Hourglass network#####
    Hourglass(net, 'conv1', 32, 64, stride, 'hg1', stage)

    ######linear layer######
    ConvBNLayer(net, 'hg1_size4_low_add', 'hg1_linear1', use_bn=True, use_relu=True, num_output=64, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg1_linear1', 'hg1_linear2', use_bn=True, use_relu=True, num_output=32, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg1_linear2', 'hg1_proj1', use_bn=False, use_relu=False, num_output=64, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg1_proj1', 'hg1_proj2', use_bn=False, use_relu=False, num_output=96, kernel_size=1, pad=0, stride=1)

    ######loss1#####
    ConvBNLayer(net, 'hg1_size1_low_res3', 'loss1/linear1', use_bn=True, use_relu=True, num_output=64, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'loss1/linear1', 'loss1/linear2', use_bn=True, use_relu=True, num_output=64, kernel_size=1, pad=0, stride=1)
    
    net['loss1/dropout'] = L.Dropout(net['loss1/linear2'], dropout_ratio=0.5, in_place=True)
    net['loss1/classifier'] = L.InnerProduct(net['loss1/dropout'], num_output=10, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss1/loss'] = L.SoftmaxWithLoss(net['loss1/classifier'], net[from_layer_label])
    net['loss1/top-1'] = L.Accuracy(net['loss1/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net['loss1/top-5'] = L.Accuracy(net['loss1/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')), top_k=5)
    net['loss1/loss'] = L.SoftmaxWithLoss(net['loss1/classifier'], net[from_layer_label])

    #####Concat and Eltwise layer#####
    net['concat'] = L.Concat(net['conv1'], net['hg1_linear2'])
    ConvBNLayer(net, 'concat', 'concat_proj', use_bn=False, use_relu=False, num_output=96, kernel_size=1, pad=0, stride=1)
    net['concat_add'] = L.Eltwise(net['hg1_proj2'], net['concat_proj'])

    #####1.5 Hourglass#####
    HalfHourglass(net, 'concat_add', 96, 128, stride, 'hg2', stage)

    ######linear layer######
    ConvBNLayer(net, 'hg2_size1_low_res3', 'hg2_linear1', use_bn=True, use_relu=True, num_output=64, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg2_linear1', 'hg2_linear2', use_bn=True, use_relu=True, num_output=64, kernel_size=1, pad=0, stride=1)

    ######loss2#####
    net['loss2/dropout'] = L.Dropout(net['hg2_linear2'], dropout_ratio=0.5, in_place=True)
    net['loss2/classifier'] = L.InnerProduct(net['loss2/dropout'], num_output=10, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss2/loss'] = L.SoftmaxWithLoss(net['loss2/classifier'], net[from_layer_label])
    net['loss2/top-1'] = L.Accuracy(net['loss2/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net['loss2/top-5'] = L.Accuracy(net['loss2/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')), top_k=5)


    return net

def StackHourglass(net, from_layer_data, from_layer_label, stride, stage):
    ConvBNLayer(net, from_layer_data, 'conv1', use_bn=True, use_relu=True, num_output=64, kernel_size=7, pad=3, stride=2)
    IdenMapOutHourglass(net, 'conv1', '1', 64, 128, stride) 
    net['pool1'] = L.Pooling(net['res1'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  
    IdenMapOutHourglass(net, 'pool1', '2', 128, 128, stride) 
    IdenMapOutHourglass(net, 'res2', '3', 128, 128, stride) 
    IdenMapOutHourglass(net, 'res3', '4', 128, 256, stride)

    ######First Hourglass network#####
    Hourglass(net, 'res4', 256, 512, stride, 'hg1', stage)

    ######linear layer######
    ConvBNLayer(net, 'hg1_size4_low_add', 'hg1_linear1', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg1_linear1', 'hg1_linear2', use_bn=True, use_relu=True, num_output=256, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg1_linear2', 'hg1_proj1', use_bn=False, use_relu=False, num_output=256, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg1_proj1', 'hg1_proj2', use_bn=False, use_relu=False, num_output=384, kernel_size=1, pad=0, stride=1)

    ######loss1#####    
    net['loss1/loss'] = L.SoftmaxWithLoss(net['hg1_proj1'], net[from_layer_label])
    net['loss1/top-1'] = L.Accuracy(net['hg1_proj1'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    #####Concat and Eltwise layer#####
    net['concat'] = L.Concat(net['pool1'], net['hg1_linear2'])
    ConvBNLayer(net, 'concat', 'concat_proj', use_bn=False, use_relu=False, num_output=384, kernel_size=1, pad=0, stride=1)
    net['concat_add'] = L.Eltwise(net['hg1_proj2'], net['concat_proj'])

    #####1.5 Hourglass#####
    Hourglass(net, 'concat_add', 384, 512, stride, 'hg2', stage)

    ######linear layer######
    ConvBNLayer(net, 'hg1_size4_low_add', 'hg2_linear1', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg2_linear1', 'hg2_linear2', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)

    ######loss2#####
    net['loss2/loss'] = L.SoftmaxWithLoss(net[from_layer_label], net['hg2_linear2'])
    net['loss2/top-1'] = L.Accuracy(net[from_layer_label], net['hg2_linear2'], include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    return net

def StackHourglassForImageNet(net, from_layer_data, from_layer_label, stride, stage):
    ConvBNLayer(net, from_layer_data, 'conv1', use_bn=True, use_relu=True, num_output=64, kernel_size=7, pad=3, stride=2)
    IdenMapOutHourglass(net, 'conv1', '1', 64, 128, stride) 
    net['pool1'] = L.Pooling(net['res1'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  
    IdenMapOutHourglass(net, 'pool1', '2', 128, 128, stride) 
    IdenMapOutHourglass(net, 'res2', '3', 128, 128, stride) 
    IdenMapOutHourglass(net, 'res3', '4', 128, 256, stride)

    ######First Hourglass network#####
    Hourglass(net, 'res4', 256, 512, stride, 'hg1', stage)

    ######linear layer######
    ConvBNLayer(net, 'hg1_size4_low_add', 'hg1_linear1', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg1_linear1', 'hg1_linear2', use_bn=True, use_relu=True, num_output=256, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg1_linear2', 'hg1_proj1', use_bn=False, use_relu=False, num_output=256, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg1_proj1', 'hg1_proj2', use_bn=False, use_relu=False, num_output=384, kernel_size=1, pad=0, stride=1)
  
    ######loss1#####
    ConvBNLayer(net, 'hg1_size1_low_res3', 'loss1/linear1', use_bn=True, use_relu=True, num_output=1024, kernel_size=1, pad=0, stride=1)
    net['loss1/pool'] = L.Pooling(net['loss1/linear1'], pool=P.Pooling.AVE, global_pooling=True)
    net['loss1/dropout'] = L.Dropout(net['loss1/pool'], dropout_ratio=0.5, in_place=True)
    net['loss1/classifier'] = L.InnerProduct(net['loss1/dropout'], num_output=1000, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss1/loss'] = L.SoftmaxWithLoss(net['loss1/classifier'], net[from_layer_label])
    net['loss1/top-1'] = L.Accuracy(net['loss1/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net['loss1/top-5'] = L.Accuracy(net['loss1/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')), top_k=5)

    #####Concat and Eltwise layer#####
    net['concat'] = L.Concat(net['pool1'], net['hg1_linear2'])
    ConvBNLayer(net, 'concat', 'concat_proj', use_bn=False, use_relu=False, num_output=384, kernel_size=1, pad=0, stride=1)
    net['concat_add'] = L.Eltwise(net['hg1_proj2'], net['concat_proj'])

    #####1.5 Hourglass#####
    Hourglass(net, 'concat_add', 384, 512, stride, 'hg2', stage)

    ######linear layer######
    ConvBNLayer(net, 'hg1_size4_low_add', 'hg2_linear1', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)
    ConvBNLayer(net, 'hg2_linear1', 'hg2_linear2', use_bn=True, use_relu=True, num_output=512, kernel_size=1, pad=0, stride=1)

    ######loss2#####
    ConvBNLayer(net, 'hg2_linear2', 'loss2/linear1', use_bn=True, use_relu=True, num_output=1024, kernel_size=1, pad=0, stride=1)
    net['loss2/pool'] = L.Pooling(net['loss2/linear1'], pool=P.Pooling.AVE, global_pooling=True)
    net['loss2/dropout'] = L.Dropout(net['loss2/pool'], dropout_ratio=0.5, in_place=True)
    net['loss2/classifier'] = L.InnerProduct(net['loss2/dropout'], num_output=1000, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss2/loss'] = L.SoftmaxWithLoss(net['loss2/classifier'], net[from_layer_label])
    net['loss2/top-1'] = L.Accuracy(net['loss2/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net['loss2/top-5'] = L.Accuracy(net['loss2/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')), top_k=5)
    
    return net

def Cifar10IdenMap(net, from_layer_data, from_layer_label, use_pool1, n):

    ConvBNLayer(net, from_layer_data, 'conv1', use_bn=True, use_relu=True,
                num_output=16, kernel_size=3, pad=1, stride=1)

    from_layer = 'conv1'
    for i in xrange(1, n + 1):
        block_name = '2a{}'.format(i) 
        IdenMapBody2(net, from_layer, block_name, out2a=16, out2b=16, stride=1, use_branch1=False)
        from_layer = 'res{}'.format(block_name)

    from_layer = 'res2a{}'.format(n)
    for i in xrange(1, n + 1):
        block_name = '3a{}'.format(i)
        if i == 1:
            IdenMapBody2(net, from_layer, block_name, out2a=32, out2b=32, stride=2, use_branch1=True)
        else:      
            IdenMapBody2(net, from_layer, block_name, out2a=32, out2b=32, stride=1, use_branch1=False)
        from_layer = 'res{}'.format(block_name)

    from_layer = 'res3a{}'.format(n)
    for i in xrange(1, n + 1):
        block_name = '4a{}'.format(i)
        if i == 1:
            IdenMapBody2(net, from_layer, block_name, out2a=64, out2b=64, stride=2, use_branch1=True)
        else:      
            IdenMapBody2(net, from_layer, block_name, out2a=64, out2b=64, stride=1, use_branch1=False)
        from_layer = 'res{}'.format(block_name)

    if use_pool1:
        from_layer = 'res4a{}'.format(n)
        net['pool1'] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, global_pooling=True)

    net['dropout'] = L.Dropout(net['pool1'], dropout_ratio=0.5, in_place=True)
    net['classifier'] = L.InnerProduct(net['dropout'], num_output=10, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss'] = L.SoftmaxWithLoss(net['classifier'], net[from_layer_label])
    net['accuracy'] = L.Accuracy(net['classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')),)

    return net

def Cifar10ResNet(net, from_layer_data, from_layer_label, use_pool1, n):

    ConvBNLayer(net, from_layer_data, 'conv1', use_bn=True, use_relu=True,
                num_output=16, kernel_size=3, pad=1, stride=1)

    from_layer = 'conv1'
    for i in xrange(1, n + 1):
        block_name = '2a{}'.format(i)   
        ResBodyCifar(net, from_layer, block_name, out2a=16, out2b=16, stride=1, use_branch1=False)
        from_layer = 'res{}'.format(block_name)

    from_layer = 'res2a{}'.format(n)
    for i in xrange(1, n + 1):
        block_name = '3a{}'.format(i)
        if i == 1:
            ResBodyCifar(net, from_layer, block_name, out2a=32, out2b=32, stride=2, use_branch1=True)
        else:      
            ResBodyCifar(net, from_layer, block_name, out2a=32, out2b=32, stride=1, use_branch1=False)
        from_layer = 'res{}'.format(block_name)

    from_layer = 'res3a{}'.format(n)
    for i in xrange(1, n + 1):
        block_name = '4a{}'.format(i)
        if i == 1:
            ResBodyCifar(net, from_layer, block_name, out2a=64, out2b=64, stride=2, use_branch1=True)
        else:      
            ResBodyCifar(net, from_layer, block_name, out2a=64, out2b=64, stride=1, use_branch1=False)
        from_layer = 'res{}'.format(block_name)

    if use_pool1:
        from_layer = 'res4a{}'.format(n)
        net['pool1'] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, global_pooling=True)

    net['dropout'] = L.Dropout(net['pool1'], dropout_ratio=0.5, in_place=True)
    net['classifier'] = L.InnerProduct(net['dropout'], num_output=10, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss'] = L.SoftmaxWithLoss(net['classifier'], net[from_layer_label])
    net['accuracy'] = L.Accuracy(net['classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    
    return net


def Cifar10VGG(net, from_layer_data, from_layer_label):

    ConvBNLayer(net, from_layer_data, 'conv1_1', use_bn=True, use_relu=True, num_output=64, kernel_size=3, pad=1, stride=1)
    net['conv1_1_dropout'] = L.Dropout(net['conv1_1'], dropout_ratio=0.3, in_place=True)
    ConvBNLayer(net, 'conv1_1_dropout', 'conv1_2', use_bn=True, use_relu=True, num_output=64, kernel_size=3, pad=1, stride=1)
    net['pool1'] = L.Pooling(net['conv1_2'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  

    ConvBNLayer(net, 'pool1', 'conv2_1', use_bn=True, use_relu=True, num_output=128, kernel_size=3, pad=1, stride=1)
    net['conv2_1_dropout'] = L.Dropout(net['conv2_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'conv2_1_dropout', 'conv2_2', use_bn=True, use_relu=True, num_output=128, kernel_size=3, pad=1, stride=1)
    net['pool2'] = L.Pooling(net['conv2_2'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  

    ConvBNLayer(net, 'pool2', 'conv3_1', use_bn=True, use_relu=True, num_output=256, kernel_size=3, pad=1, stride=1)
    net['conv3_1_dropout'] = L.Dropout(net['conv3_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'conv3_1_dropout', 'conv3_2', use_bn=True, use_relu=True, num_output=256, kernel_size=3, pad=1, stride=1)
    net['conv3_2_dropout'] = L.Dropout(net['conv3_2'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'conv3_2_dropout', 'conv3_3', use_bn=True, use_relu=True, num_output=256, kernel_size=3, pad=1, stride=1)
    net['pool3'] = L.Pooling(net['conv3_3'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  

    ConvBNLayer(net, 'pool3', 'conv4_1', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['conv4_1_dropout'] = L.Dropout(net['conv4_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'conv4_1_dropout', 'conv4_2', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['conv4_2_dropout'] = L.Dropout(net['conv4_2'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'conv4_2_dropout', 'conv4_3', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['pool4'] = L.Pooling(net['conv4_3'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  

    ConvBNLayer(net, 'pool4', 'conv5_1', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['conv5_1_dropout'] = L.Dropout(net['conv5_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'conv5_1_dropout', 'conv5_2', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['conv5_2_dropout'] = L.Dropout(net['conv5_2'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'conv5_2_dropout', 'conv5_3', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['pool5'] = L.Pooling(net['conv5_3'], pool=P.Pooling.MAX, kernel_size=2, stride=2) 

    net['pool5_dropout'] = L.Dropout(net['pool5'], dropout_ratio=0.5, in_place=True)
    net['fc1'] = L.InnerProduct(net['pool5_dropout'], num_output=512, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['fc1_bn'] = L.BatchNorm(net['fc1'], in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)], use_global_stats=False)
    net['fc1_scale'] = L.Scale(net['fc1_bn'], in_place=True, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], bias_term=True)
    net['fc1_relu'] = L.ReLU(net['fc1_scale'], in_place=True)
    net['fc1_dropout'] = L.Dropout(net['fc1_relu'], dropout_ratio=0.5, in_place=True)
    net['classifier'] = L.InnerProduct(net['fc1_dropout'], num_output=10, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss'] = L.SoftmaxWithLoss(net['classifier'], net[from_layer_label])
    net['accuracy'] = L.Accuracy(net['classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    return net

def Cifar10VGG_Hourglass_1_5(net, from_layer_data, from_layer_label):

    ConvBNLayer(net, from_layer_data, 'conv1_1', use_bn=True, use_relu=True, num_output=64, kernel_size=3, pad=1, stride=1)
    net['conv1_1_dropout'] = L.Dropout(net['conv1_1'], dropout_ratio=0.3, in_place=True)
    ConvBNLayer(net, 'conv1_1_dropout', 'conv1_2', use_bn=True, use_relu=True, num_output=64, kernel_size=3, pad=1, stride=1)
    net['hg1_pool1'] = L.Pooling(net['conv1_2'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  

    ConvBNLayer(net, 'hg1_pool1', 'hg1_conv2_1', use_bn=True, use_relu=True, num_output=128, kernel_size=3, pad=1, stride=1)
    net['hg1_conv2_1_dropout'] = L.Dropout(net['hg1_conv2_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg1_conv2_1_dropout', 'hg1_conv2_2', use_bn=True, use_relu=True, num_output=128, kernel_size=3, pad=1, stride=1)
    net['hg1_pool2'] = L.Pooling(net['hg1_conv2_2'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  

    ConvBNLayer(net, 'hg1_pool2', 'hg1_conv3_1', use_bn=True, use_relu=True, num_output=256, kernel_size=3, pad=1, stride=1)
    net['hg1_conv3_1_dropout'] = L.Dropout(net['hg1_conv3_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg1_conv3_1_dropout', 'hg1_conv3_2', use_bn=True, use_relu=True, num_output=256, kernel_size=3, pad=1, stride=1)
    net['hg1_conv3_2_dropout'] = L.Dropout(net['hg1_conv3_2'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg1_conv3_2_dropout', 'hg1_conv3_3', use_bn=True, use_relu=True, num_output=256, kernel_size=3, pad=1, stride=1)
    net['hg1_pool3'] = L.Pooling(net['hg1_conv3_3'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  

    ConvBNLayer(net, 'hg1_pool3', 'hg1_conv4_1', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg1_conv4_1_dropout'] = L.Dropout(net['hg1_conv4_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg1_conv4_1_dropout', 'hg1_conv4_2', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg1_conv4_2_dropout'] = L.Dropout(net['hg1_conv4_2'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg1_conv4_2_dropout', 'hg1_conv4_3', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg1_pool4'] = L.Pooling(net['hg1_conv4_3'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  

    ConvBNLayer(net, 'hg1_pool4', 'hg1_conv5_1', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg1_conv5_1_dropout'] = L.Dropout(net['hg1_conv5_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg1_conv5_1_dropout', 'hg1_conv5_2', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg1_conv5_2_dropout'] = L.Dropout(net['hg1_conv5_2'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg1_conv5_2_dropout', 'hg1_conv5_3', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg1_pool5'] = L.Pooling(net['hg1_conv5_3'], pool=P.Pooling.MAX, kernel_size=2, stride=2) 

    net['hg1_pool5_dropout'] = L.Dropout(net['hg1_pool5'], dropout_ratio=0.5, in_place=True)
    net['hg1_fc1'] = L.InnerProduct(net['hg1_pool5_dropout'], num_output=512, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['hg1_fc1_bn'] = L.BatchNorm(net['hg1_fc1'], in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)], use_global_stats=False)
    net['hg1_fc1_scale'] = L.Scale(net['hg1_fc1_bn'], in_place=True, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], bias_term=True)
    net['hg1_fc1_relu'] = L.ReLU(net['hg1_fc1_scale'], in_place=True)
    net['hg1_fc1_dropout'] = L.Dropout(net['hg1_fc1_relu'], dropout_ratio=0.5, in_place=True)
    net['loss1/classifier'] = L.InnerProduct(net['hg1_fc1_dropout'], num_output=10, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss1/loss'] = L.SoftmaxWithLoss(net['loss1/classifier'], net[from_layer_label])
    net['loss1/accuracy'] = L.Accuracy(net['loss1/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    ConvBNLayer(net, 'hg1_pool5', 'hg1_conv6_1', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg1_conv6_1_dropout'] = L.Dropout(net['hg1_conv6_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg1_conv6_1_dropout', 'hg1_conv6_2', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg1_conv6_2_dropout'] = L.Dropout(net['hg1_conv6_2'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg1_conv6_2_dropout', 'hg1_conv6_3', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    deconv_name = 'hg1_deconv1'
    net[deconv_name] = L.Deconvolution(net['hg1_conv6_3'], convolution_param=dict(num_output=512, kernel_size=2, stride=2, pad=0, bias_term=False, weight_filler=dict(type='msra')), param=[dict(lr_mult=1, decay_mult=1)])
    add_name = 'hg1_add1'
    up_layer = 'hg1_conv5_3'
    net[add_name] = L.Eltwise(net[deconv_name], net[up_layer])

    deconv_name = 'hg1_deconv2'
    net[deconv_name] = L.Deconvolution(net[add_name], convolution_param=dict(num_output=512, kernel_size=2, stride=2, pad=0, bias_term=False, weight_filler=dict(type='msra')), param=[dict(lr_mult=1, decay_mult=1)])
    add_name = 'hg1_add2'
    up_layer = 'hg1_conv4_3'
    net[add_name] = L.Eltwise(net[deconv_name], net[up_layer])

    deconv_name = 'hg1_deconv3'
    net[deconv_name] = L.Deconvolution(net[add_name], convolution_param=dict(num_output=256, kernel_size=2, stride=2, pad=0, bias_term=False, weight_filler=dict(type='msra')), param=[dict(lr_mult=1, decay_mult=1)])
    add_name = 'hg1_add3'
    up_layer = 'hg1_conv3_3'
    net[add_name] = L.Eltwise(net[deconv_name], net[up_layer])

    deconv_name = 'hg1_deconv4'
    net[deconv_name] = L.Deconvolution(net[add_name], convolution_param=dict(num_output=128, kernel_size=2, stride=2, pad=0, bias_term=False, weight_filler=dict(type='msra')), param=[dict(lr_mult=1, decay_mult=1)])
    add_name = 'hg1_add4'
    up_layer = 'hg1_conv2_2'
    net[add_name] = L.Eltwise(net[deconv_name], net[up_layer])

    net['concat'] = L.Concat(net['hg1_pool1'], net['hg1_add4'])
    ConvBNLayer(net, 'concat', 'concat_proj', use_bn=False, use_relu=False, num_output=512, kernel_size=1, pad=0, stride=1)

    ConvBNLayer(net, 'concat_proj', 'hg2_conv2_1', use_bn=True, use_relu=True, num_output=128, kernel_size=3, pad=1, stride=1)
    net['hg2_conv2_1_dropout'] = L.Dropout(net['hg2_conv2_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg2_conv2_1_dropout', 'hg2_conv2_2', use_bn=True, use_relu=True, num_output=128, kernel_size=3, pad=1, stride=1)
    net['hg2_pool2'] = L.Pooling(net['hg2_conv2_2'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  

    ConvBNLayer(net, 'hg2_pool2', 'hg2_conv3_1', use_bn=True, use_relu=True, num_output=256, kernel_size=3, pad=1, stride=1)
    net['hg2_conv3_1_dropout'] = L.Dropout(net['hg2_conv3_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg2_conv3_1_dropout', 'hg2_conv3_2', use_bn=True, use_relu=True, num_output=256, kernel_size=3, pad=1, stride=1)
    net['hg2_conv3_2_dropout'] = L.Dropout(net['hg2_conv3_2'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg2_conv3_2_dropout', 'hg2_conv3_3', use_bn=True, use_relu=True, num_output=256, kernel_size=3, pad=1, stride=1)
    net['hg2_pool3'] = L.Pooling(net['hg2_conv3_3'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  

    ConvBNLayer(net, 'hg2_pool3', 'hg2_conv4_1', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg2_conv4_1_dropout'] = L.Dropout(net['hg2_conv4_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg2_conv4_1_dropout', 'hg2_conv4_2', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg2_conv4_2_dropout'] = L.Dropout(net['hg2_conv4_2'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg2_conv4_2_dropout', 'hg2_conv4_3', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg2_pool4'] = L.Pooling(net['hg2_conv4_3'], pool=P.Pooling.MAX, kernel_size=2, stride=2)  

    ConvBNLayer(net, 'hg2_pool4', 'hg2_conv5_1', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg2_conv5_1_dropout'] = L.Dropout(net['hg2_conv5_1'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg2_conv5_1_dropout', 'hg2_conv5_2', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg2_conv5_2_dropout'] = L.Dropout(net['hg2_conv5_2'], dropout_ratio=0.4, in_place=True)
    ConvBNLayer(net, 'hg2_conv5_2_dropout', 'hg2_conv5_3', use_bn=True, use_relu=True, num_output=512, kernel_size=3, pad=1, stride=1)
    net['hg2_pool5'] = L.Pooling(net['hg2_conv5_3'], pool=P.Pooling.MAX, kernel_size=2, stride=2) 

    net['hg2_pool5_dropout'] = L.Dropout(net['hg2_pool5'], dropout_ratio=0.5, in_place=True)
    net['hg2_fc1'] = L.InnerProduct(net['hg2_pool5_dropout'], num_output=512, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['hg2_fc1_bn'] = L.BatchNorm(net['hg2_fc1'], in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)], use_global_stats=False)
    net['hg2_fc1_scale'] = L.Scale(net['hg2_fc1_bn'], in_place=True, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], bias_term=True)
    net['hg2_fc1_relu'] = L.ReLU(net['hg2_fc1_scale'], in_place=True)
    net['hg2_fc1_dropout'] = L.Dropout(net['hg2_fc1_relu'], dropout_ratio=0.5, in_place=True)
    net['loss2/classifier'] = L.InnerProduct(net['hg2_fc1_dropout'], num_output=10, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss2/loss'] = L.SoftmaxWithLoss(net['loss2/classifier'], net[from_layer_label])
    net['loss2/accuracy'] = L.Accuracy(net['loss2/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    return net

def Cifar10ResNetHourglass_1_5(net, from_layer_data, from_layer_label, n):

    ConvBNLayer(net, from_layer_data, 'conv1', use_bn=True, use_relu=True,
                num_output=16, kernel_size=3, pad=1, stride=1)

    from_layer = 'conv1'
    for i in xrange(1, n + 1):
        block_name = '2a{}'.format(i)    
        ResBodyWithPrefix(net, from_layer, block_name, out2a=16, out2b=16, stride=1, use_branch1=False, hg_prefix='hg1')
        from_layer = '{}_res{}'.format('hg1', block_name)
    
    net['hg1_pool1'] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2)  
    from_layer = 'hg1_pool1'
    for i in xrange(1, n + 1):
        block_name = '3a{}'.format(i)
        if i == 1:
            ResBodyWithPrefix(net, from_layer, block_name, out2a=32, out2b=32, stride=1, use_branch1=True, hg_prefix='hg1')
        else:      
            ResBodyWithPrefix(net, from_layer, block_name, out2a=32, out2b=32, stride=1, use_branch1=False, hg_prefix='hg1')
        from_layer = '{}_res{}'.format('hg1', block_name)

    net['hg1_pool2'] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2) 
    from_layer = 'hg1_pool2'
    for i in xrange(1, n + 1):
        block_name = '4a{}'.format(i)
        if i == 1:
            ResBodyWithPrefix(net, from_layer, block_name, out2a=64, out2b=64, stride=1, use_branch1=True, hg_prefix='hg1')
        else:      
            ResBodyWithPrefix(net, from_layer, block_name, out2a=64, out2b=64, stride=1, use_branch1=False, hg_prefix='hg1')
        from_layer = '{}_res{}'.format('hg1', block_name)

    from_layer = 'hg1_res4a{}'.format(n)
    net['hg1_pool3'] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
    net['loss1/classifier'] = L.InnerProduct(net['hg1_pool3'], num_output=10, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss1/loss'] = L.SoftmaxWithLoss(net['loss1/classifier'], net[from_layer_label])
    net['loss1/accuracy'] = L.Accuracy(net['loss1/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    
    ResBodyWithPrefix(net, 'hg1_res2a{}'.format(n), '3b', out2a=64, out2b=64, stride=1, use_branch1=True, hg_prefix='hg1')
    ResBodyWithPrefix(net, 'hg1_res3a{}'.format(n), '4b', out2a=64, out2b=64, stride=1, use_branch1=True, hg_prefix='hg1')
    
    deconv_name = 'hg1_deconv1'
    from_layer = 'hg1_res4a{}'.format(n)
    net[deconv_name] = L.Deconvolution(net[from_layer], convolution_param=dict(num_output=64, group=64, kernel_size=3, stride=2, pad=1, bias_term=False, weight_filler=dict(type='bilinear')), param=[dict(lr_mult=0, decay_mult=0)])
    add_name = 'hg1_add1'
    up_layer = 'hg1_res4b'
    net[add_name] = L.Eltwise(net[deconv_name], net[up_layer])

    deconv_name = 'hg1_deconv2'
    net[deconv_name] = L.Deconvolution(net[add_name], convolution_param=dict(num_output=64, group=64, kernel_size=3, stride=2, pad=1, bias_term=False, weight_filler=dict(type='bilinear')), param=[dict(lr_mult=0, decay_mult=0)])
    add_name = 'hg1_add2'
    up_layer = 'hg1_res3b'
    net[add_name] = L.Eltwise(net[deconv_name], net[up_layer])

    net['concat'] = L.Concat(net['conv1'], net['hg1_add2'])
#    ConvBNLayer(net, 'concat', 'concat_proj', use_bn=False, use_relu=False, num_output=128, kernel_size=1, pad=0, stride=1)

    from_layer = 'concat'
    for i in xrange(1, n + 1):
        block_name = '2a{}'.format(i)     
        ResBodyWithPrefix(net, from_layer, block_name, out2a=16, out2b=16, stride=1, use_branch1=False, hg_prefix='hg2')
        from_layer = '{}_res{}'.format('hg2', block_name)

    
    net['hg2_pool1'] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2)  
    from_layer = 'hg2_pool1'
    for i in xrange(1, n + 1):
        block_name = '3a{}'.format(i)
        if i == 1:
            ResBodyWithPrefix(net, from_layer, block_name, out2a=32, out2b=32, stride=1, use_branch1=True, hg_prefix='hg2')
        else:      
            ResBodyWithPrefix(net, from_layer, block_name, out2a=32, out2b=32, stride=1, use_branch1=False, hg_prefix='hg2')
        from_layer = '{}_res{}'.format('hg2', block_name)

    net['hg2_pool2'] = L.Pooling(net[from_layer], pool=P.Pooling.MAX, kernel_size=2, stride=2) 
    from_layer = 'hg2_pool2'
    for i in xrange(1, n + 1):
        block_name = '4a{}'.format(i)
        if i == 1:
            ResBodyWithPrefix(net, from_layer, block_name, out2a=64, out2b=64, stride=1, use_branch1=True, hg_prefix='hg2')
        else:      
            ResBodyWithPrefix(net, from_layer, block_name, out2a=64, out2b=64, stride=1, use_branch1=False, hg_prefix='hg2')
        from_layer = '{}_res{}'.format('hg2', block_name)

    from_layer = 'hg2_res4a{}'.format(n)
    net['hg2_pool3'] = L.Pooling(net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
    net['loss2/classifier'] = L.InnerProduct(net['hg2_pool3'], num_output=10, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0))
    net['loss2/loss'] = L.SoftmaxWithLoss(net['loss2/classifier'], net[from_layer_label])
    net['loss2/accuracy'] = L.Accuracy(net['loss2/classifier'], net[from_layer_label], include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    
    return net

def InceptionTower(net, from_layer, tower_name, layer_params):
    use_scale = False
    for param in layer_params:
        tower_layer = '{}/{}'.format(tower_name, param['name'])
        del param['name']
        if 'pool' in tower_layer:
            net[tower_layer] = L.Pooling(net[from_layer], **param)
        else:
            ConvBNLayer(net, from_layer, tower_layer, use_bn=True, use_relu=True,
                        use_scale=use_scale, **param)
        from_layer = tower_layer
    return net[from_layer]

def CreateAnnotatedDataLayer(source, batch_size=32, backend=P.Data.LMDB,
                             output_label=True, train=True, label_map_file='',
                             transform_param={}, batch_sampler=[{}]):
    if train:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            'transform_param': transform_param,
        }
    else:
        kwargs = {
            'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
            'transform_param': transform_param,
        }
    if output_label:
        data, label = L.AnnotatedData(name="data",
                                      annotated_data_param=dict(label_map_file=label_map_file,
                                                                batch_sampler=batch_sampler),
                                      data_param=dict(batch_size=batch_size, backend=backend, source=source),
                                      ntop=2, **kwargs)
        return [data, label]
    else:
        data = L.AnnotatedData(name="data",
                               annotated_data_param=dict(label_map_file=label_map_file,
                                                         batch_sampler=batch_sampler),
                               data_param=dict(batch_size=batch_size, backend=backend, source=source),
                               ntop=1, **kwargs)
        return data


def VGGNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
               dilated=False, nopool=False, dropout=True, freeze_layers=[]):
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(net.relu4_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool4'
        net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv5_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=6, kernel_size=3, dilation=6, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=6, kernel_size=7, dilation=2, **kwargs)
            else:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=3, kernel_size=3, dilation=3, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=3, kernel_size=7, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net

def ResBody_dilation(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1, dilat):
    # ResBody(net, 'pool1', '2a', 64, 64, 256, 1, True)

    conv_prefix = 'res{}_'.format(block_name)
    conv_postfix = ''
    bn_prefix = 'bn{}_'.format(block_name)
    bn_postfix = ''
    scale_prefix = 'scale{}_'.format(block_name)
    scale_postfix = ''
    use_scale = True

    if use_branch1:
        branch_name = 'branch1'
        ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=False,
                    num_output=out2c, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                    conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                    bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                    scale_prefix=scale_prefix, scale_postfix=scale_postfix)
        branch1 = '{}{}'.format(conv_prefix, branch_name)
    else:
        branch1 = from_layer

    branch_name = 'branch2a'
    ConvBNLayer(net, from_layer, branch_name, use_bn=True, use_relu=True,
                num_output=out2a, kernel_size=1, pad=0, stride=stride, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2b'
    ConvBNLayer_dilaiton(net, out_name, branch_name, use_bn=True, use_relu=True,
                num_output=out2b, kernel_size=3, pad=dilat, dilation=dilat, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    out_name = '{}{}'.format(conv_prefix, branch_name)

    branch_name = 'branch2c'
    ConvBNLayer(net, out_name, branch_name, use_bn=True, use_relu=False,
                num_output=out2c, kernel_size=1, pad=0, stride=1, use_scale=use_scale,
                conv_prefix=conv_prefix, conv_postfix=conv_postfix,
                bn_prefix=bn_prefix, bn_postfix=bn_postfix,
                scale_prefix=scale_prefix, scale_postfix=scale_postfix)
    branch2 = '{}{}'.format(conv_prefix, branch_name)

    res_name = 'res{}'.format(block_name)
    net[res_name] = L.Eltwise(net[branch1], net[branch2])
    relu_name = '{}_relu'.format(res_name)
    net[relu_name] = L.ReLU(net[res_name], in_place=True)


def ResNet152Body(net, from_layer, use_pool5=True):
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,
                num_output=64, kernel_size=7, pad=3, stride=2)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True)

    from_layer = 'res3a'
    for i in xrange(1, 8):
        block_name = '3b{}'.format(i)
        ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
        from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True)

    from_layer = 'res4a'
    for i in xrange(1, 36):
        block_name = '4b{}'.format(i)
        ResBody_dilation(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
        from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=2, use_branch1=True)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)

    if use_pool5:
        net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net

def InceptionV3Body(net, from_layer, output_pred=False):
    # scale is fixed to 1, thus we ignore it.
    use_scale = False

    out_layer = 'conv'
    ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                num_output=32, kernel_size=3, pad=0, stride=2, use_scale=use_scale)
    from_layer = out_layer

    out_layer = 'conv_1'
    ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                num_output=32, kernel_size=3, pad=0, stride=1, use_scale=use_scale)
    from_layer = out_layer

    out_layer = 'conv_2'
    ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                num_output=64, kernel_size=3, pad=1, stride=1, use_scale=use_scale)
    from_layer = out_layer

    out_layer = 'pool'
    net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
                               kernel_size=3, stride=2, pad=0)
    from_layer = out_layer

    out_layer = 'conv_3'
    ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                num_output=80, kernel_size=1, pad=0, stride=1, use_scale=use_scale)
    from_layer = out_layer

    out_layer = 'conv_4'
    ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                num_output=192, kernel_size=3, pad=0, stride=1, use_scale=use_scale)
    from_layer = out_layer

    out_layer = 'pool_1'
    net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
                               kernel_size=3, stride=2, pad=0)
    from_layer = out_layer

    # inceptions with 1x1, 3x3, 5x5 convolutions
    for inception_id in xrange(0, 3):
        if inception_id == 0:
            out_layer = 'mixed'
            tower_2_conv_num_output = 32
        else:
            out_layer = 'mixed_{}'.format(inception_id)
            tower_2_conv_num_output = 64
        towers = []
        tower_name = '{}'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)
        tower_name = '{}/tower'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=48, kernel_size=1, pad=0, stride=1),
            dict(name='conv_1', num_output=64, kernel_size=5, pad=2, stride=1),
        ])
        towers.append(tower)
        tower_name = '{}/tower_1'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
            dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
            dict(name='conv_2', num_output=96, kernel_size=3, pad=1, stride=1),
        ])
        towers.append(tower)
        tower_name = '{}/tower_2'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
            dict(name='conv', num_output=tower_2_conv_num_output, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)
        out_layer = '{}/join'.format(out_layer)
        net[out_layer] = L.Concat(*towers, axis=1)
        from_layer = out_layer

    # inceptions with 1x1, 3x3(in sequence) convolutions
    out_layer = 'mixed_3'
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=384, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
        dict(name='conv_2', num_output=96, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

    # inceptions with 1x1, 7x1, 1x7 convolutions
    for inception_id in xrange(4, 8):
        if inception_id == 4:
            num_output = 128
        elif inception_id == 5 or inception_id == 6:
            num_output = 160
        elif inception_id == 7:
            num_output = 192
        out_layer = 'mixed_{}'.format(inception_id)
        towers = []
        tower_name = '{}'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)
        tower_name = '{}/tower'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
            dict(name='conv_1', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
            dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        ])
        towers.append(tower)
        tower_name = '{}/tower_1'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
            dict(name='conv_1', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
            dict(name='conv_2', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
            dict(name='conv_3', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
            dict(name='conv_4', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        ])
        towers.append(tower)
        tower_name = '{}/tower_2'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
            dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)
        out_layer = '{}/join'.format(out_layer)
        net[out_layer] = L.Concat(*towers, axis=1)
        from_layer = out_layer

    # inceptions with 1x1, 3x3, 1x7, 7x1 filters
    out_layer = 'mixed_8'
    towers = []
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=320, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_3', num_output=192, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

    for inception_id in xrange(9, 11):
        num_output = 384
        num_output2 = 448
        if inception_id == 9:
            pool = P.Pooling.AVE
        else:
            pool = P.Pooling.MAX
        out_layer = 'mixed_{}'.format(inception_id)
        towers = []
        tower_name = '{}'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=320, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)

        tower_name = '{}/tower'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        ])
        subtowers = []
        subtower_name = '{}/mixed'.format(tower_name)
        subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
            dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
            dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        net[subtower_name] = L.Concat(*subtowers, axis=1)
        towers.append(net[subtower_name])

        tower_name = '{}/tower_1'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=num_output2, kernel_size=1, pad=0, stride=1),
            dict(name='conv_1', num_output=num_output, kernel_size=3, pad=1, stride=1),
        ])
        subtowers = []
        subtower_name = '{}/mixed'.format(tower_name)
        subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
            dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
            dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        net[subtower_name] = L.Concat(*subtowers, axis=1)
        towers.append(net[subtower_name])

        tower_name = '{}/tower_2'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='pool', pool=pool, kernel_size=3, pad=1, stride=1),
            dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)
        out_layer = '{}/join'.format(out_layer)
        net[out_layer] = L.Concat(*towers, axis=1)
        from_layer = out_layer

    if output_pred:
        net.pool_3 = L.Pooling(net[from_layer], pool=P.Pooling.AVE, kernel_size=8, pad=0, stride=1)
        net.softmax = L.InnerProduct(net.pool_3, num_output=1008)
        net.softmax_prob = L.Softmax(net.softmax)

    return net

def InceptionV4Body(net, from_layer, output_pred=False):
    # scale is fixed to 1, thus we ignore it.
    use_scale = False

    out_layer = 'conv'
    ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                num_output=32, kernel_size=3, pad=0, stride=2, use_scale=use_scale)
    from_layer = out_layer

    out_layer = 'conv_1'
    ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                num_output=32, kernel_size=3, pad=0, stride=1, use_scale=use_scale)
    from_layer = out_layer

    out_layer = 'conv_2'
    ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                num_output=64, kernel_size=3, pad=1, stride=1, use_scale=use_scale)
    from_layer = out_layer

    out_layer = 'pool'
    net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
                               kernel_size=3, stride=2, pad=0)
    from_layer = out_layer

    out_layer = 'conv_3'
    ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                num_output=80, kernel_size=1, pad=0, stride=1, use_scale=use_scale)
    from_layer = out_layer

    out_layer = 'conv_4'
    ConvBNLayer(net, from_layer, out_layer, use_bn=True, use_relu=True,
                num_output=192, kernel_size=3, pad=0, stride=1, use_scale=use_scale)
    from_layer = out_layer

    out_layer = 'pool_1'
    net[out_layer] = L.Pooling(net[from_layer], pool=P.Pooling.MAX,
                               kernel_size=3, stride=2, pad=0)
    from_layer = out_layer

    # inceptions with 1x1, 3x3, 5x5 convolutions
    for inception_id in xrange(0, 3):
        if inception_id == 0:
            out_layer = 'mixed'
            tower_2_conv_num_output = 32
        else:
            out_layer = 'mixed_{}'.format(inception_id)
            tower_2_conv_num_output = 64
        towers = []
        tower_name = '{}'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)
        tower_name = '{}/tower'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=48, kernel_size=1, pad=0, stride=1),
            dict(name='conv_1', num_output=64, kernel_size=5, pad=2, stride=1),
        ])
        towers.append(tower)
        tower_name = '{}/tower_1'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
            dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
            dict(name='conv_2', num_output=96, kernel_size=3, pad=1, stride=1),
        ])
        towers.append(tower)
        tower_name = '{}/tower_2'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
            dict(name='conv', num_output=tower_2_conv_num_output, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)
        out_layer = '{}/join'.format(out_layer)
        net[out_layer] = L.Concat(*towers, axis=1)
        from_layer = out_layer

    # inceptions with 1x1, 3x3(in sequence) convolutions
    out_layer = 'mixed_3'
    towers = []
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=384, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=64, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=96, kernel_size=3, pad=1, stride=1),
        dict(name='conv_2', num_output=96, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

    # inceptions with 1x1, 7x1, 1x7 convolutions
    for inception_id in xrange(4, 8):
        if inception_id == 4:
            num_output = 128
        elif inception_id == 5 or inception_id == 6:
            num_output = 160
        elif inception_id == 7:
            num_output = 192
        out_layer = 'mixed_{}'.format(inception_id)
        towers = []
        tower_name = '{}'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)
        tower_name = '{}/tower'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
            dict(name='conv_1', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
            dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        ])
        towers.append(tower)
        tower_name = '{}/tower_1'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
            dict(name='conv_1', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
            dict(name='conv_2', num_output=num_output, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
            dict(name='conv_3', num_output=num_output, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
            dict(name='conv_4', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        ])
        towers.append(tower)
        tower_name = '{}/tower_2'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='pool', pool=P.Pooling.AVE, kernel_size=3, pad=1, stride=1),
            dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)
        out_layer = '{}/join'.format(out_layer)
        net[out_layer] = L.Concat(*towers, axis=1)
        from_layer = out_layer

    # inceptions with 1x1, 3x3, 1x7, 7x1 filters
    out_layer = 'mixed_8'
    towers = []
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=320, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=192, kernel_size=[1, 7], pad=[0, 3], stride=[1, 1]),
        dict(name='conv_2', num_output=192, kernel_size=[7, 1], pad=[3, 0], stride=[1, 1]),
        dict(name='conv_3', num_output=192, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

    for inception_id in xrange(9, 11):
        num_output = 384
        num_output2 = 448
        if inception_id == 9:
            pool = P.Pooling.AVE
        else:
            pool = P.Pooling.MAX
        out_layer = 'mixed_{}'.format(inception_id)
        towers = []
        tower_name = '{}'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=320, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)

        tower_name = '{}/tower'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        ])
        subtowers = []
        subtower_name = '{}/mixed'.format(tower_name)
        subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
            dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
            dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        net[subtower_name] = L.Concat(*subtowers, axis=1)
        towers.append(net[subtower_name])

        tower_name = '{}/tower_1'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=num_output2, kernel_size=1, pad=0, stride=1),
            dict(name='conv_1', num_output=num_output, kernel_size=3, pad=1, stride=1),
        ])
        subtowers = []
        subtower_name = '{}/mixed'.format(tower_name)
        subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
            dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
            dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        net[subtower_name] = L.Concat(*subtowers, axis=1)
        towers.append(net[subtower_name])

        tower_name = '{}/tower_2'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='pool', pool=pool, kernel_size=3, pad=1, stride=1),
            dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)
        out_layer = '{}/join'.format(out_layer)
        net[out_layer] = L.Concat(*towers, axis=1)
        from_layer = out_layer

    # inceptions with 1x1, 3x3, 1x7, 7x1 filters add extra layers for ssd (conv6)
    for inception_id in xrange(11, 13):
        num_output = 192
        num_output2 = 224
        if inception_id == 12:
            pool = P.Pooling.AVE
        else:
            pool = P.Pooling.MAX
        out_layer = 'mixed_{}'.format(inception_id)
        towers = []
        tower_name = '{}'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=320, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)

        tower_name = '{}/tower'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        ])
        subtowers = []
        subtower_name = '{}/mixed'.format(tower_name)
        subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
            dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
            dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        net[subtower_name] = L.Concat(*subtowers, axis=1)
        towers.append(net[subtower_name])

        tower_name = '{}/tower_1'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=num_output2, kernel_size=1, pad=0, stride=1),
            dict(name='conv_1', num_output=num_output, kernel_size=3, pad=1, stride=1),
        ])
        subtowers = []
        subtower_name = '{}/mixed'.format(tower_name)
        subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
            dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
            dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        net[subtower_name] = L.Concat(*subtowers, axis=1)
        towers.append(net[subtower_name])

        tower_name = '{}/tower_2'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='pool', pool=pool, kernel_size=3, pad=1, stride=1),
            dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)
        out_layer = '{}/join'.format(out_layer)
        net[out_layer] = L.Concat(*towers, axis=1)
        from_layer = out_layer

    out_layer = 'mixed_13'
    towers = []
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=384, kernel_size=3, pad=1, stride=2),
    ])
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=192, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        dict(name='conv_2', num_output=192, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        dict(name='conv_3', num_output=384, kernel_size=3, pad=1, stride=2),
    ])
    towers.append(tower)
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer


    # inceptions with 1x1, 3x3, 1x7, 7x1 filters add extra layers for ssd (conv7)
    for inception_id in xrange(14, 16):
        num_output = 192
        num_output2 = 224
        if inception_id == 12:
            pool = P.Pooling.AVE
        else:
            pool = P.Pooling.MAX
        out_layer = 'mixed_{}'.format(inception_id)
        towers = []
        tower_name = '{}'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=320, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)

        tower_name = '{}/tower'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=num_output, kernel_size=1, pad=0, stride=1),
        ])
        subtowers = []
        subtower_name = '{}/mixed'.format(tower_name)
        subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
            dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        subtower = InceptionTower(net, '{}/conv'.format(tower_name), subtower_name, [
            dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        net[subtower_name] = L.Concat(*subtowers, axis=1)
        towers.append(net[subtower_name])

        tower_name = '{}/tower_1'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='conv', num_output=num_output2, kernel_size=1, pad=0, stride=1),
            dict(name='conv_1', num_output=num_output, kernel_size=3, pad=1, stride=1),
        ])
        subtowers = []
        subtower_name = '{}/mixed'.format(tower_name)
        subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
            dict(name='conv', num_output=num_output, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        subtower = InceptionTower(net, '{}/conv_1'.format(tower_name), subtower_name, [
            dict(name='conv_1', num_output=num_output, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        ])
        subtowers.append(subtower)
        net[subtower_name] = L.Concat(*subtowers, axis=1)
        towers.append(net[subtower_name])

        tower_name = '{}/tower_2'.format(out_layer)
        tower = InceptionTower(net, from_layer, tower_name, [
            dict(name='pool', pool=pool, kernel_size=3, pad=1, stride=1),
            dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        ])
        towers.append(tower)
        out_layer = '{}/join'.format(out_layer)
        net[out_layer] = L.Concat(*towers, axis=1)
        from_layer = out_layer

    out_layer = 'mixed_16'
    towers = []
    tower_name = '{}/tower'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=384, kernel_size=3, pad=1, stride=2),
    ])
    towers.append(tower)
    tower_name = '{}/tower_1'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='conv', num_output=192, kernel_size=1, pad=0, stride=1),
        dict(name='conv_1', num_output=192, kernel_size=[1, 3], pad=[0, 1], stride=[1, 1]),
        dict(name='conv_2', num_output=192, kernel_size=[3, 1], pad=[1, 0], stride=[1, 1]),
        dict(name='conv_3', num_output=384, kernel_size=3, pad=1, stride=2),
    ])
    towers.append(tower)
    tower_name = '{}'.format(out_layer)
    tower = InceptionTower(net, from_layer, tower_name, [
        dict(name='pool', pool=P.Pooling.MAX, kernel_size=3, pad=0, stride=2),
    ])
    towers.append(tower)
    out_layer = '{}/join'.format(out_layer)
    net[out_layer] = L.Concat(*towers, axis=1)
    from_layer = out_layer

    if output_pred:
        net.pool_3 = L.Pooling(net[from_layer], pool=P.Pooling.AVE, kernel_size=8, pad=0, stride=1)
        net.softmax = L.InnerProduct(net.pool_3, num_output=1008)
        net.softmax_prob = L.Softmax(net.softmax)

    return net

def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
                       use_objectness=False, normalizations=[], use_batchnorm=True,
                       min_sizes=[], max_sizes=[], prior_variance = [0.1],
                       aspect_ratios=[], share_location=True, flip=True, clip=True,
                       inter_layer_depth=0, kernel_size=1, pad=0, conf_postfix='', loc_postfix='', scales=1):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                                             across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth > 0:
            inter_name = "{}_inter".format(from_layer)
            ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True,
                        num_output=inter_layer_depth, kernel_size=3, pad=1, stride=1)
            from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        if max_sizes and max_sizes[i]:
            num_priors_per_location = 2 + len(aspect_ratio)
        else:
            num_priors_per_location = 1 + len(aspect_ratio)
        if flip:
            num_priors_per_location += len(aspect_ratio)

        num_priors_per_location *= scales

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4;
        if not share_location:
            num_loc_output *= num_classes
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False,
                    num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes;
        ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False,
                    num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        if max_sizes and max_sizes[i]:
            if aspect_ratio:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i], max_size=max_sizes[i],
                                       aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance, scale_num=scales)
            else:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i], max_size=max_sizes[i],
                                       clip=clip, variance=prior_variance, scale_num=scales)
        else:
            if aspect_ratio:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i],
                                       aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance, scale_num=scales)
            else:
                net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes[i],
                                       clip=clip, variance=prior_variance, scale_num=scales)
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2;
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False,
                        num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers
