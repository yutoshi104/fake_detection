


from tensorflow.python.keras import layers
import tensorflow as tf


def activation(fn='silu', name=None):
    if fn=='sigmoid':
        return layers.Lambda( lambda y: tf.sigmoid(y), name=name)
    return layers.Lambda( lambda y: tf.nn.silu(y), name=name)

def se(inputs, se_filters, filters, prefix):
    SE_CONV_KWARGS = {'use_bias':True, 'padding':'same', 
                    'kernel_initializer': conv_kernel_initializer,
                    'kernel_regularizer': tf.keras.regularizers.l2(mconfig.weight_decay)}
    x = layers.Lambda(lambda y:tf.reduce_mean(y, [1, 2], keepdims=True), name=prefix+'_se_mean')(inputs)
    x = layers.Conv2D(se_filters, 1, 1, name=prefix+'_se_conv2d_s', **SE_CONV_KWARGS)(x)
    x = activation(name=prefix+'_se_act_s')(x)
    x = layers.Conv2D(filters, 1, 1, name=prefix+'_se_conv2d_e', **SE_CONV_KWARGS)(x)
    x = activation('sigmoid', name=prefix+'_se_act_e')(x)
    x = layers.Multiply(name=prefix+'_se_mult')( [x ,inputs ])
    return x

def mb_conv(inputs, block_args, prefix):
    filters = block_args.input_filters * block_args.expand_ratio
    se_filters = int(block_args.input_filters*block_args.se_ratio)
    x = inputs
    if block_args.expand_ratio != 1:
        x = layers.Conv2D( filters, 1, 1, name=prefix+'_conv2d_expand', **CONV_KWARGS)(x)
        x = layers.BatchNormalization(name=prefix+'_bnorm_expand', **BN_KWARGS)(x)
        x = activation(name=prefix+'_act_expand')(x)

    x = layers.DepthwiseConv2D( block_args.kernel_size, block_args.strides, name=prefix+'_dwconv2d', **CONV_KWARGS)(x)
    x = layers.BatchNormalization(name=prefix+'_bnorm', **BN_KWARGS)(x)
    x = activation(name=prefix+'_act')(x)
    if se_filters!=0:
        x = se(x, se_filters, filters, prefix=prefix)
    x = layers.Conv2D( block_args.output_filters, 1, 1, name=prefix+'_conv2d_proj', **CONV_KWARGS)(x)
    x = layers.BatchNormalization(name=prefix+'_bnorm_proj', **BN_KWARGS)(x)

    return x

def fused_mb_conv(inputs, block_args, prefix):
    filters = block_args.input_filters * block_args.expand_ratio
    se_filters = int(block_args.input_filters*block_args.se_ratio)
    x = inputs
    if block_args.expand_ratio != 1:
        x = layers.Conv2D( filters, block_args.kernel_size, block_args.strides, 
            name=prefix+'_conv2d_expand', **CONV_KWARGS)(x)
        x = layers.BatchNormalization(name=prefix+'_bnorm_expand', **BN_KWARGS)(x)
        x = activation(name=prefix+'_act_expand')(x)
        if se_filters!=0:
            x = se(x, se_filters, filters, prefix=prefix)
        x = layers.Conv2D( block_args.output_filters, 1, 1, 
            name = prefix+'_conv2d', **CONV_KWARGS )(x)
        x = layers.BatchNormalization(name=prefix+'_bnorm', **BN_KWARGS)(x)
    else:
        if se_filters!=0:
            x = se(x, se_filters, filters, prefix=prefix)
        x = layers.Conv2D( block_args.output_filters, block_args.kernel_size, block_args.strides,
            name = prefix+'_conv2d', **CONV_KWARGS )(x)
        x = layers.BatchNormalization(name=prefix+'_bnorm', **BN_KWARGS)(x)
        x = activation(name=prefix+'_act')(x) # add act if no expansion.

    return x















    