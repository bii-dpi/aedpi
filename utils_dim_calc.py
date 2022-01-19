width = 16
old_width = width


def get_conv_dim(dimension, kernel_size, stride, padding=0, dilation=1):
    nom = dimension + 2 * padding - dilation * (kernel_size - 1) - 1

    result = nom / stride + 1

    if int(result) != result:
        return get_conv_dim(dimension - 1, kernel_size, stride, padding, dilation)
    return  result


def get_trans_dim(dimension, kernel_size, stride, output_padding=0, padding=0, dilation=1):
    return (dimension - 1) * stride - 2 * padding + dilation * (kernel_size - 1) \
            + output_padding + 1


encoder = [(32, 4, 2),
           (64, 4, 2)]

for out_channels, kernel_size, stride in encoder:
    width = int(get_conv_dim(width, kernel_size, stride))
    print(width)

print(f"After encoding: {width ** 3 * out_channels}")

decoder = [(32, 6, 2),
           (1, 6, 2)]

width = 1

for out_channels, kernel_size, stride in decoder:
    width = get_trans_dim(width, kernel_size, stride)
    print(width)

print(f"After decoding, width = {width}")

assert width == old_width, "Does not match."

