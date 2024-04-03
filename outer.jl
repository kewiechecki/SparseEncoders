using Flux

function outerenc(z,y,x,m=3)
    kern = (3, 3)
    s = (2, 2)

    θ_conv = Chain(Conv(kern, z => 3*z, relu, stride=s),
                   Conv(kern, 3*z => 6*z, relu, stride=s),
                   Conv(kern, 6*z => 9*z, relu, stride=s),
                   Conv((2,2), 9*z => 12*z, relu))

    # Compute the output size of the last conv layer dynamically
    dim_out = convdim(y, x, θ_conv)
    dim_flat = prod(dim_out) * 12*z

    θ_mlp = Chain(Dense(dim_flat => 6, relu),
                  Dense(6 => m, relu))

    θ = Chain(θ_conv,
              x -> reshape(x, :, size(x, 4)),
              θ_mlp)
    return θ
end

function outerclassifier(m=3,k=10)
    π = Chain(Dense(m => 5, relu),
              Dense(5 => k, relu),
              softmax)
    return π
end

function outerdec(z, y, x, m=3)
    kern = (3, 3)
    s = (2, 2)

    ϕ_mlp = Chain(Dense(m => 6, relu),
                  Dense(6 => 12*z, relu))

    # Calculate the required intermediate dimensions
    intermediate_dim = calculate_intermediate_dim(y, x)

    ϕ_deconv = Chain(ConvTranspose((2,2), 12*z => 9*z, relu),
                     ConvTranspose(kern, 9*z => 6*z, relu, stride=s),
                     ConvTranspose(kern, 6*z => 3*z, relu, stride=s),
                     ConvTranspose(kern, 3*z => z, relu, stride=s))

    ϕ = Chain(ϕ_mlp,
              x -> reshape(x, intermediate_dim...),
              ϕ_deconv)
    return ϕ
end

function convdim(y, x,
                 kern_y, kern_x,
                 stride_y, stride_x,
                 pad_y, pad_x)
    y_out = floor(Int, (y - kern_y + 2 * pad_y) / stride_y) + 1
    x_out = floor(Int, (x - kern_x + 2 * pad_x) / stride_x) + 1
    return (y_out, x_out)
end

function deconvdim(y, x,
                   kern_y, kern_x,
                   stride_y, stride_x,
                   pad_y, pad_x,
                   pad_out)
    y_out = (y - 1) * stride_y - 2 * pad_y + kern_y + pad_out
    x_out = (x - 1) * stride_x - 2 * pad_x + kern_x + pad_out
    return (y_out, x_out)
end

function deconvparams(y_in, x_in,
                      y_out, x_out,
                      kern_y, kern_x,
                      stride_y, stride_x)
    # Start with output padding as 0 (common case)
    outpad_y = 0
    outpad_x = 0

    pad_y = ((y_in - 1) * stride_y + kern_y + outpad_y - y_out) / 2
    pad_x = ((x_in - 1) * stride_x + kern_x + outpad_x - x_out) / 2

    # Check if padding is an integer and non-negative, adjust output padding if necessary
    if pad_y < 0 || pad_x < 0 || pad_y % 1 != 0 || pad_x % 1 != 0
        # Adjust output padding (you might need a more sophisticated adjustment strategy)
        outpad_y = 1
        outpad_x = 1
        pad_y = ((y_in - 1) * stride_y + kern_y + outpad_y - y_out) / 2
        pad_x = ((x_in - 1) * stride_x + kern_x + outpad_x - x_out) / 2
    end

    return (Int(pad_y), Int(pad_x), outpad_y, outpad_x)
end
