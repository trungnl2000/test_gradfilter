import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d
import torch.nn as nn


#################### Average along batch size dimension ###################################
# 'var' variable is unused, I havent removed it yet.

def conv_avg_batch(X): # Tính trung bình theo chiều đầu tiên
    return th.mean(X, dim=0)

def restore_tensor(X_avg_batch, shape):
    shape = tuple(shape)
    # return th.broadcast_to(X_avg_batch, shape)
    return X_avg_batch.unsqueeze(0)

# def restore_tensor_keep_dim(X_avg_batch, shape):
#     shape = tuple(shape)
#     return th.broadcast_to(X_avg_batch, shape)
#     # return X_avg_batch.unsqueeze(0)

###############################################################
class Conv2d_Avg_Batch_op(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, var = args

        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups) # Chỗ này như bình thường

        ### Kiểu svd dựa trên variance với chiều đầu tiên
        # Phân rã svd ở đây

        input_avg_batch = conv_avg_batch(input)
        ctx.save_for_backward(input_avg_batch, th.tensor(input.shape), weight, bias)

        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:

        ### Kiểu svd dựa trên variance
        input_avg_batch, input_shape, weight, bias = ctx.saved_tensors
        input = restore_tensor(input_avg_batch, input_shape)
        # input_ = restore_tensor_keep_dim(input_avg_batch, input_shape)
        # print("weight ", weight.shape)
        # print("input ", input.shape)

        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None
        grad_output, = grad_outputs
        
        # _, c_in, _, _ = input.shape
        # _, c_out, _, _ = grad_output.shape
        # k_h, k_w = weight.shape[-2:]

        grad_output_ = th.mean(grad_output, dim=0).unsqueeze(0)

        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input((input_shape[0], input_shape[1], input_shape[2], input_shape[3]), weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = nn.grad.conv2d_weight(input, weight.shape, grad_output_, stride, padding, dilation, groups)
            # if groups != 1:
            #     grad_w_sum = (input * grad_output_).sum(dim=(0, 2, 3)) # Tính Frobenius inner product
            #     grad_weight = th.broadcast_to(grad_w_sum.view(c_out, 1, 1, 1), (c_out, 1, k_h, k_w)).clone()
            # else:
            #     gy = grad_output_.permute(1, 0, 2, 3).flatten(start_dim=1)
            #     gx = input.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=-2)
            #     grad_w_sum = gy @ gx
            #     grad_weight = th.broadcast_to(grad_w_sum.view(c_out, c_in, 1, 1), (c_out, c_in, k_h, k_w)).clone()
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)
        
        # shape_grad_weight = (weight.shape[0], grad_weight.shape[1], grad_weight.shape[2], grad_weight.shape[3])
        # grad_weight = th.broadcast_to(grad_weight, shape_grad_weight)
        # print("grad_weight: ", grad_weight.shape)
        # print("grad_input: ", grad_input)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None # Trả về gradient ứng với cái arg ở forward


class Conv2d_Avg_Batch(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            padding=0,
            device=None,
            dtype=None,
            activate=False,
            var=1
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        # assert padding[0] == kernel_size[0] // 2 and padding[1] == kernel_size[1] // 2
        super(Conv2d_Avg_Batch, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding=padding,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)
        self.activate = activate
        self.var = var

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x, weight, bias, stride, padding, order, groups = args
        if self.activate:
            y = Conv2d_Avg_Batch_op.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, self.var)
        else:
            y = super().forward(x)
        return y

def wrap_conv_Avg_Batch_layer(conv, SVD_var, active):
    new_conv = Conv2d_Avg_Batch(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active,
                         var=SVD_var
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv