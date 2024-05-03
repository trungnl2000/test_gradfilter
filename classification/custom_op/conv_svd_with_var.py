import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d, avg_pool2d
import torch.nn as nn
from math import ceil


###### SVD by choosing principle components based on variance

# Cho 2 chiều
# def truncated_svd(X, var=0.9, dim=0):
#     # dim là số chiều mà mình sẽ svd theo
#     n_samples, n_features = th.prod(th.tensor(X.shape[:dim+1])), th.prod(th.tensor(X.shape[dim+1:]))
#     X_reshaped = X.view(n_samples, n_features)
#     U, S, Vt = th.linalg.svd(X_reshaped)
#     total_variance = th.sum(S**2)

#     explained_variance = th.cumsum(S**2, dim=0) / total_variance
#     k = (explained_variance >= var).nonzero()[0].item() + 1
#     # print("explained_variance: ", explained_variance)
#     # print("k: ", k)
#     return th.matmul(U[:, :k], th.diag_embed(S[:k])) , Vt[:k, :]

# def restore_tensor(Uk_Sk, Vk_t, shape):
#     reconstructed_matrix = th.matmul(Uk_Sk, Vk_t)
#     shape = tuple(shape)
#     return reconstructed_matrix.view(shape)

def truncated_svd(X, var=0.9, dim=0):
    # dim là số chiều mà mình sẽ svd theo
    n_samples, n_features = th.prod(th.tensor(X.shape[:dim+1])), th.prod(th.tensor(X.shape[dim+1:]))
    X_reshaped = X.view(n_samples, n_features)
    U, S, V = th.svd(X_reshaped)
    total_variance = th.sum(S**2)

    explained_variance = th.cumsum(S**2, dim=0) / total_variance
    k = (explained_variance >= var).nonzero()[0].item() + 1
    return th.matmul(U[:, :k], th.diag_embed(S[:k])) , V[:, :k]


def restore_tensor(Uk_Sk, Vk, shape):
    Vk_t = Vk.t()
    reconstructed_matrix = th.matmul(Uk_Sk, Vk_t)
    shape = tuple(shape)
    return reconstructed_matrix.view(shape)

###############################################################
class Conv2dSVDop_with_var(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, var = args

        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups) # Chỗ này như bình thường

        ### Kiểu svd dựa trên variance với chiều đầu tiên
        # Phân rã svd ở đây

        input_Uk_Sk, input_Vk_t = truncated_svd(input, var=var)
        ctx.save_for_backward(input_Uk_Sk, input_Vk_t, th.tensor(input.shape), weight, bias)

        # cfgs = th.tensor([groups != 1])
        # ctx.save_for_backward(input_Uk_Sk, input_Vk_t, th.tensor(input.shape), weight, bias, cfgs)


        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:

        ### Kiểu svd dựa trên variance
        input_Uk_Sk, input_Vk_t, input_shape, weight, bias = ctx.saved_tensors
        input = restore_tensor(input_Uk_Sk, input_Vk_t, input_shape)

        # input_Uk_Sk, input_Vk_t, input_shape, weight, bias, cfgs = ctx.saved_tensors
        # grouping = int(cfgs)
         
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None
        grad_output, = grad_outputs

        # ###### Tăng chiều tensor bằng cách 1
        # UkSk_in = th.zeros(input_shape[0], input_shape[1], 1, 1).to(input_Uk_Sk.device)
        # # # Lấy số lượng phần tử cần sao chép từ tensor ban đầu
        # num_elements = min(input_Uk_Sk.numel(), UkSk_in.numel())
        # # # Sao chép dữ liệu từ tensor ban đầu vào tensor mới
        # UkSk_in.view(-1)[:num_elements] = input_Uk_Sk.view(-1)[:num_elements]

        # Vk_in = th.zeros(1, input_shape[1], input_shape[2], input_shape[3]).to(input_Uk_Sk.device)
        # num_elements = min(input_Vk_t.numel(), Vk_in.numel())
        # Vk_in.view(-1)[:num_elements] = input_Vk_t.view(-1)[:num_elements]

        



        # n, c_in, p_h, p_w = input_shape[0], input_shape[1], 1, 1
        # _, c_out, gy_h, gy_w = grad_output.shape
        # k_h, k_w = weight.shape[-2:]
        # s_h, s_w = stride[0], stride[1]

        # x_h, x_w = int(input_shape[2]), int(input_shape[3])
        # x_order_h, x_order_w = x_h * stride[0], x_w * stride[1]
        # x_pad_h, x_pad_w = ceil((p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)


        # grad_y_pad_h, grad_y_pad_w = ceil((p_h * x_h - gy_h) / 2), ceil((p_w * x_w - gy_w) / 2)
        # grad_y_avg = avg_pool2d(grad_output, kernel_size=x_h, stride=x_w,
        #                         padding=(grad_y_pad_h, grad_y_pad_w),
        #                         count_include_pad=False)
        # weight_sum = weight.sum(dim=(-1, -2))

        # if grouping:
        #     grad_x_sum = grad_y_avg * weight_sum.view(1, c_out, 1, 1) # Tính tích thay vì chập

        #     grad_w_sum = (UkSk_in * grad_y_avg).sum(dim=(0, 2, 3)) # Tính Frobenius inner product
        #     print(grad_w_sum)
        #     grad_w = th.broadcast_to(grad_w_sum.view(c_out, 1, 1, 1), (c_out, 1, k_h, k_w)).clone()
        # else:
        #     grad_x_sum = (
        #         weight_sum.t() @ grad_y_avg.flatten(start_dim=2)).view(n, c_in, p_h, p_w)

        #     gy = grad_y_avg.permute(1, 0, 2, 3).flatten(start_dim=1)
        #     gx = UkSk_in.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=-2)
        #     grad_w_sum = gy @ gx
        #     print(grad_w_sum)
        #     grad_w = th.broadcast_to(grad_w_sum.view(c_out, c_in, 1, 1), (c_out, c_in, k_h, k_w)).clone()
        # grad_x = th.broadcast_to(grad_x_sum.view(n, c_in, p_h, p_w, 1, 1), (n, c_in, p_h, p_w, x_h * s_h, x_w * s_w))
        # grad_x = grad_x.permute(0, 1, 2, 4, 3, 5).reshape(
        #     n, c_in, p_h * x_h * s_h, p_w * x_w * s_w)
        # grad_x = grad_x[..., x_pad_h:x_pad_h + x_h, x_pad_w:x_pad_w + x_w]






        ####### Tăng chiều tensor bằng cách 2
        # # Tạo một tensor mới có kích thước mong muốn, ban đầu filled bằng 0
        # input = th.zeros(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).to(input_Uk_Sk.device)

        # # Copy dữ liệu từ tensor_ab vào vị trí tương ứng của tensor_new_size
        # input[:, :input_Uk_Sk.shape[1], :, :] = input_Uk_Sk.unsqueeze(-1).unsqueeze(-1)

        ########
        # grad_output_Uk_Sk, _ = truncated_svd_k(grad_output, k=1)


        # n, c_in, p_h, p_w = input_shape[0], input_shape[1], 1, 1
        # _, c_out, gy_h, gy_w = grad_output.shape
        # k_h, k_w = weight.shape[-2:]
        # s_h, s_w = stride[0], stride[1]

        # x_h, x_w = int(input_shape[2]), int(input_shape[3])
        # x_order_h, x_order_w = x_h * stride[0], x_w * stride[1]
        # x_pad_h, x_pad_w = ceil((p_h * x_order_h - x_h) / 2), ceil((p_w * x_order_w - x_w) / 2)


        # grad_y_pad_h, grad_y_pad_w = ceil((p_h * x_h - gy_h) / 2), ceil((p_w * x_w - gy_w) / 2)
        # grad_y_avg = avg_pool2d(grad_output, kernel_size=x_h, stride=x_w,
        #                         padding=(grad_y_pad_h, grad_y_pad_w),
        #                         count_include_pad=False)
        # weight_sum = weight.sum(dim=(-1, -2))

        # if grouping:
        #     grad_x_sum = grad_y_avg * weight_sum.view(1, c_out, 1, 1) # Tính tích thay vì chập

        #     grad_w_sum = (input * grad_y_avg).sum(dim=(0, 2, 3)) # Tính Frobenius inner product
        #     grad_w = th.broadcast_to(grad_w_sum.view(c_out, 1, 1, 1), (c_out, 1, k_h, k_w)).clone()
        # else:
        #     grad_x_sum = (
        #         weight_sum.t() @ grad_y_avg.flatten(start_dim=2)).view(n, c_in, p_h, p_w)

        #     gy = grad_y_avg.permute(1, 0, 2, 3).flatten(start_dim=1)
        #     gx = input.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=-2)
        #     grad_w_sum = gy @ gx
        #     grad_w = th.broadcast_to(grad_w_sum.view(c_out, c_in, 1, 1), (c_out, c_in, k_h, k_w)).clone()
        # grad_x = th.broadcast_to(grad_x_sum.view(n, c_in, p_h, p_w, 1, 1), (n, c_in, p_h, p_w, x_h * s_h, x_w * s_w))
        # grad_x = grad_x.permute(0, 1, 2, 4, 3, 5).reshape(
        #     n, c_in, p_h * x_h * s_h, p_w * x_w * s_w)
        # grad_x = grad_x[..., x_pad_h:x_pad_h + x_h, x_pad_w:x_pad_w + x_w]


        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None # Trả về gradient ứng với cái arg ở forward
        # return grad_input, grad_w, grad_bias, None, None, None, None, None

class Conv2dSVD_with_var(nn.Conv2d):
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
        super(Conv2dSVD_with_var, self).__init__(in_channels=in_channels,
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
            y = Conv2dSVDop_with_var.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, self.var)
        else:
            y = super().forward(x)
        return y

def wrap_convSVD_with_var_layer(conv, SVD_var, active):
    new_conv = Conv2dSVD_with_var(in_channels=conv.in_channels,
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