import torch
import triton
import triton.language as tl



@triton.jit
def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)

# # 1. 定义 Kernel
# @triton.jit
# def add_kernel(
#         x_ptr,  # 输入向量 X 的首地址指针
#         y_ptr,  # 输入向量 Y 的首地址指针
#         output_ptr,  # 输出向量 Output 的首地址指针
#         n_elements,  # 向量总长度
#         BLOCK_SIZE: tl.constexpr,  # 每个 Program 处理的元素个数 (必须是 2 的幂次)
# ):
#     # 获取当前 Program 的 ID (一维网格)
#     pid = tl.program_id(axis=0)
#
#     # 计算当前 Block 负责的内存偏移量
#     block_start = pid * BLOCK_SIZE
#     offsets = block_start + tl.arange(0, BLOCK_SIZE)
#
#     # 边界保护 Mask
#     mask = offsets < n_elements
#
#     # 从 Global Memory 加载数据到 SRAM / 寄存器
#     x = tl.load(x_ptr + offsets, mask=mask)
#     y = tl.load(y_ptr + offsets, mask=mask)
#
#     # 执行加法
#     output = x + y
#
#     # 将结果写回 Global Memory
#     tl.store(output_ptr + offsets, output, mask=mask)


# 2. Python 包装层 (Launcher)
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    # 计算需要启动的 Program 数量 (Grid 维度)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 调用 Triton 内核
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# 3. 正确性验证
if __name__ == "__main__":
    size = 984321
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')

    # x = torch.rand(size, device='cpu')
    # y = torch.rand(size, device='cpu')

    triton_output = add(x, y)
    torch_output = x + y

    # 检查误差
    print(f"最大绝对误差: {torch.max(torch.abs(triton_output - torch_output))}")
    assert torch.allclose(triton_output, torch_output)
    print("向量加法验证成功！")