import triton
import triton.language as tl
import torch
import numpy as np

@triton.jit
def apply_transformation_residual_kernel(
    src_ptr, # [n, 3]
    tgt_ptr, # [n, 3] 
    transformed_ptr, # [n, 3]
    residuals_ptr, # [n]
    s,
    R00, R01, R02,
    R10, R11, R12,
    R20, R21, R22,
    t0, t1, t2,
    n_points,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_points
    
    # 获取点坐标
    src_x = tl.load(src_ptr + offsets * 3 + 0, mask=mask)
    src_y = tl.load(src_ptr + offsets * 3 + 1, mask=mask)
    src_z = tl.load(src_ptr + offsets * 3 + 2, mask=mask)
    
    tgt_x = tl.load(tgt_ptr + offsets * 3 + 0, mask=mask)
    tgt_y = tl.load(tgt_ptr + offsets * 3 + 1, mask=mask)
    tgt_z = tl.load(tgt_ptr + offsets * 3 + 2, mask=mask)
    
    # transformed = s * (R @ p) + t
    # 转换后的点云
    transformed_x = s * (R00*src_x + R01*src_y + R02*src_z) + t0
    transformed_y = s * (R10*src_x + R11*src_y + R12*src_z) + t1
    transformed_z = s * (R20*src_x + R21*src_y + R22*src_z) + t2
    
    # 疑问：为什么要把点云又存进去？
    tl.store(transformed_ptr + offsets * 3 + 0, transformed_x, mask=mask)
    tl.store(transformed_ptr + offsets * 3 + 1, transformed_y, mask=mask)
    tl.store(transformed_ptr + offsets * 3 + 2, transformed_z, mask=mask)
    
    dx = tgt_x - transformed_x
    dy = tgt_y - transformed_y
    dz = tgt_z - transformed_z
    #残差也就是欧式距离
    residual = tl.sqrt(dx*dx + dy*dy + dz*dz)
    tl.store(residuals_ptr + offsets, residual, mask=mask)

# 加权交叉协方差矩阵 H
@triton.jit
def weighted_covariance_kernel(
    src_ptr, # [n, 3]
    tgt_ptr, # [n, 3]
    weights_ptr, # [n]
    mu_src0, mu_src1, mu_src2, 
    mu_tgt0, mu_tgt1, mu_tgt2,
    H_ptr, # [3, 3]
    n_points,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_points
    
    # 加载原点坐标
    w = tl.load(weights_ptr + offsets, mask=mask)
    src_x = tl.load(src_ptr + offsets * 3 + 0, mask=mask)
    src_y = tl.load(src_ptr + offsets * 3 + 1, mask=mask)
    src_z = tl.load(src_ptr + offsets * 3 + 2, mask=mask)
    tgt_x = tl.load(tgt_ptr + offsets * 3 + 0, mask=mask)
    tgt_y = tl.load(tgt_ptr + offsets * 3 + 1, mask=mask)
    tgt_z = tl.load(tgt_ptr + offsets * 3 + 2, mask=mask)
    
    
    # 中心化坐标
    src_centered_x = src_x - mu_src0
    src_centered_y = src_y - mu_src1
    src_centered_z = src_z - mu_src2
    
    tgt_centered_x = tgt_x - mu_tgt0
    tgt_centered_y = tgt_y - mu_tgt1
    tgt_centered_z = tgt_z - mu_tgt2
    
     # 使用 √w 技巧：(√w · s̃)(√w · t̃) = w · s̃ · t̃
    sqrt_w = tl.sqrt(w)
    weighted_src_x = src_centered_x * sqrt_w
    weighted_src_y = src_centered_y * sqrt_w
    weighted_src_z = src_centered_z * sqrt_w
    
    weighted_tgt_x = tgt_centered_x * sqrt_w
    weighted_tgt_y = tgt_centered_y * sqrt_w
    weighted_tgt_z = tgt_centered_z * sqrt_w
    

    h00 = weighted_src_x * weighted_tgt_x
    h01 = weighted_src_x * weighted_tgt_y
    h02 = weighted_src_x * weighted_tgt_z
    
    h10 = weighted_src_y * weighted_tgt_x
    h11 = weighted_src_y * weighted_tgt_y
    h12 = weighted_src_y * weighted_tgt_z
    
    h20 = weighted_src_z * weighted_tgt_x
    h21 = weighted_src_z * weighted_tgt_y
    h22 = weighted_src_z * weighted_tgt_z
    
    tl.atomic_add(H_ptr + 0, tl.sum(h00, axis=0))
    tl.atomic_add(H_ptr + 1, tl.sum(h01, axis=0))
    tl.atomic_add(H_ptr + 2, tl.sum(h02, axis=0))
    
    tl.atomic_add(H_ptr + 3, tl.sum(h10, axis=0))
    tl.atomic_add(H_ptr + 4, tl.sum(h11, axis=0))
    tl.atomic_add(H_ptr + 5, tl.sum(h12, axis=0))
    
    tl.atomic_add(H_ptr + 6, tl.sum(h20, axis=0))
    tl.atomic_add(H_ptr + 7, tl.sum(h21, axis=0))
    tl.atomic_add(H_ptr + 8, tl.sum(h22, axis=0))

# Huber损失计算
@triton.jit
def compute_huber_weights_kernel(
    residuals_ptr,
    weights_ptr,
    delta,
    n_points,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_points
    
    # 加载残差
    r = tl.load(residuals_ptr + offsets, mask=mask)
    
    # 抑制离群点
    weight = tl.where(r > delta, delta / r, 1.0)
    
    tl.store(weights_ptr + offsets, weight, mask=mask)


"""
有些类似于C++中的TBB并行方法
"""
@triton.jit # JIT 编译装饰器，将 Python 代码编译为 GPU 机器码
def weighted_mean_kernel(
    points_ptr, # [n, 3]
    weights_ptr, # [n]
    mean_ptr, # [sum(w*x), sum(w*y), sum(w*z), sum(w)]
    n_points,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_points
    
    w = tl.load(weights_ptr + offsets, mask=mask)
    x = tl.load(points_ptr + offsets * 3 + 0, mask=mask)
    y = tl.load(points_ptr + offsets * 3 + 1, mask=mask)
    z = tl.load(points_ptr + offsets * 3 + 2, mask=mask)
    
    wx = w * x
    wy = w * y
    wz = w * z
    
    tl.atomic_add(mean_ptr + 0, tl.sum(wx, axis=0))
    tl.atomic_add(mean_ptr + 1, tl.sum(wy, axis=0))
    tl.atomic_add(mean_ptr + 2, tl.sum(wz, axis=0))
    tl.atomic_add(mean_ptr + 3, tl.sum(w, axis=0))

"""
Todo hfx:用于计算sim3变换后的点云残差？
"""
def apply_transformation_residual_triton(src, tgt, s, R, t):
    n_points = src.shape[0]
    
    transformed = torch.empty_like(src)
    residuals = torch.empty(n_points, device=src.device, dtype=src.dtype)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_points, BLOCK_SIZE),)
    
    R_flat = R.contiguous().view(-1)
    t_flat = t.contiguous().view(-1)
    
    # 启动内核
    apply_transformation_residual_kernel[grid](
        src, tgt, transformed, residuals,
        float(s),
        float(R_flat[0]), float(R_flat[1]), float(R_flat[2]),
        float(R_flat[3]), float(R_flat[4]), float(R_flat[5]),
        float(R_flat[6]), float(R_flat[7]), float(R_flat[8]),
        float(t_flat[0]), float(t_flat[1]), float(t_flat[2]),
        n_points,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return transformed, residuals
"""
Todo hfx:
好奇里面的这个triton是什么意思 -> 在这段代码中指的是 OpenAI Triton，一个用于编写高效 GPU 内核的编程语言和编译器。
"""
def compute_weighted_mean_triton(points, weights):
    n_points = points.shape[0]
    
    # [sum(w*x), sum(w*y), sum(w*z), sum(w)] 
    # 加全求和
    mean_buffer = torch.zeros(4, device=points.device, dtype=points.dtype)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_points, BLOCK_SIZE),)
    
    #这个是什么形式的代码？类似于TBB并发了
    weighted_mean_kernel[grid](
        points, weights, mean_buffer, n_points, BLOCK_SIZE=BLOCK_SIZE
    )
    
    total_weight = mean_buffer[3]
    if total_weight > 1e-12:
        mean = mean_buffer[:3] / total_weight
    else:
        mean = torch.zeros(3, device=points.device, dtype=points.dtype)
    
    return mean, total_weight

"""
Todo hfx:
计算两者的协方差矩阵吗，用于求解旋转变换
"""
def compute_weighted_covariance_triton(src, tgt, weights, mu_src, mu_tgt):
    n_points = src.shape[0]
    
    H = torch.zeros(9, device=src.device, dtype=src.dtype)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_points, BLOCK_SIZE),)
    
    # .contiguous() 确保在内存中是连续存储的，展平为一维
    mu_src_flat = mu_src.contiguous().view(-1)
    mu_tgt_flat = mu_tgt.contiguous().view(-1)
    
    weighted_covariance_kernel[grid](
        src, tgt, weights,
        float(mu_src_flat[0]), float(mu_src_flat[1]), float(mu_src_flat[2]),
        float(mu_tgt_flat[0]), float(mu_tgt_flat[1]), float(mu_tgt_flat[2]),
        H, n_points, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return H.reshape(3, 3)
"""
Todo hfx:
用于计算Huber
"""
def compute_huber_weights_triton(residuals, delta):
    n_points = residuals.shape[0]
    weights = torch.empty_like(residuals)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_points, BLOCK_SIZE),)
    
    compute_huber_weights_kernel[grid](
        residuals, weights, float(delta), n_points, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return weights


def weighted_estimate_se3_triton(source_points, target_points, weights):

    source_points = torch.from_numpy(source_points).cuda().float()
    target_points = torch.from_numpy(target_points).cuda().float()
    weights = torch.from_numpy(weights).cuda().float()
    
    total_weight = torch.sum(weights)
    if total_weight < 1e-6:
        return 1.0, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros((3, 3), dtype=np.float32)
    
    normalized_weights = weights / total_weight
    
    mu_src, _ = compute_weighted_mean_triton(source_points, normalized_weights)
    mu_tgt, _ = compute_weighted_mean_triton(target_points, normalized_weights)
    
    H = compute_weighted_covariance_triton(
        source_points, target_points, normalized_weights, mu_src, mu_tgt
    )
    
    return 1.0, mu_src.cpu().numpy(), mu_tgt.cpu().numpy(), H.cpu().numpy()
"""
Todo hfx: 这才是真正的计算吗？
weight的作用是什么?(本质上是点云的置信度)
"""
def weighted_estimate_sim3_triton(source_points, target_points, weights):

    source_points = torch.from_numpy(source_points).cuda().float()
    target_points = torch.from_numpy(target_points).cuda().float()
    weights = torch.from_numpy(weights).cuda().float()
    
    total_weight = torch.sum(weights)
    # 置信度过低则不继续计算
    if total_weight < 1e-6:
        return -1.0, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros((3, 3), dtype=np.float32)
    
    normalized_weights = weights / total_weight
    
    #输出的mu_xxx实际上就是除以权重的点云均值( sum(w*point)/sum(w) )
    mu_src, _ = compute_weighted_mean_triton(source_points, normalized_weights)
    mu_tgt, _ = compute_weighted_mean_triton(target_points, normalized_weights)
    
    # 点云中心吗？ 应该是偏移之后的点云（中心化？）
    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt
    
    #应该是计算两个点云的尺度大小，对于尺度处理其实可以看看Lidar-VGGT如何实现的
    scale_src = torch.sqrt(torch.sum(normalized_weights * torch.sum(src_centered**2, dim=1)))
    scale_tgt = torch.sqrt(torch.sum(normalized_weights * torch.sum(tgt_centered**2, dim=1)))
    s = scale_tgt / scale_src #计算src->tgt的缩放因子
    
    weighted_src = s * src_centered # 将src缩放到tgt的尺度

    # 加权交叉协方差矩阵 H，用于点云配准中的旋转估计。
    H = compute_weighted_covariance_triton(
        weighted_src, tgt_centered, normalized_weights, torch.zeros_like(mu_src), torch.zeros_like(mu_tgt)
    )
    # s 缩放因子
    # mu_src mu_tgt 中心化点云
    # H 加权交叉协方差矩阵
    return s.cpu().numpy(), mu_src.cpu().numpy(), mu_tgt.cpu().numpy(), H.cpu().numpy()

"""
Todo hfx: 带权重的去计算两个点云的 sim3 变换
"""
def weighted_estimate_sim3_numba_triton(source_points, target_points, weights, align_method='sim3'):

    if align_method == 'sim3':
        s, mu_src, mu_tgt, H = weighted_estimate_sim3_triton(source_points, target_points, weights)
    elif align_method == 'se3' or align_method == 'scale+se3':
        s, mu_src, mu_tgt, H = weighted_estimate_se3_triton(source_points, target_points, weights)
    
    if s < 0:
        raise ValueError("Total weight too small for meaningful estimation")
    
    # 通过SVD求解旋转估计矩阵R
    H_torch = torch.from_numpy(H).cuda().float()
    U, _, Vt = torch.linalg.svd(H_torch)
    
    U = U.cpu().numpy()
    Vt = Vt.cpu().numpy()
    
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    mu_src = mu_src.astype(np.float32)
    mu_tgt = mu_tgt.astype(np.float32)
    R = R.astype(np.float32)
    
    if align_method == 'se3' or align_method == 'scale+se3':
        t = mu_tgt - R @ mu_src
    else:
        t = mu_tgt - s * R @ mu_src
    
    return s, R, t.astype(np.float32)
"""
Todo hfx: 
把处理过的点云传入进行对准
使用 迭代重加权最小二乘法 (IRLS) 结合 Huber 损失 来鲁棒估计两组点云之间 Sim3 变换的函数。
"""
def robust_weighted_estimate_sim3_triton(src, tgt, init_weights, delta=0.1, max_iters=20, tol=1e-9, align_method='sim3'):

    src = src.astype(np.float32)
    tgt = tgt.astype(np.float32)
    init_weights = init_weights.astype(np.float32)
    
    src_torch = torch.from_numpy(src).cuda().float()
    tgt_torch = torch.from_numpy(tgt).cuda().float()
    init_weights_torch = torch.from_numpy(init_weights).cuda().float()
    
    # 初始变换估计（使用初始权重）
    s, R, t = weighted_estimate_sim3_numba_triton(src, tgt, init_weights, align_method=align_method)
    
    R_torch = torch.from_numpy(R).cuda().float()
    t_torch = torch.from_numpy(t).cuda().float()
    s_torch = torch.tensor(s, device='cuda', dtype=torch.float32)
    
    prev_error = float('inf')
    
    for iter in range(max_iters):
        # 实际上 transformed 没有用，可以不保留
        transformed, residuals = apply_transformation_residual_triton(
            src_torch, tgt_torch, s_torch, R_torch, t_torch
        )
        
        #计算平均残差
        mean_residual = torch.mean(residuals).cpu().numpy()
        print(f'Iter {iter}: Mean residual = {mean_residual:.6f}')
        # Huber 损失是一种结合了 MSE 和 MAE 优点的鲁棒损失函数，广泛用于回归问题和点云配准。
        # 这只是权重计算
        huber_weights = compute_huber_weights_triton(residuals, delta)
        
        # 初始权重 * Huber损失
        combined_weights = init_weights_torch * huber_weights
        combined_weights_sum = torch.sum(combined_weights)
        if combined_weights_sum > 1e-12:
            combined_weights /= combined_weights_sum # 均值化
        else:
            combined_weights = init_weights_torch / torch.sum(init_weights_torch)
        
        combined_weights_np = combined_weights.cpu().numpy()
        # 通过优化权重（置信度）来实现点云配准
        s_new, R_new, t_new = weighted_estimate_sim3_numba_triton(
            src, tgt, combined_weights_np, align_method=align_method
        )
        
        # rot_angle 可以理解是角度变换，但是param_change为啥要和平移耦合在一起不分开？
        param_change = np.abs(s_new - s) + np.linalg.norm(t_new - t)
        rot_angle = np.arccos(min(1.0, max(-1.0, (np.trace(R_new @ R.T) - 1)/2)))
        
        residuals_np = residuals.cpu().numpy()
        # 如果残差小于阈值，则
        huber_loss_values = np.where(
            residuals_np <= delta,
            0.5 * residuals_np**2,  # 小残差：MSE
            delta * (residuals_np - 0.5 * delta)  # 大残差：线性
        )
        # 总误差吗
        current_error = np.sum(huber_loss_values * init_weights)
        # tol 容差/收敛阈值
        # 判断是否收敛
        if (param_change < tol and rot_angle < np.radians(0.1)) or \
           (abs(prev_error - current_error) < tol * prev_error):
            print(f'Converged at iteration {iter}')
            break
        
        s, R, t = s_new, R_new, t_new
        s_torch = torch.tensor(s, device='cuda', dtype=torch.float32)
        R_torch = torch.from_numpy(R).cuda().float()
        t_torch = torch.from_numpy(t).cuda().float()
        prev_error = current_error
    
    return s, R, t

def warmup_triton():
    print("\nWarming up Triton functions...")
    
    n_points = 10000
    src = np.random.randn(n_points, 3).astype(np.float32)
    tgt = np.random.randn(n_points, 3).astype(np.float32)
    weights = np.ones(n_points, dtype=np.float32)
    
    src_torch = torch.from_numpy(src).cuda().float()
    tgt_torch = torch.from_numpy(tgt).cuda().float()
    weights_torch = torch.from_numpy(weights).cuda().float()
    
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    s = np.float32(1.0)
    delta = np.float32(0.1)
    
    R_torch = torch.from_numpy(R).cuda().float()
    t_torch = torch.from_numpy(t).cuda().float()
    s_torch = torch.tensor(s, device='cuda', dtype=torch.float32)
    
    try:
        _, _ = apply_transformation_residual_triton(src_torch, tgt_torch, s_torch, R_torch, t_torch)
        print(" - apply_transformation_residual_triton warmed up.")
    except Exception as e:
        print(f" ! Failed to warm up apply_transformation_residual_triton: {e}")
    
    try:
        _, _ = compute_weighted_mean_triton(src_torch, weights_torch)
        print(" - compute_weighted_mean_triton warmed up.")
    except Exception as e:
        print(f" ! Failed to warm up compute_weighted_mean_triton: {e}")
    
    try:
        mu_src, _ = compute_weighted_mean_triton(src_torch, weights_torch)
        mu_tgt, _ = compute_weighted_mean_triton(tgt_torch, weights_torch)
        _ = compute_weighted_covariance_triton(src_torch, tgt_torch, weights_torch, mu_src, mu_tgt)
        print(" - compute_weighted_covariance_triton warmed up.")
    except Exception as e:
        print(f" ! Failed to warm up compute_weighted_covariance_triton: {e}")
    
    try:
        residuals = torch.abs(torch.randn(n_points, device='cuda', dtype=torch.float32))
        _ = compute_huber_weights_triton(residuals, delta)
        print(" - compute_huber_weights_triton warmed up.")
    except Exception as e:
        print(f" ! Failed to warm up compute_huber_weights_triton: {e}")
    
    print("Triton warm-up complete.\n")

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

if __name__ == "__main__":

    warmup_triton()
    
    n_points = 7_500_000
    src = np.random.randn(n_points, 3).astype(np.float32)
    
    true_R = np.array([[0.866, -0.5, 0],
                      [0.5, 0.866, 0],
                      [0, 0, 1]], dtype=np.float32)
    true_t = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    true_s = 1.2
    
    tgt = true_s * (src @ true_R.T) + true_t
    tgt += 0.01 * np.random.randn(*tgt.shape).astype(np.float32)
    
    weights = np.ones(n_points, dtype=np.float32)
    
    print_gpu_memory()
    
    s, R, t = robust_weighted_estimate_sim3_triton(
        src, tgt, weights, 
        delta=0.1, max_iters=5, align_method='sim3'
    )
    
    print(f"\nEstimated scale: {s:.6f}")
    print(f"Estimated rotation:\n{R}")
    print(f"Estimated translation: {t}")
    
    print_gpu_memory()