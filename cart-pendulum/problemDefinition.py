# problemDefinition.py
import numpy as np

# 物理パラメータ
M = 5.0    # 台車質量
m = 0.01    # 振り子質量
l = 4.0    # 振り子長さ
k = 20.0   # バネ剛性
g = 9.81   # 重力

# 初期条件
x0 = 0.0       # 台車初期位置
v0 = 0.1       # 台車初期速度
theta0 = 0.0   # 振り子初期角度
omega0 = -0.01  # 振り子初期角速度

# 台車 (Cart) のモデル
class Cart:
    M_eff = M + m  # 台車+振り子質量
    K_eff = k      # バネ剛性
    u0 = x0
    v0 = v0

# 振り子 (Pendulum) のモデル
class Pendulum:
    m = m
    l = l
    M_eff = l
    K_eff = g
    u0 = theta0
    v0 = omega0

# --------------------------------------------------
# ここから解析解を計算
# --------------------------------------------------

_analytical_initialized = False

# A1,B1,A2,B2, alpha1,alpha2, w1, w2 をグローバルに保持
_a1 = _b1 = _a2 = _b2 = 0.0
_alpha1 = _alpha2 = 0.0
_w1 = _w2 = 0.0

def setupAnalyticalSolution():
    """
    固有振動数 w1, w2 と 固有ベクトル比 alpha1, alpha2 を求め，
    さらに初期条件から A1,B1,A2,B2 を解く。
    """
    global _analytical_initialized
    global _a1, _b1, _a2, _b2
    global _alpha1, _alpha2, _w1, _w2

    # --- 1) w^2 の2根を求める ---
    # 2次方程式: M*l * (w^2)^2 + [-(m+M)*g - k*l]*(w^2) + k*g = 0
    Mtot = M + m
    A = M * l
    B = - (Mtot * g + k * l)
    C = k * g

    # w^2 の2つの実根
    w2_roots = np.roots([A, B, C])  # 2つの w^2
    # 小さい方, 大きい方 という形でソートしておく(負の根が出ない想定)
    w2_1, w2_2 = sorted(w2_roots)
    w1 = np.sqrt(w2_1)
    w2 = np.sqrt(w2_2)

    # --- 2) 固有ベクトル比 alpha_i = theta/x を算出 ---
    # 2次方程式の行列式から導出: alpha = w^2 / (-l w^2 + g)
    alpha1 = w2_1 / (- l * w2_1 + g)
    alpha2 = w2_2 / (- l * w2_2 + g)

    # --- 3) 一般解: 
    #  x(t) = A1 cos(w1 t) + B1 sin(w1 t) + A2 cos(w2 t) + B2 sin(w2 t)
    #  theta(t) = alpha1( ... ) + alpha2( ... )
    #
    # 初期条件 x(0)=x0, x'(0)=v0, theta(0)=theta0, theta'(0)=omega0
    #
    # t=0 での値:
    #   x(0) = A1 + A2 = x0
    #   theta(0) = alpha1*A1 + alpha2*A2 = theta0
    #
    # t=0 での速度:
    #   x'(0) = B1*w1 + B2*w2 = v0
    #   theta'(0) = alpha1*B1*w1 + alpha2*B2*w2 = omega0
    #
    # この4本を連立して A1,B1,A2,B2 を解く。

    mat = np.array([
        [ 1.0,    0.0,  1.0,    0.0    ],  # x(0)=x0
        [ 0.0, w1,  0.0,    w2    ],  # x'(0)=v0
        [ alpha1, 0.0, alpha2, 0.0 ],  # theta(0)=theta0
        [ 0.0, alpha1*w1, 0.0, alpha2*w2 ]  # theta'(0)=omega0
    ], dtype=float)

    rhs = np.array([ x0, v0, theta0, omega0 ], dtype=float)

    # 連立一次方程式を解く (A1, B1, A2, B2)
    sol = np.linalg.solve(mat, rhs)
    A1, B1, A2, B2 = sol

    # --- 4) 解をグローバル変数に保存 ---
    _a1, _b1, _a2, _b2 = A1, B1, A2, B2
    _alpha1, _alpha2 = alpha1, alpha2
    _w1, _w2 = w1, w2
    _analytical_initialized = True
    print("[AnalyticalSolution] Setup done.")
    print(f"  w1={w1:.4f}, w2={w2:.4f}")
    print(f"  alpha1={alpha1:.4f}, alpha2={alpha2:.4f}")
    print(f"  A1={A1:.4f}, B1={B1:.4f}, A2={A2:.4f}, B2={B2:.4f}")

def getAnalyticalSolution(t):
    """
    任意時刻 t に対して
      x(t), x'(t), theta(t), theta'(t)
    を返す。
    """
    if not _analytical_initialized:
        # 未初期化なら計算しておく
        setupAnalyticalSolution()

    # 呼びやすいように
    A1, B1, A2, B2 = _a1, _b1, _a2, _b2
    alpha1, alpha2 = _alpha1, _alpha2
    w1, w2 = _w1, _w2

    # x(t)
    x_t = (
        A1*np.cos(w1*t) + B1*np.sin(w1*t)
      + A2*np.cos(w2*t) + B2*np.sin(w2*t)
    )
    # x'(t)
    dx_t = (
       -A1*w1*np.sin(w1*t) + B1*w1*np.cos(w1*t)
       -A2*w2*np.sin(w2*t) + B2*w2*np.cos(w2*t)
    )
    # theta(t)
    th_t = alpha1 * (
        A1*np.cos(w1*t) + B1*np.sin(w1*t)
    ) + alpha2 * (
        A2*np.cos(w2*t) + B2*np.sin(w2*t)
    )
    # theta'(t)
    dth_t = alpha1 * (
       -A1*w1*np.sin(w1*t) + B1*w1*np.cos(w1*t)
    ) + alpha2 * (
       -A2*w2*np.sin(w2*t) + B2*w2*np.cos(w2*t)
    )

    return x_t, dx_t, th_t, dth_t
