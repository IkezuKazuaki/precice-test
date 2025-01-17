import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font for graphs (you can changeフォントに日本語対応が必要です)
rcParams['font.family'] = 'IPAGothic'  # または 'IPAMincho'


def plot_trajectories(cart_csv, pendulum_csv):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2行2列のグラフ配置

    if os.path.exists(cart_csv):
        # Read Cart CSV into DataFrame
        df_cart = pd.read_csv(cart_csv, delimiter=';')

        # Extract time and relevant columns
        time = df_cart['time']
        u_num = df_cart['u(num)']
        x_anal = df_cart['x(anal)']
        v_num = df_cart['v(num)']
        v_anal = df_cart['v(anal)']

        # 左上: 台車の変位 vs 時間
        axs[0, 0].plot(time, u_num, label='数値解', color='blue', linestyle='-')
        axs[0, 0].plot(time, x_anal, label='解析解', color='blue', linestyle='--')
        axs[0, 0].set_title("台車の変位", fontsize=14)
        axs[0, 0].set_xlabel("時間 [s]", fontsize=12)
        axs[0, 0].set_ylabel("変位 [m]", fontsize=12)
        axs[0, 0].legend(fontsize=12, loc='upper right')  # 凡例を右上に固定
        axs[0, 0].grid(True)

        # 左下: 台車の速度 vs 時間
        axs[1, 0].plot(time, v_num, label='数値解', color='orange', linestyle='-')
        axs[1, 0].plot(time, v_anal, label='解析解', color='orange', linestyle='--')
        axs[1, 0].set_title("台車の速度", fontsize=14)
        axs[1, 0].set_xlabel("時間 [s]", fontsize=12)
        axs[1, 0].set_ylabel("速度 [m/s]", fontsize=12)
        axs[1, 0].legend(fontsize=12, loc='upper right')  # 凡例を右上に固定
        axs[1, 0].grid(True)
    else:
        print(f"ファイルが見つかりません: {cart_csv}")

    if os.path.exists(pendulum_csv):
        # Read Pendulum CSV into DataFrame
        df_pendulum = pd.read_csv(pendulum_csv, delimiter=';')

        # Extract time and relevant columns
        time = df_pendulum['time']
        theta_num = df_pendulum['theta(num)']
        theta_anal = df_pendulum['theta(anal)']
        omega_num = df_pendulum['omega(num)']
        omega_anal = df_pendulum['omega(anal)']

        # 右上: 振り子の角度 vs 時間
        axs[0, 1].plot(time, theta_num, label='数値解', color='green', linestyle='-')
        axs[0, 1].plot(time, theta_anal, label='解析解', color='green', linestyle='--')
        axs[0, 1].set_title("振り子の角度", fontsize=14)
        axs[0, 1].set_xlabel("時間 [s]", fontsize=12)
        axs[0, 1].set_ylabel("角度 [rad]", fontsize=12)
        axs[0, 1].legend(fontsize=12, loc='upper right')  # 凡例を右上に固定
        axs[0, 1].grid(True)

        # 右下: 振り子の角速度 vs 時間
        axs[1, 1].plot(time, omega_num, label='数値解', color='purple', linestyle='-')
        axs[1, 1].plot(time, omega_anal, label='解析解', color='purple', linestyle='--')
        axs[1, 1].set_title("振り子の角速度", fontsize=14)
        axs[1, 1].set_xlabel("時間 [s]", fontsize=12)
        axs[1, 1].set_ylabel("角速度 [rad/s]", fontsize=12)
        axs[1, 1].legend(fontsize=12, loc='upper right')  # 凡例を右上に固定
        axs[1, 1].grid(True)
    else:
        print(f"ファイルが見つかりません: {pendulum_csv}")

    plt.tight_layout()
    plt.show()

def main():
    # Assuming the output folder is in the same directory
    cart_csv = "output/trajectory-Cart.csv"
    pendulum_csv = "output/trajectory-Pendulum.csv"

    plot_trajectories(cart_csv, pendulum_csv)

if __name__ == "__main__":
    main()
