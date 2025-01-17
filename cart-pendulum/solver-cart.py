import precice
import numpy as np
import os
import csv

import problemDefinition
from timeSteppers import NewmarkBeta

def main():
    participant_name = "Cart"
    config_file = "precice-config.xml"
    solver_process_index = 0
    solver_process_size = 1

    participant = precice.Participant(
        participant_name, config_file, solver_process_index, solver_process_size
    )

    write_data_name = "accCart"
    read_data_name  = "accPendulum"
    write_mesh_name = "Cart-Mesh"
    read_mesh_name  = "Pendulum-Mesh"

    # メッシュ設定
    write_dims = participant.get_mesh_dimensions(write_mesh_name)
    write_vertex = np.zeros(write_dims)
    write_vertex_id = participant.set_mesh_vertex(write_mesh_name, write_vertex)

    # 初期データ書き込み (必要な場合)
    if participant.requires_initial_data():
        participant.write_data(write_mesh_name, write_data_name, [write_vertex_id], [0.0])

    participant.initialize()

    # 台車パラメータ
    M_eff = problemDefinition.Cart.M_eff
    K_eff = problemDefinition.Cart.K_eff
    time_stepper = NewmarkBeta(M_eff, K_eff, beta=0.25, gamma=0.5)

    # 初期条件
    u = problemDefinition.Cart.u0  # 変位
    v = problemDefinition.Cart.v0  # 速度
    a = 0.0                        # 加速度 (とりあえず0で初期化)
    t = 0.0                        # 時刻
    print(f"[Cart] Initial conditions: u={u}, v={v}, a={a}, t={t}")

    # チェックポイント用バックアップ変数
    u_cp, v_cp, a_cp, t_cp = u, v, a, t

    # 結果出力用
    results = []
    # (解析解) xAnal, vAnal, thetaAnal, omegaAnal
    xA, vA, thA, wA = problemDefinition.getAnalyticalSolution(t)
    results.append((t, u, v, a, xA, vA, thA, wA))

    # ---- カップリング反復ループ ----
    while participant.is_coupling_ongoing():

        # (1) チェックポイント読み込み要求
        if participant.requires_reading_checkpoint():
            # 自前変数をバックアップから復元
            u, v, a, t = u_cp, v_cp, a_cp, t_cp

        # (2) この反復で使う “最大ステップ幅”
        precice_dt = participant.get_max_time_step_size()

        # (3) 相手 (Pendulum) の加速度を読む
        pend_acc = participant.read_data(read_mesh_name, read_data_name, [0], 0.0)
        theta_ddot = pend_acc[0] if len(pend_acc) > 0 else 0.0

        # (4) 外力 F = - m * l * θ''
        F = - problemDefinition.Pendulum.m * problemDefinition.Pendulum.l * theta_ddot

        # (5) Newmark で 1ステップ (＝ precice_dt) 進める
        u_new, v_new, a_new = time_stepper.do_step(u, v, a, F, precice_dt)
        t_new = t + precice_dt

        # (6) 自分の加速度を書き込む
        participant.write_data(write_mesh_name, write_data_name, [write_vertex_id], [a_new])

        # (7) チェックポイント書き込み要求があれば (cypreciceにはwrite_checkpointがないので注意)
        if participant.requires_writing_checkpoint():
            u_cp, v_cp, a_cp, t_cp = u_new, v_new, a_new, t_new

        # (8) 時間を1ステップ進める
        participant.advance(precice_dt)

        # (9) 状態更新
        u, v, a, t = u_new, v_new, a_new, t_new

         # --- 解析解を取得して保存 ---
        xA, vA, thA, wA = problemDefinition.getAnalyticalSolution(t)
        results.append((t, u, v, a, xA, vA, thA, wA))

    # ---- ループを抜けたあと (全カップリング終了) ----
    # 出力ディレクトリを作成
    if not os.path.exists("output"):
        os.makedirs("output")

     # CSVに書き出し
    csv_filename = "output/trajectory-Cart.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["time", "u(num)", "v(num)", "a(num)",
                         "x(anal)", "v(anal)", "theta(anal)", "omega(anal)"])
        for row in results:
            writer.writerow(row)

    print(f"[Cart] Results saved to {csv_filename}")
    participant.finalize()
    print("[Cart] Finalized participant.")

if __name__ == "__main__":
    main()
