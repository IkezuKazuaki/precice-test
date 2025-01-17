import precice
import numpy as np
import os
import csv

import problemDefinition
from timeSteppers import NewmarkBeta
import math

def main():
    participant_name = "Pendulum"
    config_file = "precice-config.xml"
    solver_process_index = 0
    solver_process_size = 1

    participant = precice.Participant(
        participant_name, config_file, solver_process_index, solver_process_size
    )

    write_data_name = "accPendulum"
    read_data_name  = "accCart"
    write_mesh_name = "Pendulum-Mesh"
    read_mesh_name  = "Pendulum-Mesh"

    # メッシュ設定
    write_dims = participant.get_mesh_dimensions(write_mesh_name)
    write_vertex = np.zeros(write_dims)
    write_vertex_id = participant.set_mesh_vertex(write_mesh_name, write_vertex)

    # 初期データ書き込み (必要な場合)
    if participant.requires_initial_data():
        participant.write_data(
            write_mesh_name, write_data_name, [write_vertex_id], [0.0]
        )

    participant.initialize()

    # 振り子パラメータ
    M_eff = problemDefinition.Pendulum.M_eff
    K_eff = problemDefinition.Pendulum.K_eff
    time_stepper = NewmarkBeta(M_eff, K_eff, beta=0.25, gamma=0.5)

    # 初期条件
    u = problemDefinition.Pendulum.u0   # θ
    v = problemDefinition.Pendulum.v0   # θ'
    a = 0.0
    t = 0.0
    print(f"[Pendulum] Initial conditions: u={u}, v={v}, a={a}, t={t}")

    # チェックポイント用バックアップ
    u_cp, v_cp, a_cp, t_cp = u, v, a, t

    # -- 結果を保存するリスト
    results = []
    # (解析解) xAnal, vAnal, thAnal, wAnal をとってくるが，
    # ここでは振り子成分だけ CSV に書く
    xA, vA, thA, wA = problemDefinition.getAnalyticalSolution(t)
    results.append((t, u, v, a, thA, wA))

    while participant.is_coupling_ongoing():

        # (1) checkpoint 読み込み要求
        if participant.requires_reading_checkpoint():
            u, v, a, t = u_cp, v_cp, a_cp, t_cp

        # (2) 最大ステップサイズ
        precice_dt = participant.get_max_time_step_size()

        # (3) Cart の加速度 (x'') を読む
        cart_acc = participant.read_data(
            read_mesh_name, read_data_name, [write_vertex_id], 0.0
        )
        x_ddot = cart_acc[0] if len(cart_acc) else 0.0

        # (4) 外力 F = - x''
        F = - x_ddot

        # (5) Newmark で1ステップ進める (precice_dt)
        u_new, v_new, a_new = time_stepper.do_step(u, v, a, F, precice_dt)
        t_new = t + precice_dt

        # (6) θ'' を書き込む
        participant.write_data(
            write_mesh_name, write_data_name, [write_vertex_id], [a_new]
        )

        # (8) checkpoint 書き込み要求
        if participant.requires_writing_checkpoint():
            u_cp, v_cp, a_cp, t_cp = u_new, v_new, a_new, t_new

        # (7) カップリングを1ステップ進める
        participant.advance(precice_dt)

        # (9) 状態更新
        u, v, a, t = u_new, v_new, a_new, t_new

        # 解析解を取得
        xA, vA, thA, wA = problemDefinition.getAnalyticalSolution(t)
        results.append((t, u, v, a, thA, wA))

    # 出力ディレクトリを作成
    if not os.path.exists("output"):
        os.makedirs("output")

    # CSVに書き出し
    csv_filename = "output/trajectory-Pendulum.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["time", "theta(num)", "omega(num)", "a(num)",
                         "theta(anal)", "omega(anal)"])
        for (tt, uu, vv, aa, tha, wa) in results:
            writer.writerow([tt, uu, vv, aa, tha, wa])

    print(f"[Pendulum] Results saved to {csv_filename}")
    participant.finalize()
    # 固有振動数を出力
    problemDefinition.setupAnalyticalSolution()
    w1 = problemDefinition._w1
    w2 = problemDefinition._w2
    T1 = 2 * math.pi / w1
    T2 = 2 * math.pi / w2
    print(f"[Solver] Eigenfrequencies: w1 = {w1:.4f}, w2 = {w2:.4f}")
    print(f"[Solver] Eigenperiods: T1 = {T1:.4f}, T2 = {T2:.4f}")
    print("[Pendulum] Finalized participant.")

if __name__ == "__main__":
    main()
