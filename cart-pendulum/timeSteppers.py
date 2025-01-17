class NewmarkBeta():
    def __init__(self, mass, stiffness, beta=0.25, gamma=0.5):
        self.mass = mass
        self.stiffness = stiffness
        self.beta = beta
        self.gamma = gamma

    def do_step(self, u, v, a, F, dt):
        """
        1自由度: M*ddot(u) + K*u = F(t)
        Newmark-beta法で 1ステップ進める
        """
        # 予測 (predictor)
        u_pred = u + dt*v + 0.5*(dt**2)*(1 - 2*self.beta)*a
        v_pred = v + dt*(1 - self.gamma)*a

        # 有効質量
        M_eff = self.mass + self.beta*(dt**2)*self.stiffness

        # (Fを一定とみなして) 加速度を解く
        a_new = (F - self.stiffness*u_pred) / M_eff

        # 変位・速度を補正
        u_new = u_pred + self.beta*(dt**2)*a_new
        v_new = v_pred + self.gamma*dt*a_new

        return u_new, v_new, a_new
