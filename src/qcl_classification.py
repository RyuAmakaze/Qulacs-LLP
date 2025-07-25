import numpy as np
from config import USE_GPU
from qulacs import QuantumState, Observable, QuantumCircuit, ParametricQuantumCircuit
try:
    from qulacs import QuantumStateGpu
except ImportError:  # CPU-only installation
    QuantumStateGpu = QuantumState
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from qcl_utils import create_time_evol_gate, min_max_scaling, softmax

from tqdm import tqdm


class QclClassification:
    """ quantum circuit learningを用いて分類問題を解く"""
    def __init__(self, nqubit, c_depth, num_class=None):
        """
        :param nqubit: qubitの数。必要とする出力の次元数よりも多い必要がある
        :param c_depth: circuitの深さ
        :param num_class: 分類の数（=測定するqubitの数）。``None``の場合は
                          ``fit`` 時に ``y_list`` から自動で決定する。
        """
        self.nqubit = nqubit
        self.c_depth = c_depth

        self.input_state_list = []  # |ψ_in>のリスト
        self.theta = []  # θのリスト

        self.output_gate = None  # U_out

        self.num_class = num_class  # 分類の数（=測定するqubitの数）

        self.obs = None
        if self.num_class is not None:
            self._initialize_observable()

    def _initialize_observable(self):
        """num_classに応じてオブザーバブルを準備"""
        obs = [Observable(self.nqubit) for _ in range(self.num_class)]
        for i in range(len(obs)):
            obs[i].add_operator(1., f'Z {i}')
        self.obs = obs

    def create_input_gate(self, x):
        """入力$x$をエンコードする量子回路を生成する"""
        # xの要素は[-1, 1]の範囲内
        u = QuantumCircuit(self.nqubit)

        angle_y = np.arcsin(x)
        angle_z = np.arccos(x**2)

        for i in range(self.nqubit):
            idx = i % len(x)
            u.add_RY_gate(i, angle_y[idx])
            u.add_RZ_gate(i, angle_z[idx])
        
        return u

    def set_input_state(self, x_list):
        """入力状態のリストを作成"""
        x_list_normalized = min_max_scaling(x_list)  # xを[-1, 1]の範囲に特徴量ごとスケール
        
        st_list = []
        
        state_cls = QuantumStateGpu if USE_GPU else QuantumState

        for x in x_list_normalized:
            st = state_cls(self.nqubit)
            input_gate = self.create_input_gate(x)
            input_gate.update_quantum_state(st)
            st_list.append(st.copy())
        self.input_state_list = st_list

    def create_initial_output_gate(self):
        """output用ゲートU_outの組み立て&パラメータ初期値の設定"""
        u_out = ParametricQuantumCircuit(self.nqubit)
        time_evol_gate = create_time_evol_gate(self.nqubit)
        theta = 2.0 * np.pi * np.random.rand(self.c_depth, self.nqubit, 3)
        self.theta = theta.flatten()
        for d in range(self.c_depth):
            u_out.add_gate(time_evol_gate)
            for i in range(self.nqubit):
                u_out.add_parametric_RX_gate(i, theta[d, i, 0])
                u_out.add_parametric_RZ_gate(i, theta[d, i, 1])
                u_out.add_parametric_RX_gate(i, theta[d, i, 2])
        self.output_gate = u_out
    
    def update_output_gate(self, theta):
        """U_outをパラメータθで更新"""
        self.theta = theta
        parameter_count = len(self.theta)
        for i in range(parameter_count):
            self.output_gate.set_parameter(i, self.theta[i])

    def get_output_gate_parameter(self):
        """U_outのパラメータθを取得"""
        parameter_count = self.output_gate.get_parameter_count()
        theta = [self.output_gate.get_parameter(ind) for ind in range(parameter_count)]
        return np.array(theta)

    def pred(self, theta):
        """x_listに対して、モデルの出力を計算"""

        # 入力状態準備
        # st_list = self.input_state_list
        st_list = [st.copy() for st in self.input_state_list]  # ここで各要素ごとにcopy()しないとディープコピーにならない
        # U_outの更新
        self.update_output_gate(theta)

        res = []
        # 出力状態計算 & 観測
        for st in st_list:
            # U_outで状態を更新
            self.output_gate.update_quantum_state(st)
            # モデルの出力
            r = [o.get_expectation_value(st) for o in self.obs]  # 出力多次元ver
            r = softmax(r)
            res.append(r.tolist())
        return np.array(res)

    def cost_func(self, theta):
        """コスト関数を計算するクラス
        :param theta: 回転ゲートの角度thetaのリスト
        """

        y_pred = self.pred(theta)

        # cross-entropy loss
        loss = log_loss(self.y_list, y_pred)
        
        return loss

    # for BFGS
    def B_grad(self, theta):
        # dB/dθのリストを返す
        theta_plus = [theta.copy() + np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]
        theta_minus = [theta.copy() - np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]

        grad = []
        for i in tqdm(range(len(theta)), desc="param", leave=False):
            grad.append((self.pred(theta_plus[i]) - self.pred(theta_minus[i])) / 2.)

        return np.array(grad)

    # for BFGS
    def cost_func_grad(self, theta):
        y_minus_t = self.pred(theta) - self.y_list
        B_gr_list = self.B_grad(theta)
        grad = [np.sum(y_minus_t * B_gr) for B_gr in B_gr_list]
        return np.array(grad)

    def fit(self, x_list, y_list, maxiter=1000):
        """
        :param x_list: fitしたいデータのxのリスト
        :param y_list: fitしたいデータのyのリスト
        :param maxiter: scipy.optimize.minimizeのイテレーション回数
        :return: 学習後のロス関数の値
        :return: 学習後のパラメータthetaの値
        """

        # num_classが未設定ならy_listから決定
        if self.num_class is None:
            self.num_class = y_list.shape[1]
            self._initialize_observable()
        elif self.obs is None:
            self._initialize_observable()
        # y_listの次元がnum_classと一致しているか確認
        if y_list.shape[1] != self.num_class:
            raise ValueError("y_list and num_class mismatch")

        # 初期状態生成
        self.set_input_state(x_list)

        # 乱数でU_outを作成
        self.create_initial_output_gate()
        theta_init = self.theta

        # 正解ラベル
        self.y_list = y_list

        # for callbacks
        self.n_iter = 0
        self.maxiter = maxiter
        
        print("Initial parameter:")
        print(self.theta)
        print()
        print(f"Initial value of cost function:  {self.cost_func(self.theta):.4f}")
        print()
        print('============================================================')
        print("Iteration count...")
        result = minimize(self.cost_func,
                          self.theta,
                          # method='Nelder-Mead',
                          method='BFGS',
                          jac=self.cost_func_grad,
                          options={"maxiter":maxiter},
                          callback=self.callbackF)
        theta_opt = self.theta
        print('============================================================')
        print()
        print("Optimized parameter:")
        print(self.theta)
        print()
        print(f"Final value of cost function:  {self.cost_func(self.theta):.4f}")
        print()
        return result, theta_init, theta_opt

    def callbackF(self, theta):
            self.n_iter = self.n_iter + 1
            if 10 * self.n_iter % self.maxiter == 0:
                print(f"Iteration: {self.n_iter} / {self.maxiter},   Value of cost_func: {self.cost_func(theta):.4f}")


def main():
    # 乱数のシード
    random_seed = 0
    # 乱数発生器の初期化
    np.random.seed(random_seed)

    nqubit = 4  # qubitの数。入出力の次元数よりも多い必要がある
    c_depth = 2  # circuitの深さ

    # データセットの作成 (ここでは4クラス分類のダミーデータ)
    num_class = 4
    n_sample = 10
    x_list = np.random.rand(n_sample, 4)
    y_list = np.eye(num_class)[np.random.randint(num_class, size=(n_sample,))]

    # num_class=None としてインスタンス化し、データセットから自動決定
    qcl = QclClassification(nqubit, c_depth, num_class=None)

    qcl.fit(x_list, y_list)


if __name__ == "__main__":
    main()
