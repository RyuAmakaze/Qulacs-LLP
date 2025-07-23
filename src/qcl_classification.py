import numpy as np
from qulacs import QuantumState, Observable, QuantumCircuit, ParametricQuantumCircuit
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from joblib import Parallel, delayed
from qcl_utils import create_time_evol_gate, min_max_scaling, softmax


def cross_entropy(p, q):
    """Return mean cross-entropy between two distributions."""
    eps = 1e-12
    q = np.clip(q, eps, 1.0)
    return -np.sum(p * np.log(q)) / len(p)


def kl_divergence(p, q):
    """Return mean KL-divergence between two distributions."""
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q)) / len(p)

from tqdm import tqdm


class QclClassification:
    """ quantum circuit learningを用いて分類問題を解く"""
    def __init__(self, nqubit, c_depth, num_class=None, n_jobs=1):
        """
        :param nqubit: qubitの数。必要とする出力の次元数よりも多い必要がある
        :param c_depth: circuitの深さ
        :param num_class: 分類の数（=測定するqubitの数）。``None``の場合は
                          ``fit`` 時に ``y_list`` から自動で決定する。
        :param n_jobs: 並列計算に使用するスレッド数
        """
        self.nqubit = nqubit
        self.c_depth = c_depth
        self.n_jobs = n_jobs

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
        
        for x in x_list_normalized:
            st = QuantumState(self.nqubit)
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

    def _pred_state(self, st):
        self.output_gate.update_quantum_state(st)
        r = [o.get_expectation_value(st) for o in self.obs]
        return softmax(r)

    def pred(self, theta):
        """x_listに対して、モデルの出力を計算"""

        # 入力状態準備
        # st_list = self.input_state_list
        st_list = [st.copy() for st in self.input_state_list]  # ここで各要素ごとにcopy()しないとディープコピーにならない
        # U_outの更新
        self.update_output_gate(theta)

        # 出力状態計算 & 観測
        if self.n_jobs == 1:
            res = [self._pred_state(st) for st in tqdm(st_list, desc="pred instance")]
        else:
            res = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self._pred_state)(st) for st in st_list
            )
        return np.array([r.tolist() for r in res])

    def bag_pred(self, theta, bag_size):
        """Return predicted class proportions for each bag."""
        inst_pred = self.pred(theta)
        num_bag = len(inst_pred) // bag_size
        bag_preds = []
        for i in range(num_bag):
            bag = inst_pred[i * bag_size : (i + 1) * bag_size]
            bag_preds.append(bag.mean(axis=0))
        return np.array(bag_preds)

    def cost_func(self, theta):
        """コスト関数を計算するクラス
        :param theta: 回転ゲートの角度thetaのリスト
        """

        y_pred = self.pred(theta)

        # cross-entropy loss
        loss = log_loss(self.y_list, y_pred)

        return loss

    def bag_cost_func(self, theta, bag_size, loss="ce"):
        """Cost function comparing bag proportions."""
        preds = self.bag_pred(theta, bag_size)
        if loss == "kl":
            return kl_divergence(self.teacher_props, preds)
        return cross_entropy(self.teacher_props, preds)

    # for BFGS
    def B_grad(self, theta):
        # dB/dθのリストを返す
        def single(i):
            th_p = theta.copy()
            th_p[i] += np.pi / 2.
            th_m = theta.copy()
            th_m[i] -= np.pi / 2.
            return (self.pred(th_p) - self.pred(th_m)) / 2.

        indices = range(len(theta))
        if self.n_jobs == 1:
            grad = [single(i) for i in indices]
        else:
            grad = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(single)(i) for i in indices
            )
        return np.array(grad)

    # for BFGS
    def cost_func_grad(self, theta):
        y_minus_t = self.pred(theta) - self.y_list
        B_gr_list = self.B_grad(theta)
        grad = [np.sum(y_minus_t * B_gr) for B_gr in B_gr_list]
        return np.array(grad)

    def bag_cost_func_grad(self, theta, bag_size, loss="ce"):
        bag_pred = self.bag_pred(theta, bag_size)
        if loss == "kl":
            y_minus_t = bag_pred - self.teacher_props
        else:
            y_minus_t = bag_pred - self.teacher_props
        B_gr_list = self.B_grad(theta)
        grad_vals = []
        num_bag = len(bag_pred)
        for B_gr in B_gr_list:
            bag_B_gr = []
            for i in range(num_bag):
                bag_B_gr.append(B_gr[i * bag_size : (i + 1) * bag_size].mean(axis=0))
            bag_B_gr = np.array(bag_B_gr)
            grad_vals.append(np.sum(y_minus_t * bag_B_gr))
        return np.array(grad_vals)

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
            if hasattr(self, "teacher_props"):
                val = self.bag_cost_func(theta, self.bag_size)
            else:
                val = self.cost_func(theta)
            print(
                f"Iteration: {self.n_iter} / {self.maxiter},   Value of cost_func: {val:.4f}"
            )

    def fit_bags(self, x_list, teacher_props, bag_size, maxiter=1000, loss="ce"):
        """Fit model using bag-level label proportions."""

        if self.num_class is None:
            self.num_class = teacher_props.shape[1]
            self._initialize_observable()
        elif self.obs is None:
            self._initialize_observable()

        if teacher_props.shape[1] != self.num_class:
            raise ValueError("teacher_props and num_class mismatch")

        self.set_input_state(x_list)
        self.create_initial_output_gate()
        theta_init = self.theta

        self.teacher_props = teacher_props
        self.bag_size = bag_size

        self.n_iter = 0
        self.maxiter = maxiter

        print("Initial parameter:")
        print(self.theta)
        print()
        print(
            f"Initial value of cost function:  {self.bag_cost_func(self.theta, bag_size, loss):.4f}"
        )
        print()
        print("============================================================")
        print("Iteration count...")
        result = minimize(
            self.bag_cost_func,
            self.theta,
            args=(bag_size, loss),
            method="BFGS",
            jac=self.bag_cost_func_grad,
            options={"maxiter": maxiter},
            callback=self.callbackF,
        )
        theta_opt = self.theta
        print("============================================================")
        print()
        print("Optimized parameter:")
        print(self.theta)
        print()
        print(
            f"Final value of cost function:  {self.bag_cost_func(self.theta, bag_size, loss):.4f}"
        )
        print()
        return result, theta_init, theta_opt


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
