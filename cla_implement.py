from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


class CLA:
    def __init__(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ):
        # initialize
        self.mean = mean  # [n_assets, 1]
        self.cov = cov  # [n_assets, n_assets]
        self.lower_bounds = lower_bounds  # [n_assets, 1]
        self.upper_bounds = upper_bounds  # [n_assets, 1]

        # save result
        self.weights = []
        self.lambdas = []
        self.gammas = []
        self.free_weights = []
        self.__run_algorithm()

    def __get_matrices(self, asset_list):
        """
        Get sliced matrices for the selected assets
        μ_F, Σ_F, Σ_FB, ω_B
        """
        mean_f = self.mean[asset_list]  # [n_selected_assets, 1]
        cov_f = self.cov[
            np.ix_(asset_list, asset_list)
        ]  # [n_selected_assets, n_selected_assets]
        bounded_asset = [
            i for i in range(self.mean.shape[0]) if i not in asset_list
        ]  # ← 순서 고정
        cov_fb = self.cov[np.ix_(asset_list, bounded_asset)]
        weighted_bounded = self.weights[-1][bounded_asset]
        return mean_f, cov_f, cov_fb, weighted_bounded

    def __init_algorithm(self) -> Tuple[List[int], np.ndarray]:
        """
        Initialize the optimization algorithm by setting weights to highest returns (maximum return portfolio)
        """
        df = pd.DataFrame(
            data=self.mean.flatten(), index=range(self.mean.shape[0]), columns=["mean"]
        )
        df.sort_values(by="mean", ascending=False, inplace=True)
        cur_weight = self.lower_bounds.copy()
        for i in range(self.mean.shape[0]):
            cur_weight[df.index[i]] = self.upper_bounds[df.index[i]]
            if float(np.sum(cur_weight)) >= 1:
                cur_weight[df.index[i]] += 1 - float(
                    np.sum(cur_weight)
                )  # adjust weight
                return [df.index[i]], cur_weight

    def __compute_lambda(
        self, cov_f_inv, cov_fb, mean_f, weight_b, asset_idx, asset_weight
    ):
        """
        Compute Following Equation
        λ = 1/C * [ (1 - 1'_{n-k}ω_B + 1'_k Σ_F^{-1} Σ_FB ω_B)(Σ_F^{-1}1_k)_i
                    - (1'_k Σ_F^{-1} 1_k)( b_i + (Σ_F^{-1}Σ_FBω_B)_i ) ]
        with
        C = -(1'_k Σ_F^{-1} 1_k)(Σ_F^{-1} μ_F)_i + (1'_k Σ_F^{-1} μ_F)(Σ_F^{-1} 1_k)_i
        b_i = u_i if C>0 else l_i if C<0 (b_i를 (u_i, l_i)로 받은 경우)
        """
        k = mean_f.shape[0]
        one_k = np.ones((k, 1))  # [n_selected_assets, 1]
        one_f_inv_one = (one_k.T @ cov_f_inv @ one_k).item()  # scalar
        f_inv_mean = (cov_f_inv @ mean_f).reshape(-1)  # [n_selected_assets]
        one_f_inv_mean = (one_k.T @ f_inv_mean).item()  # scalar
        f_inv_one = (cov_f_inv @ one_k).reshape(-1)  # [n_selected_assets]
        C = -(one_f_inv_one * f_inv_mean[asset_idx]) + (
            one_f_inv_mean * f_inv_one[asset_idx]
        )  # scalar

        if np.isclose(C, 0.0):
            # If C is close to zero, return None
            return None, None

        if (
            isinstance(asset_weight, (list, tuple, np.ndarray))
            and len(asset_weight) == 2
        ):
            u_i, l_i = asset_weight
            b_i = u_i if C > 0 else l_i
        else:
            b_i = asset_weight

        # Only Free bound Asset
        if weight_b is None or cov_fb is None:
            lam = (f_inv_one[asset_idx] - one_f_inv_one * b_i) / C
            return float(lam), float(b_i)

        # Contain Bounded Asset
        w_b = weight_b.reshape(-1, 1)  # [n_bounded_assets, 1]
        one_b = np.ones_like(w_b)  # [n_bounded_assets, 1]
        one_nk_wb = (one_b.T @ w_b).item()  # scalar
        f_inv_fb_wb = cov_f_inv @ cov_fb @ w_b  # [n_selected_assets, 1]
        one_nk_f_inv_fb_wb = (one_k.T @ f_inv_fb_wb).item()  # scalar

        term_a = (1 - one_nk_wb + one_nk_f_inv_fb_wb) * f_inv_one[asset_idx]
        term_b = one_f_inv_one * (b_i + float(f_inv_fb_wb.reshape(-1)[asset_idx]))
        lam = 1 / C * (term_a - term_b)
        return lam, b_i

    def __compute_w(self, cov_f_inv, cov_fb, mean_f, w_b):
        """
        Implements (paper):
        γ = -λ (1'_k Σ_F^{-1} μ_F)/(1'_k Σ_F^{-1} 1_k)
            + (1 - 1'_{n-k} ω_B + 1'_k Σ_F^{-1} Σ_FB ω_B)/(1'_k Σ_F^{-1} 1_k)

        ω_F = -Σ_F^{-1} Σ_FB ω_B + γ Σ_F^{-1} 1_k + λ Σ_F^{-1} μ_F [n_selected_assets]
        """
        lam = float(self.lambdas[-1])

        k = mean_f.shape[0]
        one_k = np.ones((k, 1))  # [n_selected_assets, 1]
        f_inv_one = (cov_f_inv @ one_k).reshape(-1)  # [n_selected_assets]
        f_inv_mean = (cov_f_inv @ mean_f).reshape(-1)  # [n_selected_assets]
        one_f_inv_one = (one_k.T @ f_inv_one.reshape(-1, 1)).item()  # scalar
        one_f_inv_mean = (one_k.T @ f_inv_mean.reshape(-1, 1)).item()  # scalar

        if w_b is None or cov_fb is None or w_b.size == 0 or cov_fb.size == 0:
            one_nk_wb = 0.0
            f_inv_fb_wb = np.zeros((k, 1))
            one_k_f_inv_fb_wb = 0.0
        else:
            w_b = w_b.reshape(-1, 1)  # [n_bounded_assets, 1]
            one_b = np.ones_like(w_b)
            one_nk_wb = (one_b.T @ w_b).item()  # scalar
            f_inv_fb_wb = cov_f_inv @ cov_fb @ w_b  # [n_selected_assets, 1]
            one_k_f_inv_fb_wb = (one_k.T @ f_inv_fb_wb).item()  # scalar

        # Compute Gamma
        gamma = (
            -lam * (one_f_inv_mean / one_f_inv_one)
            + (1 - one_nk_wb + one_k_f_inv_fb_wb) / one_f_inv_one
        )

        # Compute w_F
        w_f = (
            -f_inv_fb_wb.reshape(-1) + gamma * f_inv_one + lam * f_inv_mean
        )  # [n_selected_assets]
        return gamma, w_f

    def __purge_num_errors(self, tol: float):
        # Purge violations of inequality constraints
        lb = np.asarray(self.lower_bounds).reshape(-1)
        ub = np.asarray(self.upper_bounds).reshape(-1)

        keep = []
        for w in self.weights:
            wv = np.asarray(w).reshape(-1)
            bad = (wv < lb - tol).any() or (wv > ub + tol).any()
            keep.append(not bad)

        self.weights = [w for w, k in zip(self.weights, keep) if k]
        self.lambdas = [l for l, k in zip(self.lambdas, keep) if k]
        self.gammas = [g for g, k in zip(self.gammas, keep) if k]
        self.free_weights = [f for f, k in zip(self.free_weights, keep) if k]

    def __purge_non_convex_weights(self, tol: float = 1e-10):
        mu = np.array([float(w.T @ self.mean) for w in self.weights])
        sig = np.array([float(np.sqrt(w.T @ self.cov @ w)) for w in self.weights])
        idx = np.arange(len(self.weights))

        # σ 기준 정렬
        order = np.argsort(sig)
        mu, sig, idx = mu[order], sig[order], idx[order]

        # (1) 지배점 제거: σ는 커졌는데 μ가 안 오르면 버림
        keep = []
        best_mu = -np.inf
        for k in range(len(idx)):
            if mu[k] > best_mu + tol:
                keep.append(k)
                best_mu = mu[k]
        mu, sig, idx = mu[keep], sig[keep], idx[keep]

        # (2) 오목성(upper hull): 기울기(Δμ/Δσ)가 감소해야 함
        def slope(i, j):
            ds = sig[j] - sig[i]
            return (mu[j] - mu[i]) / ds if ds > tol else np.inf

        hull = []
        for k in range(len(idx)):
            hull.append(k)
            while len(hull) >= 3:
                i, j, l = hull[-3], hull[-2], hull[-1]
                if slope(i, j) <= slope(j, l) + 1e-12:  # 오목성 위배 → 중간점 제거
                    hull.pop(-2)
                else:
                    break

        chosen = order[np.array(keep)[hull]]
        chosen = set(chosen.tolist())
        self.weights = [w for t, w in enumerate(self.weights) if t in chosen]
        self.lambdas = [l for t, l in enumerate(self.lambdas) if t in chosen]
        self.gammas = [g for t, g in enumerate(self.gammas) if t in chosen]
        self.free_weights = [f for t, f in enumerate(self.free_weights) if t in chosen]

    def __run_algorithm(self):
        """
        Run the optimization algorithm
        """
        free_asset_list, cur_weight = self.__init_algorithm()

        self.weights.append(cur_weight.copy())
        self.lambdas.append(None)
        self.gammas.append(None)
        self.free_weights.append(free_asset_list.copy())

        # Remove & Add Assets to compute turning points
        while True:
            max_lambda_in = None
            max_lambda_asset_idx_in = None
            max_lambda_b_i_in = None
            if len(free_asset_list) > 0:
                mean_f, cov_f, cov_fb, weight_b = self.__get_matrices(free_asset_list)
                cov_f_inv = np.linalg.pinv(cov_f)
                for i, asset_idx in enumerate(free_asset_list):
                    lam, b_i = self.__compute_lambda(
                        cov_f_inv,
                        cov_fb,
                        mean_f,
                        weight_b,
                        i,
                        [self.upper_bounds[asset_idx], self.lower_bounds[asset_idx]],
                    )
                    if max_lambda_in is None or (
                        lam is not None and (lam > (max_lambda_in + 1e-12))
                    ):
                        max_lambda_in = lam
                        max_lambda_asset_idx_in = asset_idx
                        max_lambda_b_i_in = b_i

            max_lambda_out = None
            max_lambda_asset_idx_out = None
            max_lambda_b_i_out = None
            if len(free_asset_list) < self.mean.shape[0]:
                remaining_asset_list = [
                    i for i in range(self.mean.shape[0]) if i not in free_asset_list
                ]
                for asset_idx in remaining_asset_list:
                    mean_f, cov_f, cov_fb, weight_b = self.__get_matrices(
                        free_asset_list + [asset_idx]
                    )
                    cov_f_inv = np.linalg.pinv(cov_f)
                    lam, b_i = self.__compute_lambda(
                        cov_f_inv,
                        cov_fb,
                        mean_f,
                        weight_b,
                        mean_f.shape[0] - 1,
                        self.weights[-1][asset_idx],
                    )
                    if max_lambda_out is None or (
                        lam is not None
                        and lam > (max_lambda_out + 1e-12)
                        and lam < self.lambdas[-1]
                    ):
                        max_lambda_out = lam
                        max_lambda_asset_idx_out = asset_idx
                        max_lambda_b_i_out = b_i

            no_in = (max_lambda_in is None) or (max_lambda_in < 0)
            no_out = (max_lambda_out is None) or (max_lambda_out < 0)
            if no_in and no_out:
                # No more turning points, then make min variance portfolio for leftmost turning points
                self.lambdas.append(0)
                mean_f, cov_f, cov_fb, weight_b = self.__get_matrices(free_asset_list)
                cov_f_inv = np.linalg.pinv(cov_f)
                mean_f = np.zeros(mean_f.shape)
            else:
                if (not no_in) and (no_out or max_lambda_in >= max_lambda_out):
                    self.lambdas.append(max_lambda_in)
                    free_asset_list.remove(max_lambda_asset_idx_in)
                    cur_weight[max_lambda_asset_idx_in] = max_lambda_b_i_in
                else:
                    self.lambdas.append(max_lambda_out)
                    free_asset_list.append(max_lambda_asset_idx_out)

                mean_f, cov_f, cov_fb, weight_b = self.__get_matrices(free_asset_list)
                cov_f_inv = np.linalg.pinv(cov_f)

            # compute solution
            gamma, w_f = self.__compute_w(cov_f_inv, cov_fb, mean_f, weight_b)
            for i, free_asset in enumerate(free_asset_list):
                cur_weight[free_asset] = w_f[i]
            self.weights.append(np.copy(cur_weight))
            self.gammas.append(gamma)
            self.free_weights.append(np.copy(free_asset_list))
            if self.lambdas[-1] is None or np.isclose(self.lambdas[-1], 0.0):
                break

        # Purge Turning Points
        self.__purge_num_errors(1e-9)
        self.__purge_non_convex_weights()

    def get_min_var_port(self):
        """
        Get Minimum Variance Portfolio
        """
        var_list = []
        for w in self.weights:
            var_list.append(np.dot(w.T, np.dot(self.cov, w)).item())
        min_var_idx = np.argmin(var_list)
        return self.weights[min_var_idx], np.sqrt(var_list[min_var_idx])

    def get_max_sharpe_port(self):
        """
        Get Maximum Sharpe Portfolio
        """

        def compute_sharpe_ratio(a, w0, w1):
            w = a * w0 + (1 - a) * w1
            mean = (w.T @ self.mean).item()
            std = np.sqrt(w.T @ self.cov @ w).item()
            return mean / std if std > 0 else -np.inf

        w_sr, sr = [], []
        for i in range(1, len(self.weights)):
            w0 = np.copy(self.weights[i - 1])
            w1 = np.copy(self.weights[i])
            a = minimize_scalar(
                lambda x, w0=w0, w1=w1: -compute_sharpe_ratio(x, w0, w1),
                bounds=(0, 1),
                method="bounded",
                options={"xatol": 1e-9},
            ).x
            w_sr.append(a * w0 + (1 - a) * w1)
            sr.append(compute_sharpe_ratio(a, w0, w1))
        max_sr_idx = np.argmax(sr)
        return w_sr[max_sr_idx], sr[max_sr_idx]

    def get_efficient_frontiers(self, points: int):
        mu, sigma, weights = [], [], []
        n_tp = len(self.weights)
        if n_tp < 2:
            return np.array([]), np.array([]), np.array([])

        n_seg = n_tp - 1
        per_seg = max(2, int(np.ceil(points / n_seg)))  # 최소 2

        for i in range(n_seg):
            w0 = np.copy(self.weights[i])
            w1 = np.copy(self.weights[i + 1])

            # 마지막 구간만 endpoint 포함
            alphas = np.linspace(1.0, 0.0, per_seg, endpoint=(i == n_seg - 1))
            if i < n_seg - 1:
                alphas = alphas[:-1]  # 중복점 제거

            for a in alphas:
                w = a * w0 + (1 - a) * w1
                mu.append(float(w.T @ self.mean))
                sigma.append(float(np.sqrt(w.T @ self.cov @ w)))
                weights.append(w.reshape(-1))  # 1D로 통일

        mu = np.array(mu)
        sigma = np.array(sigma)
        weights = np.array(weights)

        # 완전 중복 점 제거(선택)
        if len(mu) > 1:
            keep = np.ones_like(mu, dtype=bool)
            keep[1:] = (np.abs(mu[1:] - mu[:-1]) > 1e-12) | (
                np.abs(sigma[1:] - sigma[:-1]) > 1e-12
            )
            mu, sigma, weights = mu[keep], sigma[keep], weights[keep]

        return mu, sigma, weights
