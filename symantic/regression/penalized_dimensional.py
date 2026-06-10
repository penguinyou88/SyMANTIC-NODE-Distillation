"""Penalized regression (L1/L2/ElasticNet) for dimensional SyMANTIC.

Same algorithm as penalized.py but adds dimensional filtering:
features are filtered to those matching output_dim before regression.
"""

import sys
import time

import torch
import numpy as np
import pandas as pd

from sklearn.linear_model import lasso_path, enet_path, Ridge

from ..pareto import pareto


class PenalizedRegressor:
    """Dimensional-aware penalized regression using L1/L2/ElasticNet.

    Mirrors the non-dimensional PenalizedRegressor but first filters
    features to those whose dimension matches ``output_dim``, replicating
    the logic from ``l0_greedy_dimensional.Regressor.get_dimensions_list``.

    Parameters
    ----------
    x : torch.Tensor, shape (n_samples, n_features)
        Feature matrix from the feature expansion stage.
    y : torch.Tensor, shape (n_samples,)
        Target vector.
    names : list of str
        Feature names (same length as n_features).
    dimensionality : list
        Dimension expressions for each feature (same length as n_features).
    complexity : torch.Tensor, shape (n_features,)
        Complexity cost for each feature.
    output_dim : sympy expression or None
        Target dimension. Features not matching this are filtered out.
    dimension : int or None
        Maximum number of terms (n_term).
    sis_features : int
        Features to keep per dimension level via SIS screening.
    device : str
        'cpu' or 'cuda'.
    metrics : list
        [rmse_threshold, r2_threshold] for convergence checks.
    disp : bool
        Print progress information.
    quantiles : list or None
        Ignored (kept for interface compatibility).
    regularization : str
        'l1', 'l2', or 'elastic_net'.
    reg_alpha : float or None
        Regularization strength.
    l1_ratio : float
        L1/L2 mixing for elastic_net.
    reg_threshold : float
        For L2: zero out small coefficients.
    n_alphas : int
        Number of alpha values in the regularization path.
    """

    def __init__(self, x, y, names, dimensionality, complexity,
                 output_dim=None, dimension=None, sis_features=10,
                 device='cpu', metrics=(0.06, 0.995), disp=False,
                 quantiles=None, *,
                 regularization='l1', reg_alpha=None, l1_ratio=0.5,
                 reg_threshold=1e-4, n_alphas=100, **kwargs):

        self.device = device
        self.x = x.to(device)
        self.y = y.to(device)
        self.names = names
        self.complexity = complexity
        self.dimensionality = dimensionality
        self.output_dim = output_dim
        self.disp = disp

        # Dimensional filtering — keep only features matching output_dim
        if self.output_dim is not None:
            dim_indices = self._get_dimension_indices()
            self.x = self.x[:, dim_indices]
            name_series = pd.Series(self.names)
            self.names = name_series.iloc[dim_indices].tolist()
            self.complexity = self.complexity[dim_indices]
            if self.disp:
                print(f'{len(dim_indices)} output dimension features found')

        self.complexity = self.complexity.to(device)
        self.dimension = dimension if dimension is not None else 3
        self.sis_features = sis_features
        self.rmse_metric = metrics[0]
        self.r2_metric = metrics[1]

        self.regularization = regularization
        self.reg_alpha = reg_alpha
        self.l1_ratio = l1_ratio
        self.reg_threshold = reg_threshold
        self.n_alphas = n_alphas

        # Standardize features and center target
        self.x_mean = self.x.mean(dim=0)
        self.x_std = self.x.std(dim=0)
        self.y_mean = self.y.mean()
        self.y_centered = self.y - self.y_mean
        self.x_standardized = (self.x - self.x_mean) / self.x_std

        # Replace NaN/Inf from zero-variance features
        self.x_standardized[torch.isnan(self.x_standardized)] = 0.0
        self.x_standardized[torch.isinf(self.x_standardized)] = 0.0

    def _get_dimension_indices(self):
        """Return indices of features whose dimension matches output_dim."""
        result = {}
        for index, value in enumerate(self.dimensionality):
            if value not in result:
                result[value] = []
            result[value].append(index)

        if self.output_dim in result:
            return result[self.output_dim]
        else:
            if self.disp:
                print('No target dimension features found.')
            sys.exit('No features match the target dimension.')

    # ------------------------------------------------------------------
    # SIS screening
    # ------------------------------------------------------------------
    def _sis_screening(self):
        """Sure Independence Screening: top features by |corr(y, x_j)|."""
        scores = torch.abs(
            torch.mm(self.y_centered.unsqueeze(0), self.x_standardized)
        ).flatten()
        scores[torch.isnan(scores)] = 0.0

        k = min(self.sis_features * self.dimension, self.x.shape[1])
        _, top_idx = torch.topk(scores, k=k)
        return top_idx

    # ------------------------------------------------------------------
    # Regularization path solvers
    # ------------------------------------------------------------------
    def _solve_path(self, X, y):
        """Solve regularization path. Returns (alphas, coef_path)."""
        if self.regularization == 'l1':
            alphas, coef_path, _ = lasso_path(
                X, y, n_alphas=self.n_alphas, max_iter=10000
            )
            return alphas, coef_path

        if self.regularization == 'elastic_net':
            alphas, coef_path, _ = enet_path(
                X, y, l1_ratio=self.l1_ratio,
                n_alphas=self.n_alphas, max_iter=10000
            )
            return alphas, coef_path

        # L2 (Ridge)
        alphas = np.logspace(-6, 6, self.n_alphas)
        coef_path = np.zeros((X.shape[1], self.n_alphas))
        for i, alpha in enumerate(alphas):
            ridge = Ridge(alpha=alpha, fit_intercept=False)
            ridge.fit(X, y)
            coefs = ridge.coef_.copy()
            max_abs = np.max(np.abs(coefs))
            if max_abs > 0:
                coefs[np.abs(coefs) < self.reg_threshold * max_abs] = 0.0
            coef_path[:, i] = coefs
        return alphas, coef_path

    # ------------------------------------------------------------------
    # Build Pareto front from the regularization path
    # ------------------------------------------------------------------
    def _path_to_pareto(self, coef_path, screened_idx, X_std, y_cen):
        """Convert coefficient path to Pareto-ready lists."""
        n_alphas = coef_path.shape[1]
        y_ss = float(torch.sum(self.y_centered ** 2))

        x_mean_s = self.x_mean[screened_idx].cpu().numpy()
        x_std_s = self.x_std[screened_idx].cpu().numpy()
        names_arr = np.array(self.names)[screened_idx.cpu().numpy()]
        complexity_s = self.complexity[screened_idx].cpu().numpy()

        seen_patterns = {}

        rmses, r2s, complexities = [], [], []
        all_names, all_coeffs, all_intercepts = [], [], []

        for j in range(n_alphas):
            coefs = coef_path[:, j]
            nz = np.nonzero(coefs)[0]

            if len(nz) == 0:
                continue

            if len(nz) > self.dimension:
                top_k = np.argsort(np.abs(coefs[nz]))[-self.dimension:]
                nz = nz[top_k]

            pattern = frozenset(nz.tolist())
            pred = X_std[:, nz] @ coefs[nz]
            resid = y_cen - pred
            rmse_val = float(np.sqrt(np.mean(resid ** 2)))

            if pattern in seen_patterns:
                prev_idx = seen_patterns[pattern]
                if rmse_val >= rmses[prev_idx]:
                    continue
                rmses[prev_idx] = rmse_val
                r2s[prev_idx] = 1 - float(np.sum(resid ** 2)) / y_ss
                nz_std = x_std_s[nz]
                denorm_coefs = coefs[nz] / nz_std
                intercept = float(self.y_mean) - float(
                    np.dot(x_mean_s[nz] / nz_std, coefs[nz])
                )
                all_coeffs[prev_idx] = denorm_coefs
                all_intercepts[prev_idx] = intercept
                all_names[prev_idx] = names_arr[nz].tolist()
                continue

            seen_patterns[pattern] = len(rmses)

            r2_val = 1 - float(np.sum(resid ** 2)) / y_ss
            comp_val = float(np.sum(complexity_s[nz]))

            nz_std = x_std_s[nz]
            denorm_coefs = coefs[nz] / nz_std
            intercept = float(self.y_mean) - float(
                np.dot(x_mean_s[nz] / nz_std, coefs[nz])
            )

            rmses.append(rmse_val)
            r2s.append(r2_val)
            complexities.append(comp_val)
            all_coeffs.append(denorm_coefs)
            all_intercepts.append(intercept)
            all_names.append(names_arr[nz].tolist())

        return rmses, r2s, complexities, all_names, all_coeffs, all_intercepts

    # ------------------------------------------------------------------
    # Format equation string
    # ------------------------------------------------------------------
    @staticmethod
    def _format_equation(coeffs, names, intercept):
        terms = []
        for c, n in zip(coeffs, names):
            terms.append(f"{float(c):.20f}*{n}")
        eq = '  '.join(
            (' + ' + t if float(c) > 0 else t)
            for c, t in zip(coeffs, terms)
        )
        if intercept >= 0:
            eq += f'+{float(intercept)}'
        else:
            eq += f'{float(intercept)}'
        return eq

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def regressor_fit(self):
        """Fit penalized regression and return the 9-tuple result.

        Returns
        -------
        tuple
            (rmse, equation, r2,
             pareto_rmse, pareto_complexity,
             pareto_names, pareto_intercepts, pareto_coeffs, pareto_r2)
        """
        start = time.time()

        # 1. SIS screening
        screened_idx = self._sis_screening()

        # 2. Convert to numpy for sklearn
        X_std = self.x_standardized[:, screened_idx].cpu().numpy()
        y_cen = self.y_centered.cpu().numpy()

        # 3. Solve regularization path
        alphas, coef_path = self._solve_path(X_std, y_cen)

        # 4. Convert path to Pareto-ready lists
        (rmses, r2s, complexities,
         all_names, all_coeffs, all_intercepts) = self._path_to_pareto(
            coef_path, screened_idx, X_std, y_cen
        )

        # 5. Add intercept-only baseline
        baseline_rmse = float(torch.sqrt(torch.mean(self.y_centered ** 2)))
        rmses.insert(0, baseline_rmse)
        r2s.insert(0, 0.0)
        complexities.insert(0, 0.0)
        all_names.insert(0, str(float(self.y_mean)))
        all_coeffs.insert(0, np.array([]))
        all_intercepts.insert(0, float(self.y_mean))

        if len(rmses) < 2:
            rmse_t = torch.tensor(rmses)
            comp_t = torch.tensor(complexities)
            r2_t = torch.tensor(r2s)
            equation = str(float(self.y_mean))
            coeff_tensor = torch.empty(1, 1).fill_(float('nan'))
            intercept_tensor = torch.tensor(all_intercepts)
            return (
                float(baseline_rmse), equation, 0.0,
                rmse_t, comp_t, all_names,
                intercept_tensor, coeff_tensor, r2_t,
            )

        # 6. Run Pareto selection
        rmse_t = torch.tensor(rmses, dtype=torch.float32)
        comp_t = torch.tensor(complexities, dtype=torch.float32)
        r2_t = torch.tensor(r2s, dtype=torch.float32)

        pareto_idx = pareto(rmse_t, comp_t).pareto_front()

        pareto_rmse = rmse_t[pareto_idx]
        pareto_complexity = comp_t[pareto_idx]
        pareto_r2 = r2_t[pareto_idx]
        pareto_names = [all_names[i] for i in pareto_idx]
        pareto_intercepts_list = [all_intercepts[i] for i in pareto_idx]
        pareto_coeffs_list = [all_coeffs[i] for i in pareto_idx]

        # 7. Build coefficient tensor
        max_terms = max(
            (len(c) for c in pareto_coeffs_list if len(c) > 0), default=1
        )
        coeff_tensor = torch.full(
            (len(pareto_idx), max_terms), float('nan')
        )
        for i, c in enumerate(pareto_coeffs_list):
            if len(c) > 0:
                coeff_tensor[i, :len(c)] = torch.tensor(c, dtype=torch.float32)

        intercept_tensor = torch.tensor(
            pareto_intercepts_list, dtype=torch.float32
        )

        # 8. Select best model
        non_baseline = pareto_complexity > 0
        if non_baseline.any():
            best_idx = torch.argmin(pareto_rmse[non_baseline])
            non_baseline_indices = torch.nonzero(non_baseline).flatten()
            best_pareto_idx = int(non_baseline_indices[best_idx])
        else:
            best_pareto_idx = 0

        best_rmse = float(pareto_rmse[best_pareto_idx])
        best_r2 = float(pareto_r2[best_pareto_idx])
        best_names = pareto_names[best_pareto_idx]
        best_intercept = pareto_intercepts_list[best_pareto_idx]
        best_coeffs = pareto_coeffs_list[best_pareto_idx]

        if len(best_coeffs) > 0:
            equation = self._format_equation(
                best_coeffs, best_names, best_intercept
            )
        else:
            equation = str(float(self.y_mean))

        if self.disp:
            elapsed = time.time() - start
            print(f'Dimensional penalized regression ({self.regularization}) '
                  f'completed in {elapsed:.2f}s — {len(rmses)} solutions, '
                  f'{len(pareto_idx)} Pareto-optimal')
            print(f'Best: RMSE={best_rmse:.6f}, R2={best_r2:.6f}, '
                  f'terms={len(best_coeffs)}')

        return (
            best_rmse, equation, best_r2,
            pareto_rmse, pareto_complexity,
            pareto_names, intercept_tensor, coeff_tensor, pareto_r2,
        )
