import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


# ------------------------ å·¥å…·å‡½æ•° ------------------------
def differ5_torch(y, delta):
    """äº”ç‚¹å·®åˆ†è®¡ç®—å¯¼æ•°"""
    L = y.shape[-1]
    ybar = torch.zeros_like(y)
    if L >= 3:
        ybar[..., 1:-1] = (y[..., 2:] - y[..., :-2]) / (2 * delta)
        ybar[..., 0] = (y[..., 1] - y[..., 0]) / delta
        ybar[..., -1] = (y[..., -1] - y[..., -2]) / delta
    return ybar


def cumtrapz_torch(y, dx):
    """ç´¯ç§¯æ¢¯å½¢ç§¯åˆ†ï¼Œä¿æŒé•¿åº¦ä¸€è‡´"""
    cumsum = torch.zeros_like(y)
    if y.shape[-1] > 1:
        cumsum[..., 1:] = torch.cumsum((y[..., :-1] + y[..., 1:]) * 0.5 * dx, dim=-1)
    return cumsum


def projec5(vec, var):
    """æŠ•å½±æ“ä½œï¼ŒæŽ§åˆ¶å™ªå£°"""
    if isinstance(var, (int, float)) and var == 0:
        return torch.zeros_like(vec)

    # æ”¯æŒæ‰¹é‡å¤„ç†
    if vec.dim() == 1:
        M = vec.numel()
        e = torch.sqrt(torch.tensor(M * var, dtype=vec.dtype, device=vec.device))
        n = torch.norm(vec)
        if n > e:
            return vec * (e / n)
        else:
            return vec
    else:
        # æ‰¹é‡å¤„ç†
        M = vec.shape[-1]
        e = torch.sqrt(torch.tensor(M * var, dtype=vec.dtype, device=vec.device))
        n = torch.norm(vec, dim=-1, keepdim=True)
        scale = torch.minimum(torch.ones_like(n), e / (n + 1e-12))
        return vec * scale


def build_second_diff_matrix(N, device, dtype=torch.float32):
    """æž„å»ºäºŒé˜¶å·®åˆ†çŸ©é˜µ"""
    e = torch.ones(N, dtype=dtype, device=device)
    e2 = -2.0 * torch.ones(N, dtype=dtype, device=device)
    e2[0] = -1.0
    e2[-1] = -1.0
    oper = torch.diag(e2) + torch.diag(e[:-1], -1) + torch.diag(e[:-1], 1)
    opedoub = oper.T @ oper
    return opedoub


# ------------------------ è¶…å‚æ•°å­¦ä¹ ç½‘ç»œ ------------------------
class HyperparameterRefinement(nn.Module):
    """åŸºäºŽåˆå§‹é¢‘çŽ‡å’Œå½“å‰è¶…å‚æ•°å­¦ä¹ æ®‹å·®é¡¹çš„ç½‘ç»œ"""

    def __init__(self, hidden_dim=64):
        super().__init__()

        # é¢‘çŽ‡ç‰¹å¾æå–å™¨
        self.freq_encoder = nn.Sequential(
            nn.Linear(1, 32),  # è¾“å…¥å¹³å‡é¢‘çŽ‡
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # è¶…å‚æ•°æ®‹å·®é¢„æµ‹å™¨
        self.param_refiner = nn.Sequential(
            nn.Linear(16 + 2, hidden_dim),  # 16(é¢‘çŽ‡ç‰¹å¾) + 2(å½“å‰alpha,beta)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # è¾“å‡ºalphaå’Œbetaçš„æ®‹å·®
            nn.Tanh()  # é™åˆ¶æ®‹å·®èŒƒå›´
        )

        # è¿­ä»£è‡ªé€‚åº”æƒé‡
        self.iteration_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, init_freqs, current_alpha, current_beta, iteration=1):
        """
        æ ¹æ®åˆå§‹é¢‘çŽ‡å’Œå½“å‰è¶…å‚æ•°é¢„æµ‹æ®‹å·®

        Args:
            init_freqs: (batch_size, K) åˆå§‹å¹³å‡é¢‘çŽ‡
            current_alpha, current_beta: å½“å‰è¶…å‚æ•°å€¼
            iteration: å½“å‰è¿­ä»£æ¬¡æ•°
        """
        batch_size, K = init_freqs.shape

        # æå–é¢‘çŽ‡ç‰¹å¾ - ä½¿ç”¨å¹³å‡é¢‘çŽ‡ä½œä¸ºä»£è¡¨
        avg_freqs = torch.mean(init_freqs, dim=1, keepdim=True)  # (batch_size, 1)
        freq_features = self.freq_encoder(avg_freqs)  # (batch_size, 16)

        # å½“å‰è¶…å‚æ•°
        current_params = torch.stack([
            current_alpha.expand(batch_size),
            current_beta.expand(batch_size)
        ], dim=1)  # (batch_size, 2)

        # æ‹¼æŽ¥ç‰¹å¾
        combined_features = torch.cat([freq_features, current_params], dim=1)

        # é¢„æµ‹æ®‹å·®
        residuals = self.param_refiner(combined_features)  # (batch_size, 2)

        # åº”ç”¨è¿­ä»£è‡ªé€‚åº”æƒé‡
        iteration_factor = torch.sigmoid(self.iteration_weight * iteration)
        residuals = residuals * iteration_factor * 0.1  # æŽ§åˆ¶æ®‹å·®å¹…åº¦

        # è®¡ç®—æ–°çš„è¶…å‚æ•°
        alpha_residual = residuals[:, 0] * current_alpha
        beta_residual = residuals[:, 1] * current_beta

        new_alpha = torch.clamp(current_alpha + alpha_residual.mean(), min=1e-6, max=1e-2)
        new_beta = torch.clamp(current_beta + beta_residual.mean(), min=1e-6, max=1e-1)

        return new_alpha, new_beta


# ------------------------ å¯å¾®åˆ†çº¿æ€§æ±‚è§£å™¨ ------------------------
class DifferentiableLinearSolver(nn.Module):
    """å¯å¾®åˆ†çš„çº¿æ€§ç³»ç»Ÿæ±‚è§£å™¨ï¼Œé¿å…torch.linalg.solveçš„æ¢¯åº¦é—®é¢˜"""

    def __init__(self, max_iter=50, tol=1e-6):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol

    def forward(self, A, b, x_init=None):
        """
        ä½¿ç”¨å…±è½­æ¢¯åº¦æ³•æ±‚è§£ Ax = b

        Args:
            A: (N, N) æ­£å®šå¯¹ç§°çŸ©é˜µ
            b: (N,) å³ç«¯å‘é‡
            x_init: (N,) åˆå§‹è§£ï¼ˆå¯é€‰ï¼‰
        """
        N = A.shape[0]
        device = A.device
        dtype = A.dtype

        # æ·»åŠ æ­£åˆ™åŒ–ä¿è¯æ­£å®šæ€§
        reg = 1e-6 * torch.eye(N, device=device, dtype=dtype)
        A_reg = A + reg

        # åˆå§‹åŒ–
        if x_init is None:
            x = torch.zeros_like(b)
        else:
            x = x_init.clone()

        r = b - torch.mv(A_reg, x)
        p = r.clone()
        rsold = torch.dot(r, r)

        # å…±è½­æ¢¯åº¦è¿­ä»£
        for i in range(self.max_iter):
            Ap = torch.mv(A_reg, p)
            alpha = rsold / (torch.dot(p, Ap) + 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)

            if torch.sqrt(rsnew) < self.tol:
                break

            beta = rsnew / (rsold + 1e-12)
            p = r + beta * p
            rsold = rsnew

        return x


# ------------------------ VNCMDç½‘ç»œå±‚ ------------------------
class VNCMDLayer(nn.Module):
    """å•ä¸ªVNCMDè¿­ä»£å±‚"""

    def __init__(self, use_hyperparameter_learning=True):
        super().__init__()
        self.use_hyperparameter_learning = use_hyperparameter_learning

        if use_hyperparameter_learning:
            self.hyperparam_refiner = HyperparameterRefinement()

        self.linear_solver = DifferentiableLinearSolver(max_iter=30)

    def forward(self, s, eIF, xm, ym, sum_x, sum_y, lamuda,
                alpha, beta, var, fs, init_freqs=None, iteration=1, mode_mask=None):
        """
        å•æ­¥VNCMDè¿­ä»£

        Args:
            s: (batch_size, N) è¾“å…¥ä¿¡å·
            eIF: (batch_size, K, N) çž¬æ—¶é¢‘çŽ‡
            xm, ym: (batch_size, K, N) æ¨¡æ€åˆ†é‡
            sum_x, sum_y: (batch_size, N) ç´¯ç§¯é¡¹
            lamuda: (batch_size, N) æ‹‰æ ¼æœ—æ—¥ä¹˜æ•°
            alpha, beta: è¶…å‚æ•°
            var: å™ªå£°æ–¹å·®
            fs: é‡‡æ ·é¢‘çŽ‡
            init_freqs: (batch_size, K) åˆå§‹å¹³å‡é¢‘çŽ‡ï¼ˆç”¨äºŽè¶…å‚æ•°å­¦ä¹ ï¼‰
            iteration: å½“å‰è¿­ä»£æ¬¡æ•°
            mode_mask: (batch_size, K) æ¨¡æ€æœ‰æ•ˆæ€§æŽ©ç 
        """
        device = s.device
        batch_size, N = s.shape
        K = eIF.shape[1]

        # è¶…å‚æ•°å­¦ä¹
        current_alpha = alpha
        current_beta = beta
        if self.use_hyperparameter_learning and init_freqs is not None:
            current_alpha, current_beta = self.hyperparam_refiner(
                init_freqs, alpha, beta, iteration
            )

        # åŠ¨æ€betaè°ƒæ•´
        betathr = torch.minimum(
            10 ** (iteration / 36.0 - 10.0) * torch.ones_like(current_beta),
            current_beta
        )

        # æž„å»ºäºŒé˜¶å·®åˆ†çŸ©é˜µ
        opedoub = build_second_diff_matrix(N, device, s.dtype)

        # åˆ›å»ºè¾“å‡ºå¼ é‡
        new_eIF = eIF.clone()
        new_xm = xm.clone()
        new_ym = ym.clone()
        new_sum_x = sum_x.clone()
        new_sum_y = sum_y.clone()
        new_lamuda = lamuda.clone()

        # æ‰¹é‡å¤„ç†
        for b in range(batch_size):
            # æŠ•å½±æ“ä½œ
            u_b = projec5(s[b] - sum_x[b] - sum_y[b] - lamuda[b] / current_alpha, var)

            # é‡æ–°è®¡ç®—ç´¯ç§¯é¡¹
            batch_sum_x = torch.zeros(N, device=device, dtype=s.dtype)
            batch_sum_y = torch.zeros(N, device=device, dtype=s.dtype)

            # æ›´æ–°æ¯ä¸ªæ¨¡æ€
            for k in range(K):
                # æ£€æŸ¥æ¨¡æ€æœ‰æ•ˆæ€§
                if mode_mask is not None and mode_mask[b, k] == 0:
                    continue

                # ç§»é™¤å½“å‰æ¨¡æ€çš„è´¡çŒ®
                temp_sum_x = new_sum_x[b] - new_xm[b, k, :] * torch.cos(
                    2 * math.pi * cumtrapz_torch(new_eIF[b, k, :], torch.tensor(1 / fs))
                )
                temp_sum_y = new_sum_y[b] - new_ym[b, k, :] * torch.sin(
                    2 * math.pi * cumtrapz_torch(new_eIF[b, k, :], torch.tensor(1 / fs))
                )

                # è®¡ç®—å½“å‰ç›¸ä½
                phase = 2 * math.pi * cumtrapz_torch(new_eIF[b, k, :], torch.tensor(1 / fs))
                cosm_k = torch.cos(phase)
                sinm_k = torch.sin(phase)

                # æž„å»ºçº¿æ€§ç³»ç»Ÿ - xæ›´æ–°
                A_x = (2.0 / current_alpha) * opedoub + torch.diag(cosm_k ** 2)
                rhs_x = cosm_k * (s[b] - temp_sum_x - temp_sum_y - u_b - new_lamuda[b] / current_alpha)
                new_xm[b, k, :] = self.linear_solver(A_x, rhs_x, xm[b, k, :])

                # æž„å»ºçº¿æ€§ç³»ç»Ÿ - yæ›´æ–°
                A_y = (2.0 / current_alpha) * opedoub + torch.diag(sinm_k ** 2)
                rhs_y = sinm_k * (s[b] - temp_sum_x - temp_sum_y - u_b - new_lamuda[b] / current_alpha)
                new_ym[b, k, :] = self.linear_solver(A_y, rhs_y, ym[b, k, :])

                # IFæ›´æ–°
                xbar = differ5_torch(new_xm[b, k, :], 1.0 / fs)
                ybar = differ5_torch(new_ym[b, k, :], 1.0 / fs)
                denom = new_xm[b, k, :] ** 2 + new_ym[b, k, :] ** 2 + 1e-12
                deltaIF = (new_xm[b, k, :] * ybar - new_ym[b, k, :] * xbar) / (denom * 2 * math.pi)

                # å¹³æ»‘IFæ›´æ–°
                S = (2.0 / betathr) * opedoub + torch.eye(N, device=device, dtype=s.dtype)
                deltaIF_smooth = self.linear_solver(S, deltaIF, torch.zeros_like(deltaIF))
                new_eIF[b, k, :] = new_eIF[b, k, :] - 0.5 * deltaIF_smooth

                # æ›´æ–°ç›¸ä½å’Œç´¯ç§¯é¡¹
                new_phase = 2 * math.pi * cumtrapz_torch(new_eIF[b, k, :], torch.tensor(1 / fs))
                new_cosm = torch.cos(new_phase)
                new_sinm = torch.sin(new_phase)

                batch_sum_x += new_xm[b, k, :] * new_cosm
                batch_sum_y += new_ym[b, k, :] * new_sinm

            # æ›´æ–°ç´¯ç§¯é¡¹å’Œæ‹‰æ ¼æœ—æ—¥ä¹˜æ•°
            new_sum_x[b] = batch_sum_x
            new_sum_y[b] = batch_sum_y
            new_lamuda[b] = new_lamuda[b] + current_alpha * (u_b + batch_sum_x + batch_sum_y - s[b])

        return new_eIF, new_xm, new_ym, new_sum_x, new_sum_y, new_lamuda, current_alpha, current_beta


# ------------------------ æ·±åº¦å±•å¼€VNCMDç½‘ç»œ ------------------------
class DeepUnfoldedVNCMD(nn.Module):
    """æ·±åº¦å±•å¼€çš„VNCMDç½‘ç»œ"""

    def __init__(self, max_layers=50, use_hyperparameter_learning=True):
        super().__init__()
        self.max_layers = max_layers
        self.use_hyperparameter_learning = use_hyperparameter_learning

        # å…¨å±€å¯å­¦ä¹ è¶…å‚æ•°
        self.global_alpha = nn.Parameter(torch.tensor(3e-4, dtype=torch.float32))
        self.global_beta = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))

        # ç½‘ç»œå±‚
        self.layers = nn.ModuleList([
            VNCMDLayer(use_hyperparameter_learning)
            for _ in range(max_layers)
        ])

        print(f"åˆ›å»ºæ·±åº¦å±•å¼€VNCMDç½‘ç»œ: {max_layers} å±‚, è¶…å‚æ•°å­¦ä¹ : {use_hyperparameter_learning}")

    def _detect_active_modes(self, eIF):
        """æ£€æµ‹æœ‰æ•ˆæ¨¡æ€"""
        batch_size, K, N = eIF.shape
        mode_mask = torch.zeros((batch_size, K), device=eIF.device, dtype=eIF.dtype)

        for b in range(batch_size):
            for k in range(K):
                if not torch.allclose(eIF[b, k, :], torch.zeros_like(eIF[b, k, :]), atol=1e-6):
                    mode_mask[b, k] = 1.0

        return mode_mask

    def forward(self, s, eIF, fs, var=0.0, num_iterations=None, mode_mask=None,
                tol=1e-7, return_history=False):
        """
        å‰å‘ä¼ æ’­

        Args:
            s: (batch_size, N) è¾“å…¥ä¿¡å·
            eIF: (batch_size, K, N) åˆå§‹çž¬æ—¶é¢‘çŽ‡
            fs: é‡‡æ ·é¢‘çŽ‡
            var: å™ªå£°æ–¹å·®
            num_iterations: è¿­ä»£æ¬¡æ•°
            mode_mask: (batch_size, K) æ¨¡æ€æŽ©ç 
            tol: æ”¶æ•›å®¹å·®
            return_history: æ˜¯å¦è¿”å›žåŽ†å²è®°å½•
        """
        device = s.device

        # å¤„ç†å•ä¿¡å·è¾“å…¥
        squeeze_output = False
        if s.dim() == 1:
            s = s.unsqueeze(0)
            eIF = eIF.unsqueeze(0)
            if mode_mask is not None:
                mode_mask = mode_mask.unsqueeze(0)
            squeeze_output = True

        batch_size, N = s.shape
        K = eIF.shape[1]

        # è‡ªåŠ¨æ£€æµ‹æœ‰æ•ˆæ¨¡æ€
        if mode_mask is None:
            mode_mask = self._detect_active_modes(eIF)

        # è®¡ç®—åˆå§‹å¹³å‡é¢‘çŽ‡ï¼ˆç”¨äºŽè¶…å‚æ•°å­¦ä¹ ï¼‰
        init_freqs = torch.zeros((batch_size, K), device=device, dtype=s.dtype)
        for b in range(batch_size):
            for k in range(K):
                if mode_mask[b, k] > 0:
                    init_freqs[b, k] = torch.mean(eIF[b, k, :])

        # åˆå§‹åŒ–æ¨¡æ€åˆ†é‡
        xm = torch.zeros((batch_size, K, N), device=device, dtype=s.dtype)
        ym = torch.zeros((batch_size, K, N), device=device, dtype=s.dtype)
        sum_x = torch.zeros((batch_size, N), device=device, dtype=s.dtype)
        sum_y = torch.zeros((batch_size, N), device=device, dtype=s.dtype)
        lamuda = torch.zeros((batch_size, N), device=device, dtype=s.dtype)

        # åˆå§‹åŒ–å„æ¨¡æ€
        opedoub = build_second_diff_matrix(N, device, s.dtype)
        solver = DifferentiableLinearSolver()

        for b in range(batch_size):
            batch_sum_x = torch.zeros(N, device=device, dtype=s.dtype)
            batch_sum_y = torch.zeros(N, device=device, dtype=s.dtype)

            for k in range(K):
                if mode_mask[b, k] == 0:
                    continue

                phase = 2 * math.pi * cumtrapz_torch(eIF[b, k, :], torch.tensor(1 / fs))
                cosm = torch.cos(phase)
                sinm = torch.sin(phase)

                # åˆå§‹åŒ–xm, ym
                A_x = (2.0 / self.global_alpha) * opedoub + torch.diag(cosm ** 2)
                A_y = (2.0 / self.global_alpha) * opedoub + torch.diag(sinm ** 2)

                xm[b, k, :] = solver(A_x, cosm * s[b])
                ym[b, k, :] = solver(A_y, sinm * s[b])

                batch_sum_x += xm[b, k, :] * cosm
                batch_sum_y += ym[b, k, :] * sinm

            sum_x[b] = batch_sum_x
            sum_y[b] = batch_sum_y

        # åŽ†å²è®°å½•
        if return_history:
            eIF_history = [eIF.clone()]
            alpha_history = [self.global_alpha.clone()]
            beta_history = [self.global_beta.clone()]

        # è¿­ä»£ä¼˜åŒ–
        max_iter = num_iterations if num_iterations is not None else self.max_layers
        iteration = 0
        current_alpha = self.global_alpha
        current_beta = self.global_beta

        for layer_idx in range(min(max_iter, self.max_layers)):
            old_eIF = eIF.clone()

            # é€šè¿‡å½“å‰å±‚
            eIF, xm, ym, sum_x, sum_y, lamuda, current_alpha, current_beta = self.layers[layer_idx](
                s, eIF, xm, ym, sum_x, sum_y, lamuda,
                current_alpha, current_beta, var, fs, init_freqs, layer_idx + 1, mode_mask
            )

            iteration += 1

            if return_history:
                eIF_history.append(eIF.clone())
                alpha_history.append(current_alpha.clone())
                beta_history.append(current_beta.clone())

            # æ”¶æ•›æ£€æŸ¥
            if num_iterations is None:
                sDif = torch.tensor(0.0, device=device)
                valid_modes = 0

                for b in range(batch_size):
                    for k in range(K):
                        if mode_mask[b, k] > 0:
                            diff_norm = torch.norm(eIF[b, k, :] - old_eIF[b, k, :])
                            old_norm = torch.norm(old_eIF[b, k, :])
                            sDif += (diff_norm / (old_norm + 1e-12)) ** 2
                            valid_modes += 1

                if valid_modes > 0:
                    sDif = torch.sqrt(sDif / valid_modes)
                    if sDif.item() < tol:
                        break

        # è®¡ç®—æœ€ç»ˆç»“æžœ
        IA = torch.sqrt(xm ** 2 + ym ** 2)

        # é‡æž„ä¿¡å·å’Œå„æ¨¡æ€
        reconstructed = torch.zeros_like(s)
        modes = torch.zeros_like(xm)

        for b in range(batch_size):
            for k in range(K):
                if mode_mask[b, k] == 0:
                    continue

                phase = 2 * math.pi * cumtrapz_torch(eIF[b, k, :], torch.tensor(1 / fs))
                cosm = torch.cos(phase)
                sinm = torch.sin(phase)
                mode_signal = xm[b, k, :] * cosm + ym[b, k, :] * sinm
                modes[b, k, :] = mode_signal
                reconstructed[b] += mode_signal

        # æž„å»ºç»“æžœå­—å…¸
        result = {
            'eIF': eIF,
            'IA': IA,
            'reconstructed': reconstructed,
            'modes': modes,
            'iterations': iteration,
            'final_alpha': current_alpha,
            'final_beta': current_beta,
            'mode_mask': mode_mask
        }

        if return_history:
            result.update({
                'eIF_history': eIF_history,
                'alpha_history': alpha_history,
                'beta_history': beta_history
            })

        # å¤„ç†å•ä¿¡å·è¾“å‡º
        if squeeze_output:
            for key in ['eIF', 'IA', 'reconstructed', 'modes']:
                if key in result:
                    result[key] = result[key].squeeze(0)

        return result


# ------------------------ æŸå¤±å‡½æ•° ------------------------
class VNCMDLoss(nn.Module):
    """VNCMDç½‘ç»œæŸå¤±å‡½æ•°"""

    def __init__(self, lambda_recon=1.0, lambda_if=0.5, lambda_smooth=0.1, lambda_param=0.01):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_if = lambda_if
        self.lambda_smooth = lambda_smooth
        self.lambda_param = lambda_param

    def forward(self, result, target_signal, target_if=None):
        """è®¡ç®—æŸå¤±"""
        mode_mask = result['mode_mask']

        # é‡æž„æŸå¤±
        recon_loss = F.mse_loss(result['reconstructed'], target_signal)
        total_loss = self.lambda_recon * recon_loss
        loss_dict = {'recon_loss': recon_loss}

        # IFæŸå¤±
        if target_if is not None:
            if target_if.dim() == 2:
                target_if = target_if.unsqueeze(0)

            valid_mask = mode_mask.unsqueeze(-1).expand_as(target_if)
            if_loss = F.mse_loss(result['eIF'] * valid_mask, target_if * valid_mask)
            total_loss += self.lambda_if * if_loss
            loss_dict['if_loss'] = if_loss

        # IFå¹³æ»‘æŸå¤±
        if self.lambda_smooth > 0:
            smooth_loss_list = []
            if result['eIF'].dim() == 2:
                eIF_batch = result['eIF'].unsqueeze(0)
                mask_batch = mode_mask.unsqueeze(0)
            else:
                eIF_batch = result['eIF']
                mask_batch = mode_mask

            batch_size, K, N = eIF_batch.shape
            for b in range(batch_size):
                for k in range(K):
                    if mask_batch[b, k] > 0:
                        if_diff = torch.diff(eIF_batch[b, k, :])
                        smooth_loss_list.append(torch.mean(if_diff ** 2))

            if smooth_loss_list:
                smooth_loss = torch.mean(torch.stack(smooth_loss_list))
                total_loss += self.lambda_smooth * smooth_loss
                loss_dict['smooth_loss'] = smooth_loss

        # è¶…å‚æ•°æ­£åˆ™åŒ–
        if self.lambda_param > 0:
            param_loss = (result['final_alpha'] - 3e-4) ** 2 + (result['final_beta'] - 1e-3) ** 2
            total_loss += self.lambda_param * param_loss
            loss_dict['param_loss'] = param_loss

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict


# ------------------------ æµ‹è¯•å’Œä½¿ç”¨ç¤ºä¾‹ ------------------------
def test_deep_unfolded_vncmd():
    """æµ‹è¯•æ·±åº¦å±•å¼€VNCMDç½‘ç»œ"""
    print("=== æµ‹è¯•æ·±åº¦å±•å¼€VNCMDç½‘ç»œ ===")

    # åˆ›å»ºæµ‹è¯•æ•°æ®
    fs = 256
    t = torch.arange(0, 1, 1 / fs, dtype=torch.float32)
    N = len(t)

    # ä¸¤ä¸ªçº¿æ€§è°ƒé¢‘æ¨¡æ€
    f1 = 150 - 80 * t  # mode 1 IF
    f2 = 50 + 60 * t  # mode 2 IF

    # ç”Ÿæˆæ¨¡æ€ä¿¡å·
    phase1 = 2 * math.pi * torch.cumsum(f1, dim=0) / fs
    phase2 = 2 * math.pi * torch.cumsum(f2, dim=0) / fs
    s1 = torch.sin(phase1)
    s2 = torch.sin(phase2)
    s = s1 + s2  # åˆæˆä¿¡å·

    # æ·»åŠ å™ªå£°
    noise_level = 0.1
    noise = noise_level * torch.randn_like(s)
    s_noisy = s + noise

    print(f"ä¿¡å·é•¿åº¦: {N}")
    print(f"é‡‡æ ·é¢‘çŽ‡: {fs} Hz")
    print(f"ä¿¡å™ªæ¯”: {-20 * torch.log10(torch.std(noise) / torch.std(s)):.1f} dB")

    # åˆå§‹IFä¼°è®¡
    eIF_init = torch.stack([
        f1 + torch.randn_like(f1) * 5,
        f2 + torch.randn_like(f2) * 5
    ], dim=0)

    # åˆ›å»ºç½‘ç»œ
    net = DeepUnfoldedVNCMD(max_layers=30, use_hyperparameter_learning=True)

    print(f"åˆå§‹è¶…å‚æ•°: alpha={net.global_alpha.item():.6f}, beta={net.global_beta.item():.6f}")

    # æµ‹è¯•ä¸åŒè¿­ä»£æ¬¡æ•°
    test_cases = [
        ("è®­ç»ƒæ¨¡å¼ (8æ¬¡è¿­ä»£)", 8),
        ("æŽ¨ç†æ¨¡å¼ (20æ¬¡è¿­ä»£)", 20),
    ]

    for name, num_iter in test_cases:
        print(f"\n--- {name} ---")

        with torch.no_grad():
            result = net(s_noisy, eIF_init, fs, var=noise_level ** 2,
                         num_iterations=num_iter, return_history=True)

        print(f"å®žé™…è¿­ä»£æ¬¡æ•°: {result['iterations']}")
        print(f"æœ€ç»ˆè¶…å‚æ•°: alpha={result['final_alpha'].item():.6f}, beta={result['final_beta'].item():.6f}")

        # è®¡ç®—è¯¯å·®
        if1_error = torch.mean((result['eIF'][0, :] - f1) ** 2)
        if2_error = torch.mean((result['eIF'][1, :] - f2) ** 2)
        recon_error = torch.mean((result['reconstructed'] - s) ** 2)

        print(f"IF1è¯¯å·®: {if1_error:.6f}")
        print(f"IF2è¯¯å·®: {if2_error:.6f}")
        print(f"é‡æž„è¯¯å·®: {recon_error:.6f}")

    # ç»˜åˆ¶ç»“æžœ
    result = net(s_noisy, eIF_init, fs, var=noise_level ** 2, num_iterations=20)

    plt.figure(figsize=(15, 10))

    # åŽŸå§‹ä¿¡å·å’Œé‡æž„
    plt.subplot(2, 3, 1)
    plt.plot(t, s, 'b-', label='Clean Signal', linewidth=2)
    plt.plot(t, s_noisy, 'k-', label='Noisy Signal', alpha=0.7)
    plt.plot(t, result['reconstructed'].detach().cpu().numpy(), 'r--', label='Reconstructed', linewidth=2)
    plt.title('Signal Reconstruction')
    plt.legend()
    plt.grid(True)

    # IFä¼°è®¡
    plt.subplot(2, 3, 2)
    plt.plot(t, f1, 'b--', label='True IF1', alpha=0.7, linewidth=2)
    plt.plot(t, result['eIF'][0, :].detach().cpu().numpy(), 'b-', label='Estimated IF1', linewidth=2)
    plt.plot(t, f2, 'g--', label='True IF2', alpha=0.7, linewidth=2)
    plt.plot(t, result['eIF'][1, :].detach().cpu().numpy(), 'g-', label='Estimated IF2', linewidth=2)
    plt.title('IF Estimation')
    plt.legend()
    plt.grid(True)

    # æ¨¡æ€åˆ†ç¦»
    plt.subplot(2, 3, 3)
    plt.plot(t, s1, 'b--', label='True Mode 1', alpha=0.7, linewidth=2)
    plt.plot(t, result['modes'][0, :].detach().cpu().numpy(), 'b-', label='Estimated Mode 1', linewidth=2)
    plt.title('Mode 1 Separation')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(t, s2, 'g--', label='True Mode 2', alpha=0.7, linewidth=2)
    plt.plot(t, result['modes'][1, :].detach().cpu().numpy(), 'g-', label='Estimated Mode 2', linewidth=2)
    plt.title('Mode 2 Separation')
    plt.legend()
    plt.grid(True)

    # IAä¼°è®¡
    plt.subplot(2, 3, 5)
    plt.plot(t, result['IA'][0, :].detach().cpu().numpy(), 'b-', label='IA Mode 1', linewidth=2)
    plt.plot(t, result['IA'][1, :].detach().cpu().numpy(), 'g-', label='IA Mode 2', linewidth=2)
    plt.title('Instantaneous Amplitudes')
    plt.legend()
    plt.grid(True)

    # ç½‘ç»œä¿¡æ¯
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f'Iterations: {result["iterations"]}', fontsize=12)
    plt.text(0.1, 0.6, f'Final Alpha: {result["final_alpha"]:.6f}', fontsize=12)
    plt.text(0.1, 0.4, f'Final Beta: {result["final_beta"]:.6f}', fontsize=12)
    plt.text(0.1, 0.2, f'Recon Error: {recon_error:.6f}', fontsize=12)
    plt.title('Network Info')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return net, result


if __name__ == "__main__":
    print("æ·±åº¦å±•å¼€VNCMDç½‘ç»œ - å®žæ•°ä¿¡å·å¤„ç†")
    print("ç‰¹ç‚¹:")
    print("- å¯å­¦ä¹ è¶…å‚æ•°withæ®‹å·®å­¦ä¹ ")
    print("- åŸºäºŽåˆå§‹é¢‘çŽ‡çš„è‡ªé€‚åº”è°ƒæ•´")
    print("- è®­ç»ƒæ—¶å°‘è¿­ä»£ï¼ŒæŽ¨ç†æ—¶å¤šè¿­ä»£")
    print("- å¯å¾®åˆ†çº¿æ€§æ±‚è§£å™¨")

    test_deep_unfolded_vncmd()