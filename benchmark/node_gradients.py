from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint


torch.set_default_dtype(torch.float32)


class NodeRhs(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, _t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)



def _interval_inputs(time: torch.Tensor, inputs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # Piecewise-constant exogenous inputs on [t_k, t_{k+1}).
    idx = torch.searchsorted(time, t, right=True) - 1
    idx = idx.clamp(0, len(time) - 1)
    return inputs[idx]



def train_node_and_estimate_gradients(
    time_np,
    states_np,
    inputs_np,
    model: NodeRhs | None = None,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 300,
    method: str = "rk4",
    device: str | None = None,
    verbose: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    state_weights: list | tuple | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    time = torch.as_tensor(time_np, dtype=torch.float32, device=device)
    states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    inputs = torch.as_tensor(inputs_np, dtype=torch.float32, device=device)

    # pass a previously trained model to warm-start, or create a new one if not provided
    rhs = model or NodeRhs(input_dim=2 + inputs.shape[1] + 1, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(rhs.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Validate and normalize state weights
    if state_weights is not None:
        assert len(state_weights) == states.shape[1], (
            f"state_weights must have length {states.shape[1]} (number of states)"
        )
        weights_sum = sum(state_weights)
        normalized_weights = [w / weights_sum for w in state_weights]
    else:
        normalized_weights = None

    def dynamics(t, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        u = _interval_inputs(time, inputs, t)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        t_col = torch.full((x.shape[0], 1), t, dtype=x.dtype, device=x.device)
        net_in = torch.cat([x, u, t_col], dim=1)
        return rhs(t, net_in)

    x0 = states[0]
    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in rhs.state_dict().items()}
    wait = 0
    stopped_early = False
    epochs_trained = 0

    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = odeint(dynamics, x0, time, method=method)
        
        # Compute weighted loss
        if normalized_weights is not None:
            loss = sum(
                w * torch.mean((pred[:, i] - states[:, i]) ** 2)
                for i, w in enumerate(normalized_weights)
            )
        else:
            loss = loss_fn(pred, states)
        
        loss.backward()
        opt.step()

        current_loss = float(loss.item())
        if current_loss < (best_loss - early_stopping_min_delta):
            best_loss = current_loss
            best_state = {k: v.detach().clone() for k, v in rhs.state_dict().items()}
            wait = 0
        else:
            wait += 1

        epochs_trained = epoch + 1
        
        if verbose and epoch % 10 == 0:
            with torch.no_grad():
                r2_list = []
                for i in range(states.shape[1]):
                    ss_res_i = torch.sum((states[:, i] - pred[:, i]) ** 2)
                    ss_tot_i = torch.sum((states[:, i] - states[:, i].mean()) ** 2)
                    r2_i = float(1.0 - ss_res_i / ss_tot_i) if float(ss_tot_i) > 0.0 else float("nan")
                    r2_list.append(r2_i)
            r2_str = "[" + ", ".join(f"{r2:.4f}" for r2 in r2_list) + "]"
            print(
                f"  NODE epoch {epoch}: loss={current_loss:.6f} "
                f"best={best_loss:.6f} r2/state={r2_str} wait={wait}"
            )

        if early_stopping_patience > 0 and wait >= early_stopping_patience:
            stopped_early = True
            if verbose:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(no loss improvement for {early_stopping_patience} epochs)."
                )
            break

    rhs.load_state_dict(best_state)

    with torch.no_grad():
        pred = odeint(dynamics, x0, time, method=method)
        final_loss = float(loss_fn(pred, states).item())
        grads = torch.stack([dynamics(t, x) for t, x in zip(time, pred)], dim=0)

    return {
        "pred_states": pred.detach().cpu().numpy(),
        "gradients": grads.detach().cpu().numpy(),
        "model": rhs,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "epochs_trained": epochs_trained,
        "stopped_early": stopped_early,
        "state_weights": list(normalized_weights) if normalized_weights is not None else None,
    }
