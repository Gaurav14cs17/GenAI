"""
Combined SoFlow Loss.

Combines the Flow Matching loss and Solution Consistency loss
as described in the SoFlow paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union


class SoFlowLoss(nn.Module):
    """
    Combined loss for training SoFlow models.
    
    The total loss is:
        L = λ_FM * L_FM + λ_cons * L_cons
    
    where:
        - L_FM: Flow Matching loss (denoising objective)
        - L_cons: Solution Consistency loss for one-step generation
    
    Args:
        lambda_fm: Weight for Flow Matching loss.
        lambda_cons: Weight for Consistency loss.
    """
    
    def __init__(
        self,
        lambda_fm: float = 1.0,
        lambda_cons: float = 1.0,
        sigma_min: float = 0.0,
        use_adaptive_consistency: bool = True,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_cons = lambda_cons

    def forward(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        y: torch.Tensor,
        step: int = 0,
        total_steps: int = 1,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the combined SoFlow loss.
        
        Args:
            model: The SoFlow model.
            x_0: Data samples, shape (B, C, H, W).
            x_1: Noise samples, shape (B, C, H, W).
            y: Class labels, shape (B,).
            step: Current training step.
            total_steps: Total training steps.
            return_dict: Whether to return individual losses.
            
        Returns:
            Total loss (or dict with individual losses if return_dict=True).
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # === Loss 1: Denoising (main objective) ===
        # Sample t uniformly - predict x_0 from x_t
        t1 = torch.rand(B, device=device) * 0.95 + 0.05
        t1_exp = t1.view(-1, 1, 1, 1)
        x_t1 = (1 - t1_exp) * x_0 + t1_exp * x_1
        
        s = torch.zeros(B, device=device)
        x_0_pred = model.forward(x_t1, t1, s, y)
        loss_fm = F.mse_loss(x_0_pred, x_0)
        
        # #region agent log - H1,H2: Interpolation and prediction ranges
        if step % 500 == 0:
            import json, time
            log_entry = {"location": "combined.py:forward", "message": "interp_pred_ranges", "hypothesisId": "H1,H2", "timestamp": int(time.time()*1000), "sessionId": "debug-session", "data": {"step": step, "t1_mean": float(t1.mean()), "x_t1_min": float(x_t1.min()), "x_t1_max": float(x_t1.max()), "x0_pred_min": float(x_0_pred.min().detach()), "x0_pred_max": float(x_0_pred.max().detach()), "loss_fm": float(loss_fm.detach())}}
            with open("/home/ggoswami/Project/SoFlow/.cursor/debug.log", "a") as f: f.write(json.dumps(log_entry) + "\n")
        # #endregion
        
        # === Loss 2: Consistency (self-consistency) ===
        # Two different noise levels should predict same x_0
        t2 = torch.rand(B, device=device) * 0.5 + 0.05  # Lower noise
        t2_exp = t2.view(-1, 1, 1, 1)
        x_t2 = (1 - t2_exp) * x_0 + t2_exp * x_1
        
        with torch.no_grad():
            x_0_target = model.forward(x_t2, t2, s, y)
        
        # Both predictions should match x_0, so they should match each other
        loss_cons = F.mse_loss(x_0_pred, x_0_target)
        
        # #region agent log - H3,H5: Consistency loss components
        if step % 500 == 0:
            import json, time
            log_entry = {"location": "combined.py:forward", "message": "consistency_loss", "hypothesisId": "H3,H5", "timestamp": int(time.time()*1000), "sessionId": "debug-session", "data": {"step": step, "t2_mean": float(t2.mean()), "loss_cons": float(loss_cons.detach()), "x0_target_min": float(x_0_target.min()), "x0_target_max": float(x_0_target.max()), "pred_vs_gt_mse": float(F.mse_loss(x_0_pred.detach(), x_0).item()), "pred_vs_target_mse": float(loss_cons.detach())}}
            with open("/home/ggoswami/Project/SoFlow/.cursor/debug.log", "a") as f: f.write(json.dumps(log_entry) + "\n")
        # #endregion
        
        # Combined loss - FM is main, consistency is regularization
        total_loss = self.lambda_fm * loss_fm + self.lambda_cons * 0.1 * loss_cons
        
        if return_dict:
            return {
                "loss": total_loss,
                "loss_fm": loss_fm,
                "loss_cons": loss_cons,
            }
        
        return total_loss


class SoFlowLossV2(nn.Module):
    """
    Alternative SoFlow loss - direct trajectory prediction.
    
    Instead of velocity matching, directly predict points on the trajectory.
    """
    
    def __init__(
        self,
        lambda_fm: float = 1.0,
        lambda_cons: float = 1.0,
    ):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_cons = lambda_cons

    def forward(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        y: torch.Tensor,
        step: int = 0,
        total_steps: int = 1,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the unified SoFlow loss.
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample t uniformly from [0.1, 1] to avoid edge cases
        t = torch.rand(B, device=device) * 0.9 + 0.1
        
        # Compute x_t on the trajectory
        t_exp = t.view(-1, 1, 1, 1)
        x_t = (1 - t_exp) * x_0 + t_exp * x_1
        
        # === Flow Matching Loss ===
        # Predict x_0 from x_t (denoising objective)
        s = torch.zeros(B, device=device)
        x_0_pred = model.forward(x_t, t, s, y)
        loss_fm = F.mse_loss(x_0_pred, x_0)
        
        # === Consistency Loss ===
        # Self-consistency: predictions from different points should agree
        r = max(0.1, 1.0 - step / max(1, total_steps))
        l = r * t  # Intermediate time between 0 and t
        l_exp = l.view(-1, 1, 1, 1)
        x_l = (1 - l_exp) * x_0 + l_exp * x_1
        
        with torch.no_grad():
            target = model.forward(x_l, l, s, y)
        
        loss_cons = F.mse_loss(x_0_pred, target)
        
        # Combined
        total_loss = self.lambda_fm * loss_fm + self.lambda_cons * loss_cons
        
        if return_dict:
            return {
                "loss": total_loss,
                "loss_fm": loss_fm,
                "loss_cons": loss_cons,
            }
        
        return total_loss

