import math

import torch
import torch.optim as optim


class SharedRMSprop(optim.RMSprop):
    """RMSprop optimizer with state tensors placed in shared memory.

    This is useful for A3C-style multiprocessing where multiple workers
    update a shared model.
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0.0,
        centered=False,
    ):
        super().__init__(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )

        # Initialize optimizer state tensors
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                state["step"] = torch.zeros(1)
                state["square_avg"] = torch.zeros_like(p.data)

                if momentum > 0:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                if centered:
                    state["grad_avg"] = torch.zeros_like(p.data)

    def share_memory(self):
        """Move all state tensors into shared memory."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                state["step"].share_memory_()
                state["square_avg"].share_memory_()

                if "momentum_buffer" in state:
                    state["momentum_buffer"].share_memory_()
                if "grad_avg" in state:
                    state["grad_avg"].share_memory_()


