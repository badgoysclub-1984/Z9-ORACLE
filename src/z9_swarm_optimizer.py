import torch
import math

PHI = (1 + math.sqrt(5)) / 2
FIB = [1, 2, 3, 5, 8, 13, 21, 34, 55]

class Z9SwarmOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, pop_size=9, inertia=0.72, c1=1.8, c2=1.8,
                 anchor_strength=0.97, harm_gamma=0.0008):
        defaults = dict(lr=lr, pop_size=pop_size, inertia=inertia, c1=c1, c2=c2,
                        anchor_strength=anchor_strength, harm_gamma=harm_gamma)
        super().__init__(params, defaults)
        self.pop_size = pop_size
        self.cosets = torch.tensor([0, 3, 6] * (pop_size//3 + 1))[:pop_size]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            inertia = group['inertia']
            c1, c2 = group['c1'], group['c2']
            anc = group['anchor_strength']
            gamma = group['harm_gamma']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state.setdefault(p, {})
                if 'step' not in state:
                    state['step'] = 0
                    state['position'] = p.data.clone()
                    state['velocity'] = torch.zeros_like(p.data)
                    state['pbest'] = p.data.clone()
                    state['gbest'] = p.data.clone()

                state['step'] += 1
                t = state['step']

                mask = (t % 9 == 0) or any(t % f == 0 for f in FIB)
                alpha = anc if mask else 1.0
                phi_spike = PHI if any(t % f == 0 for f in FIB) else 1.0

                coset_factor = 1.0 + 0.15 * math.cos(2 * math.pi * self.cosets[t % self.pop_size] / 9)

                harm = gamma * sum(math.cos(2 * math.pi * n * (t / 9.0) + 2 * math.pi * n / 9) for n in range(1, 10))

                cognitive = c1 * torch.rand_like(grad) * (state['pbest'] - state['position'])
                social = c2 * torch.rand_like(grad) * (state['gbest'] - state['position'])

                state['velocity'] = (inertia * state['velocity'] +
                                     alpha * coset_factor * (cognitive + social) +
                                     harm * grad)

                state['position'] += state['velocity'] * lr * phi_spike

                if torch.norm(grad) < torch.norm(state['position'] - state['pbest']):
                    state['pbest'] = state['position'].clone()
                if torch.norm(grad) < torch.norm(state['position'] - state['gbest']):
                    state['gbest'] = state['position'].clone()

                p.data.copy_(state['position'])
                p.data -= 1e-5 * (p.data % 9)

        return loss
