"""Entraînement d'un modèle : seule responsabilité de ce module."""


class Trainer:
    """Entraîne n'importe quel Model avec n'importe quel Optimizer (descente full-batch)."""

    def __init__(self, model, optimizer, epochs=500, verbose=True):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
        self.history = []

    def train(self, X, y):
        log_every = max(1, self.epochs // 5)
        for epoch in range(1, self.epochs + 1):
            loss, grads = self.model.loss_and_grads(X, y)
            self.optimizer.step(self.model.params, grads)
            self.history.append(loss)
            if self.verbose and (epoch == 1 or epoch % log_every == 0):
                print(f"  epoch {epoch:>4}/{self.epochs} - perte : {loss:.4f}")
        return self.model
