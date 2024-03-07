class EarlyStopping:
    def __init__(self, patience, min_delta) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = float('inf')
        self.epoch_count = 0

    def stop_training(self, loss):
        if loss < self.min_loss - self.min_delta:
            self.epoch_count = 0
            self.min_loss = loss
        else:
            self.epoch_count += 1
            if self.epoch_count >= self.patience:
                return True
        return False