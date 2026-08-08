import torchvision
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir):
        """Khởi tạo TensorBoard writer."""
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, train_loss, train_acc, val_loss, val_acc, epoch):
        """Ghi nhận loss và accuracy vào TensorBoard."""
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        
    def log_learning_rate(self, lr, epoch):
        """Ghi nhận learning rate."""
        self.writer.add_scalar('Learning_Rate', lr, epoch)

    def log_image_batch(self, images, step=0):
        """Ghi nhận một batch ảnh lên TensorBoard."""
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image('Batch_Images', img_grid, global_step=step)

    def log_model_graph(self, model, images):
        """Vẽ graph của mô hình."""
        self.writer.add_graph(model, images)

    def close(self):
        self.writer.close()