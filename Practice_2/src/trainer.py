import torch
import os
from tqdm import tqdm
from .tensorboard_utils import TensorBoardLogger

def get_device():
    """Tự động dùng CPU hoặc GPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Lưu trữ checkpoint tốt nhất."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    torch.save(checkpoint, path)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Vòng lặp huấn luyện cho một epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Bắt buộc theo đúng tiến trình chuẩn
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    """Vòng lặp validation cho một epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Bắt buộc sử dụng torch.no_grad()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, log_dir, checkpoint_path):
    """Engine huấn luyện chính, kết hợp TensorBoard và trả về history."""
    device = get_device()
    model = model.to(device)
    tb_logger = TensorBoardLogger(log_dir)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Ghi nhận batch ảnh và graph (nếu có thể) ở epoch 0
    try:
        images, _ = next(iter(train_loader))
        tb_logger.log_image_batch(images, step=0)
        tb_logger.log_model_graph(model, images.to(device))
    except Exception as e:
        print(f"Warning: Không thể vẽ model graph: {e}")

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        # Cập nhật history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Ghi log TensorBoard
        tb_logger.log_metrics(train_loss, train_acc, val_loss, val_acc, epoch)
        
        # Giả sử optimizer chỉ có 1 param group
        current_lr = optimizer.param_groups[0]['lr']
        tb_logger.log_learning_rate(current_lr, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Lưu best model
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved ({best_val_acc:.2f}% --> {val_acc:.2f}%). Saving model...")
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            
    tb_logger.close()
    return history