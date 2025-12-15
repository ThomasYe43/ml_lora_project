import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time


def accuracy(model, data_loader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(imgs)
            _, preds = torch.max(output, dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    return correct / total


def compute_loss(model, data_loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(imgs)
            loss = criterion(output, labels)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_model(model,
                train_loader,
                val_loader,
                optimizer_name,
                batch_size=64,
                weight_decay=0.0001,
                learning_rate=0.0003,
                num_epochs=10,
                eval_every=500,
                use_scheduler=False,
                save_best_model=True,
                model_save_path="best_model.pth",
                plot=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"batch_size={batch_size}, lr={learning_rate}, epochs={num_epochs}")
    
    # Track trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    model.to(device)
    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    cost = nn.CrossEntropyLoss()

    # Create optimizer

    trainable_params_list = [p for p in model.parameters() if p.requires_grad] #only train the trainables

    if optimizer_name == "Adam":
        optimizer = optim.Adam(trainable_params_list,
                              lr=learning_rate,
                              weight_decay=weight_decay)
    elif optimizer_name == "RMSProp":
        optimizer = optim.RMSprop(trainable_params_list,
                              lr=learning_rate,
                              weight_decay=weight_decay)

    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(trainable_params_list,
                                lr=learning_rate,
                                weight_decay=weight_decay)

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        print("Scheduler: CosineAnnealingLR")
    
    # Tracking metrics
    iters, lst_train_loss, lst_train_acc, lst_val_acc, lst_val_loss = [], [], [], [], []
    epoch_times = []
    iter_count = 0
    best_val_acc = 0.0
    best_epoch = 0
    training_start_time = time.time()

    try:
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            epoch_start_time = time.time()
            model.train()  # CRITICAL: Set model to training mode
            epoch_loss = 0.0
            num_batches = 0
            
            for imgs, labels in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)  # CRITICAL: Clear gradients
                
                if scaler:
                    # AMP training steps
                    with torch.amp.autocast('cuda'):
                        out = model(imgs)
                        loss = cost(out, labels)
                    scaler.scale(loss).backward()
                    # Gradient clipping for stability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard FP32 training fallback (for CPU)
                    out = model(imgs)
                    loss = cost(out, labels)
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
                    optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                iter_count += 1

                # Periodic evaluation
                if iter_count % eval_every == 0:
                    train_loss_sample = loss.item()
                    train_acc = accuracy(model, train_loader, device, max_batches=50)
                    val_acc = accuracy(model, val_loader, device, max_batches=50)
                    val_loss = compute_loss(model, val_loader, cost, device, max_batches=50)
                    
                    print(f"Iter {iter_count} | Train Loss: {train_loss_sample:.4f} | "
                          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

                    iters.append(iter_count)
                    lst_train_loss.append(train_loss_sample)
                    lst_train_acc.append(train_acc)
                    lst_val_acc.append(val_acc)
                    lst_val_loss.append(val_loss)
                    
                    # Save best model based on validation accuracy
                    if save_best_model and val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = epoch + 1
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_acc': best_val_acc,
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        }, model_save_path)
                        print(f"  ✓ Saved best model (val_acc: {best_val_acc:.4f})")
                    
                    model.train()  # Return to training mode
            
            # End of epoch tracking
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            avg_epoch_loss = epoch_loss / num_batches
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Training Loss: {avg_epoch_loss:.4f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            if scheduler:
                print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Step the scheduler
            if scheduler:
                scheduler.step()
        
        total_training_time = time.time() - training_start_time
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best validation accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
        print(f"Total training time: {total_training_time/60:.2f} minutes")
        print(f"Average time per epoch: {sum(epoch_times)/len(epoch_times):.2f}s")
        if save_best_model:
            print(f"Best model saved to: {model_save_path}")
        print(f"{'='*60}\n")
        
        # Prepare metrics dictionary
        metrics = {
            'iters': iters,
            'train_loss': lst_train_loss,
            'train_acc': lst_train_acc,
            'val_loss': lst_val_loss,
            'val_acc': lst_val_acc,
            'total_time': total_training_time,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'trainable_params': trainable_params,
            'total_params': total_params
        }

    finally:
        # Plot training curves
        if plot and len(iters) > 0:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(iters[:len(lst_train_loss)], lst_train_loss, label='Train Loss', alpha=0.7)
            plt.plot(iters[:len(lst_val_loss)], lst_val_loss, label='Val Loss', alpha=0.7)
            plt.title("Loss over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(iters[:len(lst_train_acc)], lst_train_acc, label='Train Acc', alpha=0.7)
            plt.plot(iters[:len(lst_val_acc)], lst_val_acc, label='Val Acc', alpha=0.7)
            plt.title("Accuracy over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    return metrics