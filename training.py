import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np

def train_model(model, train_dataloader, val_dataloader, num_epochs=50,  # Increased from 10 to 50
               learning_rate=0.001,  # Decreased from 0.01 to 0.001
               early_stopping_patience=10,  # Increased from 3 to 10
               log_interval=10):
    """Train the model with improved training regime"""
    print("Training model with improved parameters...")
    
    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Create optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = "best_model.pth"
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (x, y_tuple) in enumerate(train_dataloader):
            # Move data to device
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            
            # Handle y correctly - it's a tuple
            if isinstance(y_tuple, tuple):
                y = y_tuple[0].to(device)  # Extract the first element of the tuple
            else:
                y = y_tuple.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x)
            if isinstance(outputs, tuple):
                predictions = outputs[0]
            else:
                predictions = outputs
                
            # Calculate loss
            loss = model.loss(predictions, y)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track loss
            epoch_train_loss += loss.item()
            train_batches += 1
            
            # Log progress
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_dataloader)} - Loss: {loss.item():.4f}")
        
        # Calculate average training loss
        avg_train_loss = epoch_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x, y_tuple in val_dataloader:
                # Move data to device
                x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                
                # Handle y correctly - it's a tuple
                if isinstance(y_tuple, tuple):
                    y = y_tuple[0].to(device)  # Extract the first element of the tuple
                else:
                    y = y_tuple.to(device)
                
                # Forward pass
                outputs = model(x)
                if isinstance(outputs, tuple):
                    predictions = outputs[0]
                else:
                    predictions = outputs
                
                # Calculate loss
                loss = model.loss(predictions, y)
                
                # Track loss
                epoch_val_loss += loss.item()
                val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = epoch_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    
    # Return training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    
    return model, history