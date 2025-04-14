import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def train_model(model, train_dataloader, val_dataloader, num_epochs=10, 
               learning_rate=0.01, early_stopping_patience=3, log_interval=10):
    """Train the model with early stopping and logging"""
    print("Training model...")
    
    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
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
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))
    
    # Return training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    
    return model, history