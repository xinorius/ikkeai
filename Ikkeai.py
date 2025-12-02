import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import math
import os

# Configuration
class Config:
    # Model parameters (small model optimized for 3070)
    vocab_size = 5000  # Will be set based on data
    embed_dim = 256
    num_heads = 8
    num_layers = 6
    ff_dim = 1024
    max_seq_len = 128
    dropout = 0.1
    
    # Training parameters
    batch_size = 32
    learning_rate = 3e-4
    max_training_time = 3600  # 1 hour in seconds
    checkpoint_interval = 600  # Save every 10 minutes
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

# Tokenizer (simple character-level)
class SimpleTokenizer:
    def __init__(self, text, vocab_size=5000):
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        
    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char.get(i, '') for i in indices])

# Dataset
class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

# Transformer Model
class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_encoding = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # Embeddings
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(x) + self.pos_encoding(pos)
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.fc_out(x)
        
        return x

def train():
    print(f"Using device: {config.device}")
    
    # Load and prepare data
    print("Loading data from mybooks.txt...")
    if not os.path.exists('mybooks.txt'):
        print("ERROR: mybooks.txt not found!")
        print("Please create a file named 'mybooks.txt' in the same directory as this script.")
        return
    
    with open('mybooks.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Loaded {len(text)} characters")
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = SimpleTokenizer(text)
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {config.vocab_size}")
    
    # Tokenize
    tokens = tokenizer.encode(text)
    
    # Create dataset and dataloader
    dataset = TextDataset(tokens, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    
    # Create model
    print("Initializing model...")
    model = TransformerLM(config).to(config.device)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    print("\nStarting training...")
    print(f"Max training time: {config.max_training_time} seconds (1 hour)")
    print("-" * 60)
    
    model.train()
    start_time = time.time()
    total_loss = 0
    batch_count = 0
    epoch = 0
    
    try:
        while True:
            epoch += 1
            epoch_loss = 0
            epoch_batches = 0
            
            for batch_idx, (x, y) in enumerate(dataloader):
                # Check time limit
                elapsed = time.time() - start_time
                if elapsed >= config.max_training_time:
                    print("\nTime limit reached!")
                    raise KeyboardInterrupt
                
                x, y = x.to(config.device), y.to(config.device)
                
                # Forward pass
                logits = model(x)
                loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Track loss
                total_loss += loss.item()
                epoch_loss += loss.item()
                batch_count += 1
                epoch_batches += 1
                
                # Print progress
                if batch_count % 10 == 0:
                    avg_loss = total_loss / batch_count
                    elapsed = time.time() - start_time
                    remaining = config.max_training_time - elapsed
                    print(f"Epoch {epoch} | Batch {batch_count} | Loss: {loss.item():.4f} | "
                          f"Avg Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s | Remaining: {remaining:.0f}s")
                
                # Checkpoint
                if elapsed > 0 and int(elapsed) % config.checkpoint_interval == 0 and int(elapsed) != int(elapsed - 1):
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': avg_loss,
                    }, f'checkpoint_epoch_{epoch}.pt')
                    print(f"Checkpoint saved at {elapsed:.0f} seconds")
            
            avg_epoch_loss = epoch_loss / epoch_batches
            print(f"\nEpoch {epoch} completed | Avg Loss: {avg_epoch_loss:.4f}\n")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    
    # Save final model
    print("\nSaving final model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': config.vocab_size,
            'embed_dim': config.embed_dim,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,
            'ff_dim': config.ff_dim,
            'max_seq_len': config.max_seq_len,
            'dropout': config.dropout
        },
        'tokenizer_char_to_idx': tokenizer.char_to_idx,
        'tokenizer_idx_to_char': tokenizer.idx_to_char,
        'epoch': epoch,
        'loss': total_loss / batch_count if batch_count > 0 else 0,
    }, 'final_model.pt')
    
    total_time = time.time() - start_time
    print(f"\nTraining completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Total batches: {batch_count}")
    print(f"Final average loss: {total_loss / batch_count if batch_count > 0 else 0:.4f}")
    print(f"Model saved as 'final_model.pt'")

if __name__ == "__main__":
    train()
