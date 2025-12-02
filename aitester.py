import torch
import torch.nn as nn
import torch.nn.functional as F

# Model definition (same as training script)
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(x) + self.pos_encoding(pos)
        x = self.dropout(x)
        
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.fc_out(x)
        
        return x

class SimpleTokenizer:
    def __init__(self, char_to_idx, idx_to_char):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = len(char_to_idx)
        
    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char.get(i, '') for i in indices])

def load_model(checkpoint_path='final_model.pt'):
    """Load the trained model and tokenizer"""
    print(f"Loading model from {checkpoint_path}...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config = checkpoint['config']
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(
        checkpoint['tokenizer_char_to_idx'],
        checkpoint['tokenizer_idx_to_char']
    )
    
    # Create model
    model = TransformerLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Device: {device}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training loss: {checkpoint.get('loss', 'N/A')}")
    print()
    
    return model, tokenizer, device, config

def generate_text(model, tokenizer, device, config, prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    """Generate text from a prompt"""
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    
    generated = tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Limit context to max_seq_len
            context = generated[:, -config.max_seq_len:]
            
            # Get predictions
            logits = model(context)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
    
    # Decode and return
    return tokenizer.decode(generated[0].tolist())

def interactive_mode(model, tokenizer, device, config):
    """Interactive text generation"""
    print("=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Enter a prompt and the AI will continue the text.")
    print("Commands:")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'settings' to adjust generation parameters")
    print("=" * 70)
    print()
    
    # Default settings
    max_tokens = 200
    temperature = 0.8
    top_k = 40
    
    while True:
        prompt = input("\n📝 Enter your prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if prompt.lower() == 'settings':
            print("\nCurrent settings:")
            print(f"  Max tokens: {max_tokens}")
            print(f"  Temperature: {temperature} (higher = more creative, lower = more focused)")
            print(f"  Top-k: {top_k} (lower = more focused, higher = more diverse)")
            
            try:
                max_tokens = int(input(f"\nMax tokens [{max_tokens}]: ") or max_tokens)
                temperature = float(input(f"Temperature [{temperature}]: ") or temperature)
                top_k = int(input(f"Top-k [{top_k}]: ") or top_k)
                print("Settings updated!")
            except ValueError:
                print("Invalid input, keeping current settings.")
            continue
        
        if not prompt:
            print("Please enter a prompt!")
            continue
        
        print("\n🤖 Generating...\n")
        print("-" * 70)
        
        generated = generate_text(
            model, tokenizer, device, config, 
            prompt, max_tokens, temperature, top_k
        )
        
        print(generated)
        print("-" * 70)

def batch_mode(model, tokenizer, device, config):
    """Generate from multiple prompts"""
    print("=" * 70)
    print("BATCH MODE")
    print("=" * 70)
    print("Testing with sample prompts...\n")
    
    prompts = [
        "Once upon a time",
        "The future of",
        "In the beginning",
        "It was a dark and stormy night",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: '{prompt}'")
        print("-" * 70)
        generated = generate_text(model, tokenizer, device, config, prompt, max_new_tokens=150)
        print(generated)
        print("-" * 70)

def main():
    print("\n" + "=" * 70)
    print(" AI MODEL TESTER ".center(70))
    print("=" * 70 + "\n")
    
    # Load model
    try:
        model, tokenizer, device, config = load_model('final_model.pt')
    except FileNotFoundError:
        print("ERROR: final_model.pt not found!")
        print("Please train the model first using the training script.")
        return
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # Choose mode
    print("Select mode:")
    print("  1. Interactive mode (chat with your AI)")
    print("  2. Batch mode (test with sample prompts)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        interactive_mode(model, tokenizer, device, config)
    elif choice == '2':
        batch_mode(model, tokenizer, device, config)
    else:
        print("Invalid choice. Running interactive mode...")
        interactive_mode(model, tokenizer, device, config)

if __name__ == "__main__":
    main()