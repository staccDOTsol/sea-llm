from seahorse.prelude import *

declare_id('CG3rq4URcAwEUNnfGZ3YHRKTZAZgyemttN1BQwypa8qj')

@zero_copy
class ModelRegistry(Account):
  authority: Pubkey
  chunk_count: u32
  vocab_size: u32
  embedding_dim: u32
  hidden_dim: u32
  context_length: u32
  layer_count: u32

@zero_copy
class ModelChunk(Account):
  authority: Pubkey
  registry: Pubkey
  chunk_index: u32
  chunk_type: u8   # 0: embeddings, 1: attention, 2: ffn, 3: output
  data: Array[i32, 8192]  # Weight data

@zero_copy
class ChatState(Account):
  authority: Pubkey
  model: Pubkey  # Points to ModelRegistry
  history: Array[u8, 1024]  # Token history
  history_len: u32

@instruction
def initialize_model(
  signer: Signer, 
  registry: UncheckedAccount,
  vocab_size: u32,
  embedding_dim: u32,
  hidden_dim: u32,
  context_length: u32,
  layer_count: u32
):
  """Initialize a new model registry"""
  # Initialize account
  discriminator = [152, 221, 247, 122, 185, 125, 223, 151]
  for i in range(8):
    registry.data[i] = discriminator[i]
  
  # Set authority
  authority = signer.key().to_bytes()
  for i in range(32):
    registry.data[8 + i] = authority[i]
  
  # Set model parameters
  registry.data[40] = 0  # chunk_count = 0
  registry.data[44] = vocab_size & 0xFF
  registry.data[45] = (vocab_size >> 8) & 0xFF
  registry.data[46] = (vocab_size >> 16) & 0xFF
  registry.data[47] = (vocab_size >> 24) & 0xFF
  
  registry.data[48] = embedding_dim & 0xFF
  registry.data[49] = (embedding_dim >> 8) & 0xFF
  registry.data[50] = (embedding_dim >> 16) & 0xFF
  registry.data[51] = (embedding_dim >> 24) & 0xFF
  
  registry.data[52] = hidden_dim & 0xFF
  registry.data[53] = (hidden_dim >> 8) & 0xFF
  registry.data[54] = (hidden_dim >> 16) & 0xFF
  registry.data[55] = (hidden_dim >> 24) & 0xFF
  
  registry.data[56] = context_length & 0xFF
  registry.data[57] = (context_length >> 8) & 0xFF
  registry.data[58] = (context_length >> 16) & 0xFF
  registry.data[59] = (context_length >> 24) & 0xFF
  
  registry.data[60] = layer_count & 0xFF
  registry.data[61] = (layer_count >> 8) & 0xFF
  registry.data[62] = (layer_count >> 16) & 0xFF
  registry.data[63] = (layer_count >> 24) & 0xFF

@instruction
def upload_chunk(
  signer: Signer,
  registry: ModelRegistry,
  chunk: UncheckedAccount,
  chunk_index: u32,
  chunk_type: u8,
  data: Array[i32, 1024]  # Upload in smaller chunks
):
  """Upload a chunk of model weights"""
  assert registry.authority == signer.key(), "Invalid authority"
  
  # Initialize chunk account if new
  if chunk_index >= registry.chunk_count:
    # Initialize account
    discriminator = [153, 222, 248, 123, 186, 126, 224, 153]
    for i in range(8):
      chunk.data[i] = discriminator[i]
    
    # Set authority
    authority = signer.key().to_bytes()
    for i in range(32):
      chunk.data[8 + i] = authority[i]
    
    # Set registry key
    registry_bytes = registry.key().to_bytes()
    for i in range(32):
      chunk.data[40 + i] = registry_bytes[i]
    
    # Set chunk info
    chunk.data[72] = chunk_index & 0xFF
    chunk.data[73] = (chunk_index >> 8) & 0xFF
    chunk.data[74] = (chunk_index >> 16) & 0xFF
    chunk.data[75] = (chunk_index >> 24) & 0xFF
    
    chunk.data[76] = chunk_type
    
    # Increment chunk count
    registry.chunk_count = max(registry.chunk_count, chunk_index + 1)
  
  # Copy chunk data (assuming this is appending to existing data)
  # In a real implementation, you'd need more logic to handle offsets
  offset = 80  # Start of data section
  for i in range(len(data)):
    chunk.data[offset + i * 4] = data[i] & 0xFF
    chunk.data[offset + i * 4 + 1] = (data[i] >> 8) & 0xFF
    chunk.data[offset + i * 4 + 2] = (data[i] >> 16) & 0xFF
    chunk.data[offset + i * 4 + 3] = (data[i] >> 24) & 0xFF

@instruction
def initialize_chat(
  signer: Signer,
  chat: UncheckedAccount,
  model: ModelRegistry
):
  """Initialize a new chat session"""
  # Initialize account
  discriminator = [154, 223, 249, 124, 187, 127, 225, 154]
  for i in range(8):
    chat.data[i] = discriminator[i]
  
  # Set authority
  authority = signer.key().to_bytes()
  for i in range(32):
    chat.data[8 + i] = authority[i]
  
  # Set model registry
  model_bytes = model.key().to_bytes()
  for i in range(32):
    chat.data[40 + i] = model_bytes[i]
  
  # Initialize history length
  chat.data[72] = 0
  chat.data[73] = 0
  chat.data[74] = 0
  chat.data[75] = 0

@instruction
def chat(
  signer: Signer,
  chat_state: ChatState,
  input_text: Array[u8, 256],
  input_length: u32
):
  """Process a chat message and generate a response"""
  assert chat_state.authority == signer.key(), "Invalid authority"
  assert input_length <= 256, "Input too long"
  
  # Add input to history
  for i in range(input_length):
    if chat_state.history_len < 1024:
      chat_state.history[chat_state.history_len] = input_text[i]
      chat_state.history_len += 1
  
  # Print input message
  print("==== USER INPUT ====")
  input_str = ""
  for i in range(input_length):
    input_str += chr(input_text[i])
  print(input_str)
  
  # In a real implementation, we would access the model chunks
  # and perform inference, but for now we'll return a fixed response
  print("==== LLM RESPONSE ====")
  print("This is a placeholder response from the on-chain LLM.")
  print("For full inference, use the off-chain client with on-chain weights.") 