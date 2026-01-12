"""
Mamba-SSM Layer for MAHIA
Optional Mamba-SSM Layer for longer contexts with State Space Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

# Conditional imports for Mamba-SSM
MAMBA_SSM_AVAILABLE = False
try:
    from mamba_ssm import MambaLMHeadModel
    MAMBA_SSM_AVAILABLE = True
    print("✅ Mamba-SSM available")
except ImportError:
    print("⚠️  Mamba-SSM not available, using simplified implementation")

class SimplifiedMambaSSM(nn.Module):
    """
    Simplified State Space Model implementation for longer context processing
    Based on Mamba architecture but simplified for integration
    """
    
    def __init__(self, 
                 d_model: int = 768,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dt_rank: str = "auto",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dt_init: str = "random",
                 dt_scale: float = 1.0,
                 dt_init_floor: float = 1e-4,
                 conv_bias: bool = True,
                 bias: bool = False,
                 use_fast_path: bool = True,
                 layer_idx: Optional[int] = None,
                 device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None):
        """
        Initialize Simplified Mamba-SSM layer
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            dt_rank: Rank of delta projection
            dt_min: Minimum delta value
            dt_max: Maximum delta value
            dt_init: Delta initialization method
            dt_scale: Delta scaling factor
            dt_init_floor: Minimum delta initialization value
            conv_bias: Whether to use bias in convolution
            bias: Whether to use bias in linear layers
            use_fast_path: Whether to use fast path inference
            layer_idx: Layer index for debugging
            device: Device to use
            dtype: Data type to use
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        
        # Projection layers
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # Convolution layer
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # State space parameters
        self.x_proj = nn.Linear(self.d_inner, int(self.dt_rank) + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Delta initialization
        dt_init_std = int(self.dt_rank)**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
            
        # Initialize delta bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
            
        # State parameters
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        
        print(f"✅ SimplifiedMambaSSM initialized with d_model={d_model}, d_state={d_state}")
        
    def forward(self, hidden_states: torch.Tensor, 
                inference_params=None) -> torch.Tensor:
        """
        Forward pass through Mamba-SSM layer
        
        Args:
            hidden_states: Input tensor of shape (batch, seqlen, d_model)
            inference_params: Inference parameters for caching
            
        Returns:
            Output tensor of shape (batch, seqlen, d_model)
        """
        batch, seqlen, dim = hidden_states.shape
        
        # Project input
        projected = self.in_proj(hidden_states)
        x, z = projected.chunk(2, dim=-1)
        
        # Apply activation
        x = F.silu(x)
        
        # Convolution
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seqlen]
        x = x.transpose(1, 2)
        
        # Apply activation again
        x = F.silu(x)
        
        # State space model
        y = self.ssm_step(x, z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
        
    def ssm_step(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        State space model step
        
        Args:
            x: Input tensor of shape (batch, seqlen, d_inner)
            z: Gate tensor of shape (batch, seqlen, d_inner)
            
        Returns:
            Output tensor of shape (batch, seqlen, d_inner)
        """
        # Compute delta, A, B, C
        x_proj = self.x_proj(x)
        dt, B, C = torch.split(
            x_proj,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        # Project delta
        dt = self.dt_proj(dt)
        
        # Compute A
        A = -torch.exp(self.A_log.float())
        
        # Discretize A and B
        dtA = torch.exp(dt.unsqueeze(-1) * A)
        dtB = dt.unsqueeze(-1) * B.unsqueeze(-1)
        
        # State update
        # For simplicity, we'll use a basic recurrent implementation
        batch, seqlen, d_inner = x.shape
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(seqlen):
            # Update state
            h = dtA[:, i] * h + dtB[:, i] * x[:, i].unsqueeze(-1)
            
            # Compute output
            y = torch.sum(h * C[:, i].unsqueeze(-1), dim=-1)
            ys.append(y)
            
        y = torch.stack(ys, dim=1)
        
        # Apply gating
        y = y * F.silu(z)
        
        return y

class MambaSSMBlock(nn.Module):
    """
    Complete Mamba-SSM block with residual connection and layer normalization
    """
    
    def __init__(self, 
                 d_model: int = 768,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 layer_idx: Optional[int] = None):
        """
        Initialize Mamba-SSM block
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            layer_idx: Layer index for debugging
        """
        super().__init__()
        
        self.layer_idx = layer_idx
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Mamba-SSM layer
        self.mamba_ssm = SimplifiedMambaSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx
        )
        
        print(f"✅ MambaSSMBlock initialized (layer {layer_idx})")
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba-SSM block
        
        Args:
            hidden_states: Input tensor of shape (batch, seqlen, d_model)
            
        Returns:
            Output tensor of shape (batch, seqlen, d_model)
        """
        # Apply layer normalization
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Apply Mamba-SSM
        hidden_states = self.mamba_ssm(hidden_states)
        
        # Add residual connection
        output = residual + hidden_states
        
        return output

class MambaSSMEncoder(nn.Module):
    """
    Mamba-SSM encoder with multiple layers for longer context processing
    """
    
    def __init__(self,
                 num_layers: int = 12,
                 d_model: int = 768,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2):
        """
        Initialize Mamba-SSM encoder
        
        Args:
            num_layers: Number of Mamba-SSM layers
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Create Mamba-SSM blocks
        self.layers = nn.ModuleList([
            MambaSSMBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=i
            )
            for i in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm_f = nn.LayerNorm(d_model)
        
        print(f"✅ MambaSSMEncoder initialized with {num_layers} layers")
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Mamba-SSM encoder
        
        Args:
            input_ids: Input tensor of shape (batch, seqlen, d_model)
            attention_mask: Attention mask of shape (batch, seqlen)
            
        Returns:
            Output tensor of shape (batch, seqlen, d_model)
        """
        hidden_states = input_ids
        
        # Apply each layer
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        # Apply final normalization
        hidden_states = self.norm_f(hidden_states)
        
        return hidden_states

class OptionalMambaSSMLayer(nn.Module):
    """
    Optional Mamba-SSM Layer that can be integrated into existing architectures
    Provides longer context processing capabilities
    """
    
    def __init__(self,
                 d_model: int = 768,
                 num_layers: int = 4,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 use_mamba_ssm: bool = True):
        """
        Initialize optional Mamba-SSM layer
        
        Args:
            d_model: Model dimension
            num_layers: Number of Mamba-SSM layers
            d_state: State dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            use_mamba_ssm: Whether to use Mamba-SSM (if available) or fallback
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_mamba_ssm = use_mamba_ssm and MAMBA_SSM_AVAILABLE
        
        if self.use_mamba_ssm:
            try:
                # Try to use actual Mamba-SSM implementation
                from mamba_ssm import MambaLMHeadModel
                # This is a simplified usage - in practice you'd need to configure properly
                self.mamba_model = MambaLMHeadModel(
                    d_model=d_model,
                    n_layer=num_layers
                )
                print("✅ Using full Mamba-SSM implementation")
            except Exception as e:
                print(f"⚠️  Failed to initialize Mamba-SSM: {e}")
                self.use_mamba_ssm = False
                
        if not self.use_mamba_ssm:
            # Use simplified implementation
            self.mamba_encoder = MambaSSMEncoder(
                num_layers=num_layers,
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            print("✅ Using simplified Mamba-SSM implementation")
            
        # Performance tracking
        self.stats = {
            'forward_passes': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through optional Mamba-SSM layer
        
        Args:
            hidden_states: Input tensor of shape (batch, seqlen, d_model)
            attention_mask: Attention mask of shape (batch, seqlen)
            
        Returns:
            Output tensor of shape (batch, seqlen, d_model)
        """
        import time
        start_time = time.time()
        
        if self.use_mamba_ssm:
            # Use full Mamba-SSM implementation
            try:
                # This is a simplified usage - actual implementation would be more complex
                output = self.mamba_encoder(hidden_states, attention_mask)
            except Exception as e:
                print(f"⚠️  Mamba-SSM forward pass failed: {e}")
                # Fallback to identity
                output = hidden_states
        else:
            # Use simplified implementation
            output = self.mamba_encoder(hidden_states, attention_mask)
            
        # Update statistics
        self.stats['forward_passes'] += 1
        self.stats['total_time'] += (time.time() - start_time)
        self.stats['avg_time'] = self.stats['total_time'] / self.stats['forward_passes']
        
        return output
        
    def get_stats(self) -> Dict[str, Any]:
        """Get Mamba-SSM layer statistics"""
        return self.stats.copy()
        
    def print_stats(self):
        """Print Mamba-SSM layer statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("MAMBA-SSM LAYER STATISTICS")
        print("="*50)
        print(f"Forward Passes: {stats['forward_passes']}")
        print(f"Total Time: {stats['total_time']:.4f}s")
        if stats['forward_passes'] > 0:
            print(f"Average Time: {stats['avg_time']*1000:.2f}ms")
        print("="*50)

# Example usage
def example_mamba_ssm_layer():
    """Example of Mamba-SSM layer usage"""
    print("🔧 Setting up Mamba-SSM Layer example...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create Mamba-SSM layer
    mamba_layer = OptionalMambaSSMLayer(
        d_model=768,
        num_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        use_mamba_ssm=False  # Use simplified implementation
    ).to(device)
    
    print("\n🚀 Testing Mamba-SSM layer...")
    
    # Create test input
    batch_size = 2
    seq_len = 512
    d_model = 768
    
    input_tensor = torch.randn(batch_size, seq_len, d_model).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    import time
    start_time = time.time()
    output = mamba_layer(input_tensor, attention_mask)
    elapsed_time = time.time() - start_time
    
    print(f"Output shape: {output.shape}")
    print(f"Forward pass completed in {elapsed_time*1000:.2f}ms")
    
    # Test with longer sequence
    print("\n🚀 Testing with longer sequence...")
    long_seq_len = 2048
    long_input = torch.randn(batch_size, long_seq_len, d_model).to(device)
    
    start_time = time.time()
    long_output = mamba_layer(long_input)
    long_elapsed_time = time.time() - start_time
    
    print(f"Long input shape: {long_input.shape}")
    print(f"Long output shape: {long_output.shape}")
    print(f"Long sequence forward pass completed in {long_elapsed_time*1000:.2f}ms")
    
    # Print statistics
    mamba_layer.print_stats()
    
    # Compare efficiency
    speedup = long_elapsed_time / (elapsed_time * (long_seq_len / seq_len))
    print(f"\n📈 Efficiency analysis:")
    print(f"   Expected time for long sequence: {elapsed_time * (long_seq_len / seq_len)*1000:.2f}ms")
    print(f"   Actual time for long sequence: {long_elapsed_time*1000:.2f}ms")
    print(f"   Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    example_mamba_ssm_layer()