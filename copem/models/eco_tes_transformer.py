# Developed by: [Your Research Team]
# Institution: Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering
# Date: December 15, 2025
#!/usr/bin/env python3
# Developed by: [Your Research Team]
# Institution: Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering
# Date: December 15, 2025
"""
# Developed by: [Your Research Team]
# Institution: Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering
# Date: December 15, 2025
Eco-TES Transformer - Eco-Technical-Energy-Saving Transformer
# Developed by: [Your Research Team]
# Institution: Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering
# Date: December 15, 2025
完整的电池安全操作包络预测模型，包含GTCA机制和训练功能
# Developed by: [Your Research Team]
# Institution: Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering
# Date: December 15, 2025

# Developed by: [Your Research Team]
# Institution: Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering
# Date: December 15, 2025
This module implements the complete Eco-TES Transformer for battery safe operating
# Developed by: [Your Research Team]
# Institution: Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering
# Date: December 15, 2025
envelope prediction with Gated Temporal-Channel Attention (GTCA) mechanism.
# Developed by: [Your Research Team]
# Institution: Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering
# Date: December 15, 2025
"""
# Developed by: [Your Research Team]
# Institution: Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering
# Date: December 15, 2025

# Developed by: [Your Research Team]
# Institution: Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering
# Date: December 15, 2025
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class GTCABlock(nn.Module):
    """Gated Temporal-Channel Attention Block"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(GTCABlock, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Temporal attention components
        self.temporal_query = nn.Linear(input_dim, hidden_dim)
        self.temporal_key = nn.Linear(input_dim, hidden_dim)
        self.temporal_value = nn.Linear(input_dim, hidden_dim)
        
        # Channel attention components
        self.channel_query = nn.Linear(input_dim, hidden_dim)
        self.channel_key = nn.Linear(input_dim, hidden_dim)
        self.channel_value = nn.Linear(input_dim, hidden_dim)
        
        # Gating mechanism
        self.gate_temporal = nn.Linear(hidden_dim, hidden_dim)
        self.gate_channel = nn.Linear(hidden_dim, hidden_dim)
        self.gate_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Consensus enhancement components
        self.consensus_weight = nn.Parameter(torch.ones(1))
        self.consensus_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, consensus_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of GTCA block
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            consensus_features: Consensus features [batch_size, seq_len, hidden_dim]
            
        Returns:
            Enhanced features [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        if x.size(-1) != self.hidden_dim:
            x_proj = F.linear(x, torch.randn(self.hidden_dim, x.size(-1)).to(x.device))
        else:
            x_proj = x
        
        # Temporal attention
        q_temporal = self.temporal_query(x_proj)
        k_temporal = self.temporal_key(x_proj)
        v_temporal = self.temporal_value(x_proj)
        
        # Channel attention (transpose for channel-wise attention)
        x_transposed = x_proj.transpose(1, 2)  # [batch, features, seq_len]
        q_channel = self.channel_query(x_transposed).transpose(1, 2)
        k_channel = self.channel_key(x_transposed).transpose(1, 2)
        v_channel = self.channel_value(x_transposed).transpose(1, 2)
        
        # Apply multi-head attention for temporal
        temporal_attn, _ = self.multihead_attn(q_temporal, k_temporal, v_temporal)
        
        # Apply multi-head attention for channel
        channel_attn, _ = self.multihead_attn(q_channel, k_channel, v_channel)
        
        # Gating mechanism
        gate_t = torch.sigmoid(self.gate_temporal(temporal_attn))
        gate_c = torch.sigmoid(self.gate_channel(channel_attn))
        
        # Gated features
        gated_temporal = gate_t * temporal_attn
        gated_channel = gate_c * channel_attn
        
        # Fusion
        fused_features = torch.cat([gated_temporal, gated_channel], dim=-1)
        fused_output = self.gate_fusion(fused_features)
        
        # Consensus enhancement
        if consensus_features is not None:
            consensus_enhanced = self.consensus_projection(consensus_features)
            fused_output = fused_output + self.consensus_weight * consensus_enhanced
        
        # Residual connection and layer norm
        output = self.layer_norm1(x_proj + self.dropout(fused_output))
        
        # Feed-forward network
        ffn_output = self.ffn(output)
        output = self.layer_norm2(output + ffn_output)
        
        return output

class BatteryEnvelopePredictor(nn.Module):
    """Battery Safe Operating Envelope Prediction Head"""
    
    def __init__(self, input_dim: int, output_dim: int = 8):
        super(BatteryEnvelopePredictor, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Envelope prediction layers
        self.voltage_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 2)  # [V_min, V_max]
        )
        
        self.current_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 2)  # [I_min, I_max]
        )
        
        self.temperature_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 2)  # [T_min, T_max]
        )
        
        self.power_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 2)  # [P_min, P_max]
        )
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict battery safe operating envelope
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Dictionary containing envelope predictions
        """
        # Use the last time step for prediction
        x_last = x[:, -1, :]  # [batch_size, input_dim]
        
        # Predict envelope boundaries
        voltage_bounds = self.voltage_predictor(x_last)
        current_bounds = self.current_predictor(x_last)
        temperature_bounds = self.temperature_predictor(x_last)
        power_bounds = self.power_predictor(x_last)
        
        # Predict confidence
        confidence = self.confidence_predictor(x_last)
        
        return {
            'voltage_min': voltage_bounds[:, 0],
            'voltage_max': voltage_bounds[:, 1],
            'current_min': current_bounds[:, 0],
            'current_max': current_bounds[:, 1],
            'temperature_min': temperature_bounds[:, 0],
            'temperature_max': temperature_bounds[:, 1],
            'power_min': power_bounds[:, 0],
            'power_max': power_bounds[:, 1],
            'confidence': confidence.squeeze(-1)
        }

class EcoTESTransformer(nn.Module):
    """
    Eco-TES (Eco-Technical-Energy-Saving) Transformer
    
    Advanced neural architecture for battery safe operating envelope prediction
    with Gated Temporal-Channel Attention mechanism and consensus enhancement.
    """
    
    def __init__(self, config: Dict):
        super(EcoTESTransformer, self).__init__()
        
        self.config = config
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.sequence_length = config['sequence_length']
        self.prediction_horizon = config['prediction_horizon']
        self.dropout = config['dropout']
        
        # Input embedding
        self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.positional_encoding = self._create_positional_encoding()
        
        # GTCA blocks
        self.gtca_blocks = nn.ModuleList([
            GTCABlock(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Battery envelope predictor
        self.envelope_predictor = BatteryEnvelopePredictor(
            input_dim=self.hidden_dim,
            output_dim=8
        )
        
        # Consensus feature processor
        self.consensus_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Training components
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.scheduler = None
        
        # Performance tracking
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'envelope_accuracy': [],
            'consensus_alignment': []
        }
        
        print(f"✅ Eco-TES Transformer initialized with {self.num_layers} GTCA blocks")
        
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create positional encoding for sequence modeling"""
        pe = torch.zeros(self.sequence_length, self.hidden_dim)
        position = torch.arange(0, self.sequence_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * 
                           (-math.log(10000.0) / self.hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, seq_len, hidden_dim]
    
    def forward(self, x: torch.Tensor, consensus_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Eco-TES Transformer
        
        Args:
            x: Input battery state sequence [batch_size, seq_len, input_dim]
            consensus_features: Consensus-enhanced features [batch_size, seq_len, hidden_dim]
            
        Returns:
            Dictionary containing predictions and envelope
        """
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        embedded = self.input_embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
        embedded = embedded + pos_encoding
        
        # Process consensus features if provided
        processed_consensus = None
        if consensus_features is not None:
            processed_consensus = self.consensus_processor(consensus_features)
        
        # Pass through GTCA blocks
        hidden_states = embedded
        for gtca_block in self.gtca_blocks:
            hidden_states = gtca_block(hidden_states, processed_consensus)
        
        # Predict battery envelope
        envelope_predictions = self.envelope_predictor(hidden_states)
        
        return {
            'envelope_predictions': envelope_predictions,
            'hidden_states': hidden_states,
            'consensus_alignment': self._calculate_consensus_alignment(
                hidden_states, processed_consensus
            ) if processed_consensus is not None else torch.tensor(0.0)
        }
    
    def _calculate_consensus_alignment(self, hidden_states: torch.Tensor, 
                                     consensus_features: torch.Tensor) -> torch.Tensor:
        """Calculate alignment between hidden states and consensus features"""
        if consensus_features is None:
            return torch.tensor(0.0)
        
        # Cosine similarity between hidden states and consensus features
        hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
        consensus_norm = F.normalize(consensus_features, p=2, dim=-1)
        
        similarity = torch.sum(hidden_norm * consensus_norm, dim=-1)
        return torch.mean(similarity)
    
    def setup_training(self, learning_rate: float = 1e-4):
        """Setup training components"""
        self.optimizer = optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000, 
            eta_min=1e-6
        )
        
        print(f"✅ Training setup complete with learning rate {learning_rate}")
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.train()
        
        # Extract batch data
        input_sequences = batch_data['input_sequences']
        target_envelopes = batch_data['target_envelopes']
        consensus_features = batch_data.get('consensus_features', None)
        
        # Forward pass
        outputs = self.forward(input_sequences, consensus_features)
        envelope_pred = outputs['envelope_predictions']
        consensus_alignment = outputs['consensus_alignment']
        
        # Calculate losses
        envelope_loss = self._calculate_envelope_loss(envelope_pred, target_envelopes)
        consensus_loss = -consensus_alignment if consensus_features is not None else torch.tensor(0.0)
        
        # Total loss with consensus regularization
        total_loss = envelope_loss + 0.1 * consensus_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'envelope_loss': envelope_loss.item(),
            'consensus_loss': consensus_loss.item() if isinstance(consensus_loss, torch.Tensor) else 0.0,
            'consensus_alignment': consensus_alignment.item() if isinstance(consensus_alignment, torch.Tensor) else 0.0
        }
    
    def _calculate_envelope_loss(self, predictions: Dict[str, torch.Tensor], 
                                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate envelope prediction loss"""
        total_loss = 0.0
        
        for key in ['voltage_min', 'voltage_max', 'current_min', 'current_max',
                   'temperature_min', 'temperature_max', 'power_min', 'power_max']:
            if key in predictions and key in targets:
                total_loss += F.mse_loss(predictions[key], targets[key])
        
        # Confidence loss
        if 'confidence' in predictions and 'confidence' in targets:
            total_loss += F.binary_cross_entropy(predictions['confidence'], targets['confidence'])
        
        return total_loss
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation step"""
        self.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Forward pass
                outputs = self.forward(
                    batch_data['input_sequences'],
                    batch_data.get('consensus_features', None)
                )
                
                # Calculate loss
                envelope_loss = self._calculate_envelope_loss(
                    outputs['envelope_predictions'],
                    batch_data['target_envelopes']
                )
                
                # Calculate accuracy
                accuracy = self._calculate_envelope_accuracy(
                    outputs['envelope_predictions'],
                    batch_data['target_envelopes']
                )
                
                total_loss += envelope_loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
    
    def _calculate_envelope_accuracy(self, predictions: Dict[str, torch.Tensor],
                                   targets: Dict[str, torch.Tensor]) -> float:
        """Calculate envelope prediction accuracy"""
        total_accuracy = 0.0
        num_metrics = 0
        
        for key in ['voltage_min', 'voltage_max', 'current_min', 'current_max',
                   'temperature_min', 'temperature_max', 'power_min', 'power_max']:
            if key in predictions and key in targets:
                # Calculate relative error
                rel_error = torch.abs(predictions[key] - targets[key]) / (torch.abs(targets[key]) + 1e-8)
                accuracy = torch.mean((rel_error < 0.1).float())  # 10% tolerance
                total_accuracy += accuracy.item()
                num_metrics += 1
        
        return total_accuracy / num_metrics if num_metrics > 0 else 0.0
    
    def predict_envelope(self, battery_state_sequence: np.ndarray,
                        consensus_features: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Predict battery safe operating envelope for a single sequence
        
        Args:
            battery_state_sequence: Battery state sequence [seq_len, input_dim]
            consensus_features: Consensus features [seq_len, hidden_dim]
            
        Returns:
            Dictionary containing envelope predictions
        """
        self.eval()
        
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(battery_state_sequence).unsqueeze(0)
            
            consensus_tensor = None
            if consensus_features is not None:
                consensus_tensor = torch.FloatTensor(consensus_features).unsqueeze(0)
            
            # Forward pass
            outputs = self.forward(input_tensor, consensus_tensor)
            envelope_pred = outputs['envelope_predictions']
            
            # Convert to numpy and remove batch dimension
            result = {}
            for key, value in envelope_pred.items():
                result[key] = value.squeeze(0).cpu().numpy().item()
            
            return result
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }, filepath)
        print(f"💾 Eco-TES Transformer saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"📁 Eco-TES Transformer loaded from {filepath}")

class BatteryDataset(Dataset):
    """Dataset for battery state sequences and envelope targets"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, 
                 consensus_features: Optional[np.ndarray] = None):
        """
        Args:
            sequences: Input sequences [num_samples, seq_len, input_dim]
            targets: Target envelopes [num_samples, 8]
            consensus_features: Consensus features [num_samples, seq_len, hidden_dim]
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.consensus_features = None
        
        if consensus_features is not None:
            self.consensus_features = torch.FloatTensor(consensus_features)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = {
            'input_sequences': self.sequences[idx],
            'target_envelopes': self._format_targets(self.targets[idx])
        }
        
        if self.consensus_features is not None:
            item['consensus_features'] = self.consensus_features[idx]
        
        return item
    
    def _format_targets(self, target_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Format target vector into dictionary"""
        return {
            'voltage_min': target_vector[0],
            'voltage_max': target_vector[1],
            'current_min': target_vector[2],
            'current_max': target_vector[3],
            'temperature_min': target_vector[4],
            'temperature_max': target_vector[5],
            'power_min': target_vector[6],
            'power_max': target_vector[7]
        }

def train_eco_tes_transformer(config: Dict, train_data: Dict, val_data: Dict,
                             num_epochs: int = 100) -> EcoTESTransformer:
    """
    Complete training function for Eco-TES Transformer
    
    Args:
        config: Model configuration
        train_data: Training data dictionary
        val_data: Validation data dictionary
        num_epochs: Number of training epochs
        
    Returns:
        Trained Eco-TES Transformer model
    """
    print("🚀 Starting Eco-TES Transformer training...")
    
    # Create model
    model = EcoTESTransformer(config)
    model.setup_training(learning_rate=1e-4)
    
    # Create datasets and dataloaders
    train_dataset = BatteryDataset(
        train_data['sequences'],
        train_data['targets'],
        train_data.get('consensus_features', None)
    )
    
    val_dataset = BatteryDataset(
        val_data['sequences'],
        val_data['targets'],
        val_data.get('consensus_features', None)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        for batch_data in train_loader:
            train_metrics = model.train_step(batch_data)
            epoch_train_loss += train_metrics['total_loss']
            num_train_batches += 1
        
        # Validation
        val_metrics = model.validate(val_loader)
        
        # Update learning rate
        model.scheduler.step()
        
        # Record metrics
        avg_train_loss = epoch_train_loss / num_train_batches
        model.training_history['train_loss'].append(avg_train_loss)
        model.training_history['val_loss'].append(val_metrics['val_loss'])
        model.training_history['envelope_accuracy'].append(val_metrics['val_accuracy'])
        
        # Early stopping
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            # Save best model
            model.save_model('eco_tes_best_model.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            print(f"  LR: {model.optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print("✅ Eco-TES Transformer training completed!")
    return model

if __name__ == "__main__":
    # Example usage and testing
    config = {
        'input_dim': 16,
        'hidden_dim': 128,
        'num_heads': 8,
        'num_layers': 4,
        'sequence_length': 50,
        'prediction_horizon': 10,
        'dropout': 0.1
    }
    
    # Create model
    model = EcoTESTransformer(config)
    
    # Generate sample data for testing
    batch_size = 8
    seq_len = 50
    input_dim = 16
    
    sample_input = torch.randn(batch_size, seq_len, input_dim)
    sample_consensus = torch.randn(batch_size, seq_len, config['hidden_dim'])
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(sample_input, sample_consensus)
        print("✅ Forward pass successful!")
        print(f"Envelope predictions keys: {list(outputs['envelope_predictions'].keys())}")
        print(f"Consensus alignment: {outputs['consensus_alignment'].item():.4f}")
    
    print("✅ Eco-TES Transformer implementation complete!") 