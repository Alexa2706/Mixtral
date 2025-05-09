from typing import Optional, Tuple

from jax import Array
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import os

from flax.linen import combine_masks
from jax import lax
from dataclasses import dataclass
import flax.linen as nn
import optax

from flaxconfigmixtral import MixtralConfig
from transformers.modeling_flax_utils import ACT2FN

@dataclass
class FlaxMoeModelOutputWithPast:
    last_hidden_state: Array
    hidden_states: Optional[Tuple[Array]] = None
    attentions: Optional[Tuple[Array]] = None
    router_logits: Optional[Tuple[Array]] = None

@dataclass
class FlaxMoeCausalLMOutput:
    logits: jnp.ndarray
    hidden_states: Optional[Tuple[jnp.ndarray]] = None
    attentions: Optional[Tuple[jnp.ndarray]] = None
    router_logits: Optional[Tuple[jnp.ndarray]] = None
    aux_loss: Optional[jnp.ndarray] = None
    loss: Optional[jnp.ndarray] = None



GLOBAL_MESH = None
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
jax.config.update('jax_platform_name', 'cpu')

def initialize_global_mesh(shape=(1, 8), axis_names=('tp', )):
    """Initialize the global mesh for tensor parallelism"""
    global GLOBAL_MESH
    GLOBAL_MESH = Mesh(jax.devices(), axis_names)
    return GLOBAL_MESH

def get_global_mesh():
    """Get the global mesh, initializing if necessary"""
    global GLOBAL_MESH
    if GLOBAL_MESH is None:
        return initialize_global_mesh()
    return GLOBAL_MESH

class MixtralBlockSparseTop2MLP(nnx.Module):
    """MLP module with sparse routing for Mixtral architecture."""
    
    def __init__(self, config: MixtralConfig, dtype, rngs : nnx.Rngs):
        super().__init__()
        embed_dim = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * embed_dim

        self.up_proj = nnx.Linear(embed_dim, inner_dim, use_bias=False, dtype=dtype, rngs = rngs)
        self.gate_proj = nnx.Linear(embed_dim, inner_dim, use_bias=False, dtype=dtype, rngs = rngs)
        self.down_proj = nnx.Linear(inner_dim, embed_dim, use_bias=False, dtype=dtype, rngs = rngs)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        up_proj_states = self.up_proj(hidden_states)
        gate_states = self.act_fn(self.gate_proj(hidden_states))
        hidden_states = self.down_proj(up_proj_states * gate_states)
        return hidden_states

class MixtralSparseMoeBlock(nnx.Module):
    """Sparse Mixture of Experts block for Mixtral."""
    
    def __init__(self, config: MixtralConfig, dtype, rngs : nnx.Rngs):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        
        self.gate = nnx.Linear(
            config.hidden_size, 
            config.num_local_experts, 
            use_bias=False, 
            dtype=jnp.float32, 
            rngs = rngs
        )

        for i in range(self.num_experts):
            setattr(self, f"experts_{i}", MixtralBlockSparseTop2MLP(config, dtype = dtype, rngs = rngs))
        self.jitter_noise = config.router_jitter_noise

    def _get_expert(self, idx):
        """Helper to get expert by index"""
        return getattr(self, f"experts_{idx}")

    def __call__(self, hidden_states: Array) -> Tuple[Array, Array]:
        batch_size, seq_len, hid_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hid_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = jax.nn.softmax(router_logits, axis = 1)
        routing_weights, selected_experts = lax.top_k(router_logits, self.top_k)
        routing_weights /= jnp.sum(routing_weights, axis = -1, keepdims = True)
        
        routing_weights = routing_weights.astype(hidden_states.dtype)

        final_hidden_states = jnp.zeros(
            (batch_size * seq_len, hid_dim), dtype=hidden_states.dtype)

        # Create one-hot representation of selected experts
        for expert_idx in range(self.num_experts):
              expert_layer = self._get_expert(expert_idx)
              for k in range(self.top_k):
                token_mask = (selected_experts[:, k] == expert_idx)
                weight = routing_weights[:, k]
                token_outputs = expert_layer(hidden_states)
                masked_outputs = lax.cond(
                    jnp.sum(token_mask) > 0,
                    lambda _: token_outputs * weight[:, None] * token_mask[:, None],
                    lambda _: jnp.zeros_like(token_outputs),
                    None
                )
                
                final_hidden_states = final_hidden_states + masked_outputs

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hid_dim)
        return final_hidden_states, router_logits

class MixtralRMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization for Mixtral."""
    
    def __init__(self, config: MixtralConfig, dtype=jnp.float32):
        super().__init__()
        self.epsilon = config.rms_norm_eps
        self.weight = jnp.ones(config.hidden_size)

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.epsilon)
        return self.weight * hidden_states.astype(input_dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    """Apply rotary position embeddings to query and key tensors."""
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

def create_sinusoidal_positions(dim, rope_theta):
    inv_freq = 1 / (rope_theta ** (jnp.arange(0, dim, 2) / dim))
    return inv_freq, 1

def make_causal_mask(input_ids_shape, dtype="bool"):
    """Make causal mask for self-attention."""
    batch_size, sequence_length = input_ids_shape.shape
    mask = jnp.ones((batch_size, 1, sequence_length, sequence_length), dtype=dtype)
    return jnp.triu(mask, k=1)

class MixtralRotaryEmbedding(nnx.Module):
    def __init__(self, config: MixtralConfig, dtype: jnp.dtype = jnp.float32):
        self.config = config
        self.dtype = dtype
        self.rope_theta = self.config.rope_theta
        self.rope_type = "default"

        head_dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
        self.inv_freq, self.attention_scaling = create_sinusoidal_positions(head_dim, self.rope_theta)

    def __call__(self, x, position_ids=None):
        if position_ids is None:
            position_ids = jnp.arange(x.shape[-2]).reshape(1, -1)

        # (B, T, S), (B, T)
        inv_freq_expanded = jnp.expand_dims(self.inv_freq, axis=(0, 2))
        inv_freq_expanded = jnp.repeat(inv_freq_expanded, position_ids.shape[0], axis=0)
        # Create expanded position IDs tensor
        position_ids_expanded = jnp.expand_dims(position_ids, axis=1)
        # Compute frequencies
        # Force float32 precision for this computation
        orig_dtype = x.dtype
        freqs = jnp.matmul(
            inv_freq_expanded.astype(jnp.float32),
            position_ids_expanded.astype(jnp.float32)
        )
        freqs = jnp.transpose(freqs, (0, 2, 1))
        # Concatenate frequencies for sin and cos
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        # Compute cos and sin with scaling
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling

        # Return with original dtype
        return cos.astype(orig_dtype), sin.astype(orig_dtype)

class MixtralAttention(nnx.Module):
    def __init__(self, config: MixtralConfig, dtype: jnp.dtype, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.dtype = dtype

        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = self.config.attention_dropout
        self.num_key_value_heads = self.config.num_key_value_heads
        
        # self.mesh = initialize_global_mesh()

        self.q_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            # kernel_init=nnx.with_partitioning(
            #     nnx.initializers.normal(0.02),
            #     (None, 'tp')  # Shard the output dimension
            # ),
            rngs=rngs
        )

        # Key projection
        self.k_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            # kernel_init=nnx.with_partitioning(
            #     nnx.initializers.normal(0.02),
            #     (None, 'tp')  # Shard the output dimension
            # ),
            rngs=rngs
        )

        # Value projection
        self.v_proj = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            # kernel_init=nnx.with_partitioning(
            #     nnx.initializers.normal(0.02),
            #     (None, 'tp')  # Shard the output dimension
            # ),
            rngs=rngs
        )

        # Output projection
        self.o_proj = nnx.Linear(
            in_features=self.num_heads * self.head_dim,
            out_features=self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            # kernel_init=nnx.with_partitioning(
            #     nnx.initializers.normal(0.02),
            #     ('tp', None)  # Shard the input dimension
            # ),
            rngs=rngs
        )

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
        )

        # Create rotary embeddings and causal mask
        self.rotary_emb = MixtralRotaryEmbedding(self.config, dtype=self.dtype)
        casual_mask = make_causal_mask(
            jnp.ones((1, self.config.max_position_embeddings), dtype="bool"),
            dtype="bool"
        )
        self.causal_mask = jnp.triu(casual_mask)
        self.attention_softmax_in_fp32 = self.dtype is not jnp.float32

        # Initialize cache variables
        self.cached_key = None
        self.cached_value = None
        self.cache_index = None

    def __call__(
        self,
        hidden_states,  # (B, T, embed)
        position_ids,   # (1, T)
        attention_mask,
        deterministic: bool = False,
        use_cache: Optional[bool] = True,
        past_key_value = None,  
        **kwargs
    ):
        batch_size, seq_len, embed = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # hidden_states_sharded = jax.device_put(
        #     hidden_states, 
        #     jax.sharding.NamedSharding(self.mesh, P(None, None, 'tp'))
        # )
        # Force JAX to respect this sharding decision
        # hidden_states_sharded = jax.lax.with_sharding_constraint(
        #     hidden_states_sharded, 
        #     P(None, None, 'tp')
        # )

        # Now use this tensor everywhere
        # jax.debug.visualize_array_sharding(hidden_states_sharded[0])
        # Project and reshape the states
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = position_ids
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        query_length, key_length = query_states.shape[1], key_states.shape[1]
        
        if past_key_value is not None:
            past_key, past_value = past_key_value["key"], past_key_value["value"]
            key_states = jnp.concatenate([past_key, key_states], axis=1)
            value_states = jnp.concatenate([past_value, value_states], axis=1)

        causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        if attention_mask is not None:
            attention_mask = jnp.broadcast_to(
                jnp.expand_dims(attention_mask, axis=(-3, -2)), 
                causal_mask.shape
            )
            attention_mask = combine_masks(attention_mask, causal_mask)
        else:
            attention_mask = causal_mask
        
        if use_cache:
            past_key_value = {
                "key": key_states,  # Save the original key/value (before repeating)
                "value": value_states
            }
        
        key_states = jnp.repeat(key_states, self.num_key_value_groups, axis=2)
        value_states = jnp.repeat(value_states, self.num_key_value_groups, axis=2)

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        def attention_fn(q, k, v, bias):
        # Transpose for attention
            q = jnp.transpose(q, (0, 2, 1, 3))  # [batch, heads, seq_len, head_dim]
            # q = jax.lax.with_sharding_constraint(q, P(None, 'tp', None, None))

            k = jnp.transpose(k, (0, 2, 1, 3))
            # k = jax.lax.with_sharding_constraint(q, P(None, 'tp', None, None))

            v = jnp.transpose(v, (0, 2, 1, 3))
            # v = jax.lax.with_sharding_constraint(q, P(None, 'tp', None, None))
            
            # Compute attention scores
            attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * self.scaling
            # Add bias
            attention_scores = attention_scores + bias

            # Apply softmax
            attention_dtype = jnp.float32 if self.attention_softmax_in_fp32 else self.dtype
            # jax.debug.visualize_array_sharding(attention_scores[0, :, 0, :]) 
            attn_weights = jax.nn.softmax(attention_scores, axis=-1).astype(self.dtype)
    
            # Apply dropout if training
            if not deterministic and self.attention_dropout > 0:
                dropout_rng = jax.random.PRNGKey(0)  # In a real implementation, use a proper RNG
                attn_weights = jnp.where(
                        jax.random.uniform(dropout_rng, attn_weights.shape) < self.attention_dropout,
                        jnp.zeros_like(attn_weights),
                        attn_weights / (1.0 - self.attention_dropout)
                )

            # Compute attention output
            attn_output = jnp.matmul(attn_weights, v)

            # Apply sharding constraints - NNX handles this automatically
            # based on parameter annotations

            # Transpose back
            attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
            return attn_output, attn_weights

        attn_output, attn_weights = attention_fn(
            query_states,
            key_states,
            value_states,
            attention_bias
        )

        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        attn_output = self.o_proj(attn_output)
    
        return attn_output, attn_weights, past_key_value

class MixtralDecoderLayer(nnx.Module):
    def __init__(self, config: MixtralConfig, rngs : nnx.Rngs, layer_idx: int, dtype=jnp.float32):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = dtype
        
        self.input_norm = MixtralRMSNorm(config, dtype=dtype)
        self.attn = MixtralAttention(config, dtype=dtype, rngs = rngs)
        self.block_sparse_moe = MixtralSparseMoeBlock(config, dtype=dtype, rngs = rngs)
        self.attn_norm = MixtralRMSNorm(config, dtype=dtype)

    def __call__(
        self, 
        hidden_states,
        attention_mask : Optional[Array] = None,
        position_ids : Optional[Array] = None,
        past_key_value = None,
        output_attentions : Optional[bool] = False,
        deterministic : bool = False,
        output_router_logits : Optional[bool] = False,
        use_cache : Optional[bool] = False,
        position_embeddings : Optional[Tuple[Array, Array]] = None,
        **kwargs
    ):
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)

        #Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attn(
            hidden_states = hidden_states,
            position_ids = position_ids,
            attention_mask = attention_mask,
            deterministic = deterministic,
            past_key_value = past_key_value,
            use_cache = use_cache,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states, )
        
        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)
        
        return outputs

class MixtralModel(nnx.Module):
    """Mixtral model implementation using NNX."""
    
    def __init__(self, config: MixtralConfig, dtype=jnp.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs = nnx.Rngs(0)
        )
        
        self.layers = [
            MixtralDecoderLayer(config=config, layer_idx=layer_idx, dtype=dtype, rngs=nnx.Rngs(0))
            for layer_idx in range(config.num_hidden_layers)
        ]
        
        self.norm = MixtralRMSNorm(config, dtype=dtype)
        self.rotary_emb = MixtralRotaryEmbedding(config, dtype=dtype)


    def __call__(
        self,
        input_ids : Array, 
        attention_mask : Optional[Array] = None,
        position_ids : Optional[Array] = None,
        deterministic : bool = False,
        input_embeds : Optional[Array] = None,
        use_cache : Optional[bool] = None,
        cache = None,
        output_attentions : Optional[bool] = False,
        output_hidden_states : Optional[bool] = None,
        output_router_logits : Optional[bool] = None,
        return_dict : bool = True,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache


        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
            if self.padding_idx is not None:
                mask = (input_ids != self.padding_idx).astype(jnp.float32)
                embeddings = input_embeds * mask[..., None]
        
        if cache is None and use_cache:
            cache = [None for _ in range(len(self.layers))]

        hidden_states = input_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:

            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            layer_outputs = decoder_layer(
                hidden_states = hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic = deterministic,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return FlaxMoeModelOutputWithPast(
            last_hidden_state = hidden_states,
            hidden_states = all_hidden_states,
            attentions = all_self_attns,
            router_logits = all_router_logits,
        ), cache
    
class MixtralForCausalLM(nnx.Module):
    """Mixtral model with a language modeling head implemented in NNX."""
    
    def __init__(
        self, 
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=None
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
            
        # Save configuration
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        
        # Create model components
        self.model = MixtralModel(
            config=config, 
            dtype=dtype,
        )
        
        # Create LM head as a linear layer
        self.lm_head = nnx.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
    
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        inputs_embeds=None,
        use_cache = True,
        labels=None,
        output_attentions=False,
        output_hidden_states=False,
        output_router_logits=False,
    ):
        # Forward pass through the model
        outputs, cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            use_cache = use_cache,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
        )
        
        # Get the last hidden state
        hidden_states = outputs.last_hidden_state
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            
            # Calculate cross entropy loss
            loss = optax.softmax_cross_entropy(
                shift_logits, 
                jax.nn.one_hot(shift_labels, self.config.vocab_size)
            )
            loss = loss.mean()
        
        # Calculate auxiliary loss if router logits available
        aux_loss = None
        if output_router_logits and outputs.router_logits is not None:
            router_logits = outputs.router_logits
            aux_loss = self.load_balancing_loss_func(
                router_logits,
                self.config.num_local_experts,
                self.config.num_experts_per_tok,
                attention_mask,
            )
            
            # Add auxiliary loss to main loss
            if loss is not None:
                loss += self.config.router_aux_loss_coef * aux_loss
        
        return FlaxMoeCausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
            aux_loss=aux_loss,
            loss=loss
        ), cache
    
    def load_balancing_loss_func(
        self,
        router_logits: Tuple[jnp.ndarray],
        num_experts: int,
        top_k: int = 2,
        attention_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Computes auxiliary load balancing loss as in Switch Transformer.
        
        Args:
            router_logits: Tuple of logits from the router, one per layer
            num_experts: Number of experts
            top_k: The number of experts to route per-token
            attention_mask: Attention mask tensor
            
        Returns:
            The auxiliary loss
        """
        if router_logits is None or not isinstance(router_logits, tuple):
            return jnp.array(0.0)
            
        # Stack all layer router logits
        concatenated_gate_logits = jnp.concatenate([layer_gate for layer_gate in router_logits], axis=0)
        
        # Convert to routing weights
        routing_weights = jax.nn.softmax(concatenated_gate_logits, axis=-1)
        
        # Get top-k experts
        _, selected_experts = jax.lax.top_k(routing_weights, top_k)
        
        # Create one-hot mask for selected experts
        expert_mask = jax.nn.one_hot(selected_experts, num_experts)
        
        if attention_mask is None:
            # Compute the percentage of tokens routed to each expert
            tokens_per_expert = jnp.mean(expert_mask, axis=0)
            
            # Compute the average probability of routing to these experts
            router_prob_per_expert = jnp.mean(routing_weights, axis=0)
        else:
            batch_size, sequence_length = attention_mask.shape
            num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)
            
            # Reshape and expand attention mask to align with expert_mask
            expert_attention_mask = jnp.expand_dims(
                jnp.expand_dims(jnp.expand_dims(attention_mask, axis=0), axis=-1), 
                axis=-1
            )
            expert_attention_mask = jnp.tile(
                expert_attention_mask, 
                (num_hidden_layers, 1, 1, top_k, num_experts)
            ).reshape(-1, top_k, num_experts)
            
            # Compute the percentage of tokens routed to each expert with attention mask
            tokens_per_expert = jnp.sum(expert_mask * expert_attention_mask, axis=0) / jnp.sum(
                expert_attention_mask, axis=0
            )
            
            # Create mask for router probabilities
            router_per_expert_attention_mask = jnp.expand_dims(
                jnp.expand_dims(attention_mask, axis=0), 
                axis=-1
            )
            router_per_expert_attention_mask = jnp.tile(
                router_per_expert_attention_mask,
                (num_hidden_layers, 1, 1, num_experts)
            ).reshape(-1, num_experts)
            
            # Compute the average probability of routing to these experts with attention mask
            router_prob_per_expert = jnp.sum(routing_weights * router_per_expert_attention_mask, axis=0) / jnp.sum(
                router_per_expert_attention_mask, axis=0
            )
        
        # Calculate loss
        overall_loss = jnp.sum(tokens_per_expert * router_prob_per_expert)
        return overall_loss * num_experts
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=20,
        cache = None,
        pad_token_id=None,
        eos_token_id=None,
        key=None
    ):
        """Generate text using the model."""
        # Set defaults for special tokens
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(self.config, "eos_token_id", None)
        
        # Initialize generation variables
        batch_size, seq_length = input_ids.shape
        max_length = seq_length + max_new_tokens
        has_reached_eos = jnp.zeros(batch_size, dtype=jnp.bool_)
        
        
        # Make sure attention mask and position IDs are properly set
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
            
        position_ids = jnp.cumsum(attention_mask, axis=-1) - 1
        
        # Create RNG key for sampling if not provided
        if key is None:
            key = jax.random.PRNGKey(0)
            
        # Store current inputs and generated tokens
        current_ids = input_ids
        all_ids = [input_ids]
        
        # Start generation loop
        for i in range(max_new_tokens):
            # Forward pass
            outputs, cache = self(
                input_ids=current_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=False,
                output_hidden_states=False,
                output_router_logits=False,
            )
            
            # Get logits for the next token (last position)
            next_token_logits = outputs.logits[:, -1, :]
                    
            # Add token to generated sequence
            next_token = jnp.argmax(next_token_logits, axis=-1)
            next_token = next_token[:, None]
            all_ids.append(next_token)
            current_ids = jnp.concatenate([current_ids, next_token], axis=1)
            # Update current_ids for next iteration
            
            # Update attention mask and position IDs
            attention_mask = jnp.ones_like(current_ids)
            
            position_ids = jnp.cumsum(attention_mask, axis=-1) - 1
            position_ids = position_ids[:, -1:]  # just need positions for new tokens
            
        # Concatenate all generated tokens
        return jnp.concatenate(all_ids, axis=1)
