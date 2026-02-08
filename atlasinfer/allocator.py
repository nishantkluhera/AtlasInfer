"""
AtlasInfer Precision Allocator - Budget-aware precision assignment

Given sensitivity scores and a memory budget, this module determines the
optimal precision level (FP16/FP8/FP4) for each layer to maximize accuracy
within the memory constraint.
"""
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PrecisionLevel(Enum):
    """Available precision levels, ordered from highest to lowest."""
    FP16 = "fp16"
    FP8 = "fp8"
    FP4 = "fp4"
    
    @property
    def bytes_per_param(self) -> float:
        """Approximate bytes per parameter for this precision."""
        return {
            PrecisionLevel.FP16: 2.0,
            PrecisionLevel.FP8: 1.0,
            PrecisionLevel.FP4: 0.5,
        }[self]


@dataclass
class LayerAllocation:
    """Allocation decision for a single layer."""
    name: str
    precision: PrecisionLevel
    sensitivity: float
    param_count: int
    memory_bytes: int


@dataclass
class AllocationResult:
    """Complete allocation result for a model."""
    allocations: Dict[str, PrecisionLevel]
    total_memory_bytes: int
    memory_budget_bytes: int
    layers_fp16: int
    layers_fp8: int
    layers_fp4: int
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Allocation: FP16={self.layers_fp16}, FP8={self.layers_fp8}, FP4={self.layers_fp4} | "
            f"Memory: {self.total_memory_bytes / (1024**3):.2f}/{self.memory_budget_bytes / (1024**3):.2f} GB"
        )


class PrecisionAllocator:
    """
    Allocates precision levels to layers based on sensitivity and memory budget.
    
    This is the DECISION ENGINE of LADQ:
    1. Sort layers by sensitivity (most sensitive first)
    2. Greedily assign highest precision to most sensitive layers
    3. Downgrade precision when budget is exhausted
    """
    
    def __init__(
        self,
        precision_levels: Optional[List[PrecisionLevel]] = None
    ):
        """
        Args:
            precision_levels: Available precision levels (default: FP16, FP8, FP4)
        """
        if precision_levels is None:
            precision_levels = [PrecisionLevel.FP16, PrecisionLevel.FP8, PrecisionLevel.FP4]
        
        # Sort by quality (FP16 > FP8 > FP4)
        self.precision_levels = sorted(
            precision_levels,
            key=lambda p: p.bytes_per_param,
            reverse=True  # Highest precision first
        )
    
    def allocate(
        self,
        sensitivities: Dict[str, float],
        layer_sizes: Dict[str, int],
        memory_budget_bytes: int,
        min_precision: PrecisionLevel = PrecisionLevel.FP4
    ) -> AllocationResult:
        """
        Allocate precision levels to layers within memory budget.
        
        Algorithm (Greedy Sensitivity-First):
        1. Start with all layers at minimum precision
        2. Sort layers by sensitivity (descending)
        3. For each layer, try to upgrade precision if budget allows
        
        Args:
            sensitivities: Layer sensitivity scores {name: score}
            layer_sizes: Parameter counts {name: num_params}
            memory_budget_bytes: Maximum memory in bytes
            min_precision: Minimum (lowest quality) precision level
            
        Returns:
            AllocationResult with per-layer precision assignments
        """
        # Initialize all at minimum precision
        allocations = {name: min_precision for name in sensitivities}
        
        # Calculate initial memory
        def calc_memory(allocs: Dict[str, PrecisionLevel]) -> int:
            return sum(
                int(layer_sizes.get(name, 0) * allocs[name].bytes_per_param)
                for name in allocs
            )
        
        # Sort by sensitivity (most sensitive first)
        sorted_layers = sorted(
            sensitivities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Try to upgrade each layer in order of sensitivity
        for name, _ in sorted_layers:
            if name not in layer_sizes:
                continue
            
            current_precision = allocations[name]
            current_idx = self.precision_levels.index(current_precision)
            
            # Try upgrading to higher precision levels
            for better_idx in range(current_idx - 1, -1, -1):
                better_precision = self.precision_levels[better_idx]
                
                # Calculate memory if we upgrade this layer
                test_allocs = allocations.copy()
                test_allocs[name] = better_precision
                new_memory = calc_memory(test_allocs)
                
                if new_memory <= memory_budget_bytes:
                    allocations[name] = better_precision
                else:
                    break  # Can't afford this precision, stop trying
        
        # Build result
        total_memory = calc_memory(allocations)
        fp16_count = sum(1 for p in allocations.values() if p == PrecisionLevel.FP16)
        fp8_count = sum(1 for p in allocations.values() if p == PrecisionLevel.FP8)
        fp4_count = sum(1 for p in allocations.values() if p == PrecisionLevel.FP4)
        
        return AllocationResult(
            allocations=allocations,
            total_memory_bytes=total_memory,
            memory_budget_bytes=memory_budget_bytes,
            layers_fp16=fp16_count,
            layers_fp8=fp8_count,
            layers_fp4=fp4_count
        )
    
    def allocate_uniform(
        self,
        layer_names: List[str],
        precision: PrecisionLevel = PrecisionLevel.FP8
    ) -> Dict[str, PrecisionLevel]:
        """
        Simple uniform allocation (all layers same precision).
        
        Useful as a baseline comparison.
        """
        return {name: precision for name in layer_names}


def get_layer_sizes(model: torch.nn.Module) -> Dict[str, int]:
    """
    Extract parameter counts for all linear layers.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary mapping layer names to parameter counts
    """
    sizes = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            sizes[name] = module.weight.numel()
            if module.bias is not None:
                sizes[name] += module.bias.numel()
    return sizes


def estimate_memory_usage(
    layer_sizes: Dict[str, int],
    precision: PrecisionLevel
) -> int:
    """Estimate memory for uniform precision allocation."""
    return int(sum(layer_sizes.values()) * precision.bytes_per_param)


def print_allocation_report(
    result: AllocationResult,
    sensitivities: Dict[str, float],
    top_n: int = 15
):
    """Print detailed allocation report."""
    print("\n" + "=" * 70)
    print("LADQ Precision Allocation Report")
    print("=" * 70)
    
    # Summary
    print(f"\nBudget: {result.memory_budget_bytes / (1024**3):.2f} GB")
    print(f"Allocated: {result.total_memory_bytes / (1024**3):.2f} GB "
          f"({100 * result.total_memory_bytes / result.memory_budget_bytes:.1f}% utilized)")
    print(f"\nPrecision Distribution:")
    print(f"  FP16 (high precision): {result.layers_fp16} layers")
    print(f"  FP8 (medium precision): {result.layers_fp8} layers")
    print(f"  FP4 (low precision): {result.layers_fp4} layers")
    
    # Top layers
    sorted_by_sens = sorted(
        [(n, result.allocations.get(n, PrecisionLevel.FP8), s) 
         for n, s in sensitivities.items()],
        key=lambda x: x[2],
        reverse=True
    )
    
    print(f"\n{'Layer':<45} {'Precision':>10} {'Sensitivity':>12}")
    print("-" * 70)
    
    for name, precision, sens in sorted_by_sens[:top_n]:
        display_name = name if len(name) <= 43 else "..." + name[-40:]
        print(f"{display_name:<45} {precision.value:>10} {sens:>12.6f}")
    
    if len(sorted_by_sens) > top_n:
        print(f"... and {len(sorted_by_sens) - top_n} more layers")
    
    print("=" * 70 + "\n")
