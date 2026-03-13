"""
Grid Search for Removal Experiment
===================================
Runs a systematic grid search over layers, filters, and n_pass values.
"""

import os
import sys
import time
from datetime import datetime

from run_experiment import run_single_experiment, get_experiment_dir


def run_grid_search(
    layer_range=(2, 4),      # min, max layers (inclusive)
    filter_range=(6, 10),    # min, max filters (inclusive)
    n_pass_reduction=5       # test n_pass from filters down to filters-reduction
):
    """
    Run a grid search over the specified parameter ranges.
    
    Args:
        layer_range: (min_layers, max_layers) inclusive
        filter_range: (min_filters, max_filters) inclusive
        n_pass_reduction: How many steps below max to test
                         (e.g., 5 means test filters, filters-1, ..., filters-5)
    """
    # Generate all configurations
    configs = []
    
    for num_layers in range(layer_range[0], layer_range[1] + 1):
        for num_filters in range(filter_range[0], filter_range[1] + 1):
            # Test n_pass from all (num_filters) down to (num_filters - n_pass_reduction)
            for reduction in range(n_pass_reduction + 1):
                n_pass = num_filters - reduction
                if n_pass >= 1:  # Must pass at least 1
                    configs.append((num_layers, num_filters, n_pass))
    
    total = len(configs)
    print("\n" + "=" * 70)
    print("GRID SEARCH - Removal Experiment")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nParameter ranges:")
    print(f"  Layers: {layer_range[0]} to {layer_range[1]}")
    print(f"  Filters: {filter_range[0]} to {filter_range[1]}")
    print(f"  N_pass reduction: 0 to {n_pass_reduction}")
    print(f"\nTotal configurations: {total}")
    print(f"Estimated time: {total * 2} - {total * 3} minutes")
    print("=" * 70 + "\n")
    
    results = []
    start_time = time.time()
    
    for i, (num_layers, num_filters, n_pass) in enumerate(configs):
        config_start = time.time()
        
        # Check if already exists
        exp_dir = get_experiment_dir(num_layers, num_filters, n_pass)
        model_path = os.path.join(exp_dir, "model.npz")
        
        print(f"\n[{i+1}/{total}] {num_layers}L, {num_filters}F, n_pass={n_pass}")
        
        if os.path.exists(model_path):
            print(f"  → Already exists, skipping...")
            results.append({
                'config': (num_layers, num_filters, n_pass),
                'status': 'skipped',
                'time': 0
            })
            continue
        
        try:
            result = run_single_experiment(
                num_layers, num_filters, n_pass,
                force_retrain=False
            )
            
            config_time = time.time() - config_start
            
            if result:
                results.append({
                    'config': (num_layers, num_filters, n_pass),
                    'accuracy': result['overall_accuracy'],
                    'status': 'success',
                    'time': config_time
                })
                print(f"  → Accuracy: {result['overall_accuracy']*100:.2f}% (took {config_time:.1f}s)")
            else:
                results.append({
                    'config': (num_layers, num_filters, n_pass),
                    'status': 'cancelled',
                    'time': config_time
                })
                
        except Exception as e:
            print(f"  → ERROR: {e}")
            results.append({
                'config': (num_layers, num_filters, n_pass),
                'status': 'error',
                'error': str(e),
                'time': time.time() - config_start
            })
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nResults summary:")
    
    successful = [r for r in results if r.get('status') == 'success']
    if successful:
        # Sort by accuracy
        successful.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
        
        print(f"\nTop 10 configurations:")
        print("-" * 50)
        for r in successful[:10]:
            layers, filters, n_pass = r['config']
            acc = r['accuracy'] * 100
            print(f"  {layers}L, {filters}F, n_pass={n_pass}: {acc:.2f}%")
        
        print(f"\nBottom 5 configurations:")
        print("-" * 50)
        for r in successful[-5:]:
            layers, filters, n_pass = r['config']
            acc = r['accuracy'] * 100
            print(f"  {layers}L, {filters}F, n_pass={n_pass}: {acc:.2f}%")
    
    # Save results to file
    results_path = os.path.join(os.path.dirname(__file__), "grid_search_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Grid Search Results\n")
        f.write(f"==================\n")
        f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n\n")
        
        f.write(f"Layers, Filters, N_Pass, Accuracy, Status\n")
        f.write("-" * 50 + "\n")
        for r in results:
            layers, filters, n_pass = r['config']
            status = r.get('status', 'unknown')
            acc = r.get('accuracy', 0) * 100 if 'accuracy' in r else 'N/A'
            if isinstance(acc, float):
                f.write(f"{layers}, {filters}, {n_pass}, {acc:.2f}%, {status}\n")
            else:
                f.write(f"{layers}, {filters}, {n_pass}, {acc}, {status}\n")
    
    print(f"\nResults saved to: {results_path}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Default grid search parameters
    run_grid_search(
        layer_range=(2, 4),
        filter_range=(6, 10),
        n_pass_reduction=5
    )
