#!/usr/bin/env python
"""
Run script for TFT model training and forecasting with improved parameters
"""

import argparse
import os
import json
from config import get_config_by_segment, DEFAULT_CONFIG
import importlib.util
import sys

def load_module_from_file(file_path, module_name):
    """Load a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run TFT model training and forecasting with improved parameters')
    
    # Data arguments
    parser.add_argument('--data-source', choices=['synthetic', 'real'], default='synthetic',
                        help='Data source: synthetic or real')
    parser.add_argument('--data-file', type=str, help='Path to real data file (CSV)')
    parser.add_argument('--start-date', type=str, default='2022-01-01',
                        help='Start date for synthetic data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                        help='End date for synthetic data (YYYY-MM-DD)')
    parser.add_argument('--num-skus', type=int, default=5,
                        help='Number of SKUs for synthetic data')
    
    # Model arguments
    parser.add_argument('--segment', choices=['year_round', 'seasonal', 'semi_seasonal', 'new_sku', 'default'],
                        default='default', help='SKU segment for configuration')
    parser.add_argument('--encoder-length', type=int, help='Maximum encoder length (lookback)')
    parser.add_argument('--prediction-length', type=int, help='Maximum prediction length (forecast horizon)')
    parser.add_argument('--hidden-size', type=int, help='Hidden size for TFT model')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Maximum number of epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate for optimizer')
    parser.add_argument('--early-stopping', type=int, help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Base output directory')
    parser.add_argument('--config-file', type=str, help='Path to custom configuration JSON file')
    
    # Additional arguments for improved features
    parser.add_argument('--use-cross-effects', action='store_true', 
                        help='Enable cross-product effects analysis')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating plots to save time')
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """Update configuration from command line arguments"""
    # Data configuration
    if args.data_source:
        config['data']['synthetic'] = (args.data_source == 'synthetic')
    if args.data_file:
        config['data']['file_path'] = args.data_file
    if args.start_date:
        config['data']['start_date'] = args.start_date
    if args.end_date:
        config['data']['end_date'] = args.end_date
    if args.num_skus:
        config['data']['num_skus'] = args.num_skus
    
    # Model configuration
    if args.encoder_length:
        config['model']['max_encoder_length'] = args.encoder_length
    if args.prediction_length:
        config['model']['max_prediction_length'] = args.prediction_length
    if args.hidden_size:
        config['model']['hidden_size'] = args.hidden_size
    
    # Training configuration
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.learning_rate:
        config['model']['learning_rate'] = args.learning_rate
    if args.early_stopping:
        config['training']['early_stopping_patience'] = args.early_stopping
    
    # Output configuration
    if args.output_dir:
        base_dir = args.output_dir
        config['output']['forecasts_dir'] = os.path.join(base_dir, 'forecasts')
        config['output']['plots_dir'] = os.path.join(base_dir, 'plots')
        config['output']['models_dir'] = os.path.join(base_dir, 'models')
    
    return config

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Get base configuration from segment
    config = get_config_by_segment(args.segment)
    
    # If custom config file is provided, load it
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                custom_config = json.load(f)
                
            # Update config with custom values
            config['data'].update(custom_config.get('data', {}))
            config['model'].update(custom_config.get('model', {}))
            config['training'].update(custom_config.get('training', {}))
            config['output'].update(custom_config.get('output', {}))
            
            print(f"Loaded custom configuration from {args.config_file}")
        except Exception as e:
            print(f"Error loading custom configuration: {e}")
    
    # Update configuration from command line arguments
    config = update_config_from_args(config, args)
    
    # Print configuration
    print("Running with configuration:")
    print(json.dumps(config, indent=2))
    
    # Import main module
    try:
        # First try to import as a regular module
        from main import main as run_main
    except ImportError:
        # If that fails, try to load it from a file
        main_module = load_module_from_file("main.py", "main")
        run_main = main_module.main
    
    # Run main function with configuration
    run_main(config)

if __name__ == "__main__":
    main()