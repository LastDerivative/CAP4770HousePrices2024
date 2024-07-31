import subprocess
import argparse
import logging

def run_script(script_name, *args):
    try:
        subprocess.run(['python', script_name] + list(args), check=True)
        logging.info(f"Successfully ran {script_name} with arguments {args}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_name}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Data Processing and Model Tuning Pipeline')
    parser.add_argument('--comparison_file', type=str, default='sample_submission.csv', help='Path to CSV file used for logrmse calculation')
    parser.add_argument('--run_gb', action='store_true', help='Run GB model tuning script')
    parser.add_argument('--run_nn', action='store_true', help='Run NN model tuning script')
    parser.add_argument('--run_stacked', action='store_true', help='Run Stacked model tuning script')
    parser.add_argument('--run_all', action='store_true', help='Run all models')
    parser.add_argument('--evaluate_stacked', action='store_true', help='Evaluate Stacked model after tuning')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    

    # Run preprocessing script
    run_script('AdditonalPreProcessingAndCleaning.py','train.csv', 'test.csv', 'train_with_features.csv', 'test_with_features.csv')
    
    # Determine which scripts to run
    run_gb = args.run_gb
    run_nn = args.run_nn
    run_stacked = args.run_stacked
    run_all = args.run_all
    evaluate_stacked = args.evaluate_stacked

    # If no specific script is mentioned, run only gb and nn
    if not (run_gb or run_nn or run_stacked):
        run_gb = run_nn = True
        run_stacked = False

    # Run specified scripts
    if run_gb or run_all:    
        # Run model tuning script
        run_script('GB_Approach.py', 'train_with_features.csv', 'test_with_features.csv', 'predictions_GB.csv', args.comparison_file)

    if run_nn or run_all:
        # Run NN model tuning script
        run_script('NN_Approach.py', 'train_with_features.csv', 'test_with_features.csv', 'predictions_NN.csv', args.comparison_file)

    if run_stacked or run_all:
        # Run Stacked model tuning script
        if(evaluate_stacked):#if no specified, will run without it.
            evaluate_stacked = True
        else:
            evaluate_stacked = False

        run_script('Stacked_Approach.py', 'train_with_features.csv', 'test_with_features.csv', 'predictions_Stacked.csv', args.comparison_file, str(evaluate_stacked))

if __name__ == '__main__':
    main()
