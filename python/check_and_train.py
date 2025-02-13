import sys
import subprocess
import pkg_resources
import time

def check_dependencies():
    """Check if all required packages are installed."""
    required = {'torch', 'tqdm'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if missing:
        print(f'Missing packages: {missing}')
        print('Installing missing packages...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
        print('All packages installed successfully!')
    else:
        print('All required packages are installed!')

def run_with_timeout(cmd, timeout_minutes=3):
    """Run command with timeout in minutes."""
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    while process.poll() is None:
        # Read output line by line
        for line in process.stdout:
            print(line, end='')
            
        # Check if we've exceeded timeout
        if (time.time() - start_time) > (timeout_minutes * 60):
            process.terminate()
            print(f"\nProcess terminated after {timeout_minutes} minutes")
            return False
            
        time.sleep(0.1)
    
    return process.returncode == 0

def main():
    """Main function to check deps and start training."""
    print("Checking dependencies...")
    check_dependencies()
    
    print("\nStarting training (3 minute timeout)...")
    success = run_with_timeout([sys.executable, 'python/train_assistant.py'])
    
    if success:
        print("\nTraining completed successfully!")
    else:
        print("\nTraining was interrupted or failed")

if __name__ == "__main__":
    main()