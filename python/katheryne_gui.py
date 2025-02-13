import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
import subprocess
import sys
from pathlib import Path
from queue import Queue
import time

class KatheryneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Katheryne Assistant Manager")
        self.root.geometry("800x600")
        
        # Create message queue for thread-safe GUI updates
        self.msg_queue = Queue()
        
        # Setup the notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Create tabs
        self.setup_data_tab()
        self.setup_training_tab()
        self.setup_testing_tab()
        self.setup_deployment_tab()
        
        # Start message checking
        self.check_messages()

    def setup_data_tab(self):
        """Setup the data generation and management tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text='Data Management')
        
        # Data generation controls
        ttk.Label(data_frame, text="Training Data Generation", font=('Helvetica', 12, 'bold')).pack(pady=10)
        
        # Add character button
        ttk.Button(data_frame, text="Generate Training Data", 
                  command=self.generate_training_data).pack(pady=5)
        
        # Status display
        self.data_status = scrolledtext.ScrolledText(data_frame, height=10)
        self.data_status.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_training_tab(self):
        """Setup the model training tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text='Model Training')
        
        # Training controls
        ttk.Label(training_frame, text="Model Training Controls", 
                 font=('Helvetica', 12, 'bold')).pack(pady=10)
        
        # Training parameters
        params_frame = ttk.LabelFrame(training_frame, text="Training Parameters")
        params_frame.pack(fill='x', padx=5, pady=5)
        
        # Epochs
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=5)
        self.epochs_var = tk.StringVar(value="10")
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=0, column=2, padx=5, pady=5)
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=5)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Model type
        ttk.Label(params_frame, text="Model Type:").grid(row=1, column=2, padx=5, pady=5)
        self.model_type_var = tk.StringVar(value="light")
        ttk.Combobox(params_frame, textvariable=self.model_type_var, 
                    values=["light", "heavy"], width=10).grid(row=1, column=3, padx=5, pady=5)
        
        # Training controls
        controls_frame = ttk.Frame(training_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Start Training", 
                  command=self.start_training).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Stop Training", 
                  command=self.stop_training).pack(side='left', padx=5)
        
        # Training progress
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(training_frame, variable=self.progress_var, 
                                      maximum=100)
        self.progress.pack(fill='x', padx=5, pady=5)
        
        # Training log
        self.training_log = scrolledtext.ScrolledText(training_frame, height=10)
        self.training_log.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_testing_tab(self):
        """Setup the model testing tab"""
        testing_frame = ttk.Frame(self.notebook)
        self.notebook.add(testing_frame, text='Model Testing')
        
        # Query input
        ttk.Label(testing_frame, text="Enter your query:").pack(pady=5)
        self.query_input = ttk.Entry(testing_frame, width=50)
        self.query_input.pack(pady=5)
        
        # Test button
        ttk.Button(testing_frame, text="Test Query", 
                  command=self.test_query).pack(pady=5)
        
        # Response display
        ttk.Label(testing_frame, text="Response:").pack(pady=5)
        self.response_display = scrolledtext.ScrolledText(testing_frame, height=10)
        self.response_display.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_deployment_tab(self):
        """Setup the deployment tab"""
        deployment_frame = ttk.Frame(self.notebook)
        self.notebook.add(deployment_frame, text='Deployment')
        
        # Deployment controls
        ttk.Label(deployment_frame, text="Deployment Controls", 
                 font=('Helvetica', 12, 'bold')).pack(pady=10)
        
        # Server controls
        ttk.Button(deployment_frame, text="Start Server", 
                  command=self.start_server).pack(pady=5)
        ttk.Button(deployment_frame, text="Stop Server", 
                  command=self.stop_server).pack(pady=5)
        
        # Server status
        self.server_status = scrolledtext.ScrolledText(deployment_frame, height=10)
        self.server_status.pack(fill='both', expand=True, padx=5, pady=5)

    def generate_training_data(self):
        """Generate training data"""
        def run_generation():
            try:
                process = subprocess.Popen([sys.executable, 'python/generate_training_data.py'],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        self.msg_queue.put(('data', output.strip()))
                
                rc = process.poll()
                if rc == 0:
                    self.msg_queue.put(('data', 'Training data generation completed successfully!'))
                else:
                    self.msg_queue.put(('data', 'Error generating training data!'))
            except Exception as e:
                self.msg_queue.put(('data', f'Error: {str(e)}'))
        
        thread = threading.Thread(target=run_generation)
        thread.daemon = True
        thread.start()

    def start_training(self):
        """Start model training"""
        def run_training():
            try:
                cmd = [
                    sys.executable,
                    'python/train_assistant.py',
                    '--epochs', self.epochs_var.get(),
                    '--batch-size', self.batch_size_var.get(),
                    '--learning-rate', self.lr_var.get(),
                    '--model-type', self.model_type_var.get()
                ]
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, text=True)
                
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        self.msg_queue.put(('training', output.strip()))
                
                rc = process.poll()
                if rc == 0:
                    self.msg_queue.put(('training', 'Training completed successfully!'))
                else:
                    self.msg_queue.put(('training', 'Error during training!'))
            except Exception as e:
                self.msg_queue.put(('training', f'Error: {str(e)}'))
        
        thread = threading.Thread(target=run_training)
        thread.daemon = True
        thread.start()

    def stop_training(self):
        """Stop model training"""
        # Implement training stop logic
        pass

    def test_query(self):
        """Test a query against the model"""
        query = self.query_input.get()
        if not query:
            messagebox.showwarning("Warning", "Please enter a query!")
            return
        
        def run_test():
            try:
                # Call the model inference script
                cmd = [sys.executable, 'python/test_model.py', '--query', query]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, text=True)
                output, error = process.communicate()
                
                if process.returncode == 0:
                    try:
                        response = json.loads(output)
                        formatted_response = json.dumps(response, indent=2)
                        self.msg_queue.put(('response', formatted_response))
                    except json.JSONDecodeError:
                        self.msg_queue.put(('response', output))
                else:
                    self.msg_queue.put(('response', f'Error: {error}'))
            except Exception as e:
                self.msg_queue.put(('response', f'Error: {str(e)}'))
        
        thread = threading.Thread(target=run_test)
        thread.daemon = True
        thread.start()

    def start_server(self):
        """Start the API server"""
        def run_server():
            try:
                cmd = [sys.executable, 'python/api_server.py']
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, text=True)
                
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        self.msg_queue.put(('server', output.strip()))
            except Exception as e:
                self.msg_queue.put(('server', f'Error: {str(e)}'))
        
        thread = threading.Thread(target=run_server)
        thread.daemon = True
        thread.start()

    def stop_server(self):
        """Stop the API server"""
        # Implement server stop logic
        pass

    def check_messages(self):
        """Check for messages from worker threads"""
        try:
            while True:
                msg_type, message = self.msg_queue.get_nowait()
                
                if msg_type == 'data':
                    self.data_status.insert('end', message + '\n')
                    self.data_status.see('end')
                elif msg_type == 'training':
                    self.training_log.insert('end', message + '\n')
                    self.training_log.see('end')
                elif msg_type == 'response':
                    self.response_display.delete('1.0', 'end')
                    self.response_display.insert('end', message)
                elif msg_type == 'server':
                    self.server_status.insert('end', message + '\n')
                    self.server_status.see('end')
        except:
            pass
        finally:
            self.root.after(100, self.check_messages)

def main():
    root = tk.Tk()
    app = KatheryneGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()