import express from 'express';
import { spawn } from 'child_process';
import path from 'path';

const app = express();

// Allow JSON payloads for training configuration
app.use(express.json());

interface TrainingConfig {
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  modelType?: 'characters' | 'weapons' | 'artifacts';
}

/**
 * POST /train-ai
 * Triggers the AI training process with optional configuration
 */
app.post('/train-ai', (req, res) => {
  const config: TrainingConfig = req.body;
  console.log('Received AI training request with config:', config);

  // Convert config to environment variables for Python script
  const env = {
    ...process.env,
    EPOCHS: config.epochs?.toString(),
    BATCH_SIZE: config.batchSize?.toString(),
    LEARNING_RATE: config.learningRate?.toString(),
    MODEL_TYPE: config.modelType,
  };

  // Construct path to Python script
  const scriptPath = path.join(__dirname, '..', 'python', 'extend_training_data_v2.py');

  // Spawn Python process
  const pythonProcess = spawn('python', [scriptPath], { env });

  let outputData = '';
  let errorData = '';

  pythonProcess.stdout.on('data', (data) => {
    const output = data.toString();
    outputData += output;
    console.log(`Training output: ${output}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    const error = data.toString();
    errorData += error;
    console.error(`Training error: ${error}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`AI training process exited with code ${code}`);
    if (code === 0) {
      res.status(200).json({
        message: 'AI training completed successfully',
        output: outputData,
      });
    } else {
      res.status(500).json({
        message: 'AI training failed',
        code,
        error: errorData,
        output: outputData,
      });
    }
  });
});

/**
 * GET /models
 * Returns information about available trained models
 */
app.get('/models', (req, res) => {
  const modelsPath = path.join(__dirname, '..', 'data', 'models');
  
  // List all model files in the models directory
  const fs = require('fs');
  fs.readdir(modelsPath, (err: Error | null, files: string[]) => {
    if (err) {
      res.status(500).json({
        message: 'Failed to read models directory',
        error: err.message,
      });
      return;
    }

    const models = files
      .filter(file => file.endsWith('.pt'))
      .map(file => ({
        name: file,
        path: path.join(modelsPath, file),
        type: file.split('_')[0],
        lastModified: fs.statSync(path.join(modelsPath, file)).mtime,
      }));

    res.status(200).json(models);
  });
});

const PORT = process.env.PORT || 3000;

export const server = app.listen(PORT, () => {
  console.log(`Express server with AI training endpoint is running on port ${PORT}`);
});

export default app;