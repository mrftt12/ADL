// Initialize MongoDB database for AutoML Framework

// Switch to automl database
db = db.getSiblingDB('automl');

// Create collections with validation
db.createCollection('architectures', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['id', 'name', 'layers', 'connections'],
            properties: {
                id: { bsonType: 'string' },
                name: { bsonType: 'string' },
                layers: { bsonType: 'array' },
                connections: { bsonType: 'array' },
                input_shape: { bsonType: 'array' },
                output_shape: { bsonType: 'array' },
                parameter_count: { bsonType: 'int' },
                flops: { bsonType: 'int' },
                metadata: { bsonType: 'object' },
                created_at: { bsonType: 'date' },
                updated_at: { bsonType: 'date' }
            }
        }
    }
});

db.createCollection('training_logs', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['experiment_id', 'model_id', 'epoch', 'metrics'],
            properties: {
                experiment_id: { bsonType: 'string' },
                model_id: { bsonType: 'string' },
                epoch: { bsonType: 'int' },
                metrics: { bsonType: 'object' },
                timestamp: { bsonType: 'date' },
                gpu_utilization: { bsonType: 'object' },
                memory_usage: { bsonType: 'object' }
            }
        }
    }
});

db.createCollection('hyperparameter_trials', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['experiment_id', 'trial_id', 'parameters', 'objective_value'],
            properties: {
                experiment_id: { bsonType: 'string' },
                trial_id: { bsonType: 'string' },
                parameters: { bsonType: 'object' },
                objective_value: { bsonType: 'double' },
                status: { bsonType: 'string' },
                start_time: { bsonType: 'date' },
                end_time: { bsonType: 'date' },
                intermediate_values: { bsonType: 'array' }
            }
        }
    }
});

db.createCollection('nas_search_history', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['experiment_id', 'architecture_id', 'performance'],
            properties: {
                experiment_id: { bsonType: 'string' },
                architecture_id: { bsonType: 'string' },
                architecture: { bsonType: 'object' },
                performance: { bsonType: 'double' },
                training_time: { bsonType: 'double' },
                search_method: { bsonType: 'string' },
                generation: { bsonType: 'int' },
                timestamp: { bsonType: 'date' }
            }
        }
    }
});

// Create indexes for better performance
db.architectures.createIndex({ 'id': 1 }, { unique: true });
db.architectures.createIndex({ 'name': 1 });
db.architectures.createIndex({ 'parameter_count': 1 });
db.architectures.createIndex({ 'created_at': -1 });

db.training_logs.createIndex({ 'experiment_id': 1, 'epoch': 1 });
db.training_logs.createIndex({ 'model_id': 1 });
db.training_logs.createIndex({ 'timestamp': -1 });

db.hyperparameter_trials.createIndex({ 'experiment_id': 1 });
db.hyperparameter_trials.createIndex({ 'trial_id': 1 }, { unique: true });
db.hyperparameter_trials.createIndex({ 'objective_value': -1 });
db.hyperparameter_trials.createIndex({ 'start_time': -1 });

db.nas_search_history.createIndex({ 'experiment_id': 1 });
db.nas_search_history.createIndex({ 'architecture_id': 1 });
db.nas_search_history.createIndex({ 'performance': -1 });
db.nas_search_history.createIndex({ 'timestamp': -1 });

// Insert sample architecture templates
db.architectures.insertMany([
    {
        id: 'resnet18_template',
        name: 'ResNet-18 Template',
        layers: [
            { type: 'conv2d', filters: 64, kernel_size: [7, 7], stride: [2, 2] },
            { type: 'batch_norm' },
            { type: 'relu' },
            { type: 'max_pool', pool_size: [3, 3], stride: [2, 2] },
            { type: 'residual_block', filters: 64, blocks: 2 },
            { type: 'residual_block', filters: 128, blocks: 2, stride: [2, 2] },
            { type: 'residual_block', filters: 256, blocks: 2, stride: [2, 2] },
            { type: 'residual_block', filters: 512, blocks: 2, stride: [2, 2] },
            { type: 'global_avg_pool' },
            { type: 'dense', units: 1000 }
        ],
        connections: [],
        input_shape: [224, 224, 3],
        output_shape: [1000],
        parameter_count: 11689512,
        flops: 1814073344,
        metadata: {
            task_type: 'image_classification',
            framework: 'tensorflow',
            description: 'ResNet-18 architecture template for image classification'
        },
        created_at: new Date(),
        updated_at: new Date()
    },
    {
        id: 'lstm_template',
        name: 'LSTM Template',
        layers: [
            { type: 'embedding', vocab_size: 10000, embedding_dim: 128 },
            { type: 'lstm', units: 128, return_sequences: true },
            { type: 'dropout', rate: 0.2 },
            { type: 'lstm', units: 64 },
            { type: 'dropout', rate: 0.2 },
            { type: 'dense', units: 32, activation: 'relu' },
            { type: 'dense', units: 1, activation: 'sigmoid' }
        ],
        connections: [],
        input_shape: [null],
        output_shape: [1],
        parameter_count: 1234567,
        flops: 987654,
        metadata: {
            task_type: 'text_classification',
            framework: 'tensorflow',
            description: 'LSTM architecture template for text classification'
        },
        created_at: new Date(),
        updated_at: new Date()
    }
]);

print('MongoDB initialization completed successfully');