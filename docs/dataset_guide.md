# Dataset Guide for Theta AI

This guide explains how datasets are structured, how to work with existing datasets, and how to create new training data for Theta AI.

## Dataset Structure

Theta AI datasets are stored in the `Datasets/` directory as JSON files organized by topic:

- `hardware_knowledge.json` - Computer hardware information
- `it_support.json` - IT support and troubleshooting
- `advanced_programming.json` - Programming concepts and techniques
- `advanced_cybersecurity.json` - Cybersecurity topics and best practices
- `advanced_cloud.json` - Cloud computing knowledge
- `advanced_technical.json` - General technical information

## JSON Format

Each dataset follows a consistent question-answer format:

```json
[
  {
    "question": "What is the difference between AMD and NVIDIA GPUs?",
    "answer": "AMD and NVIDIA are the two major GPU manufacturers with different architectures and features..."
  },
  {
    "question": "How do I overclock my CPU?",
    "answer": "Overclocking a CPU involves: 1) Ensure adequate cooling..."
  }
]
```

## Working with Existing Datasets

To use the existing datasets:

1. The datasets are automatically loaded by the training and interface scripts
2. You can view and explore datasets using standard JSON tools or the `data_processor.py` utility

## Creating New Training Data

To create new training data:

1. Create a new JSON file in the `Datasets/` directory with a descriptive name (e.g., `network_security.json`)
2. Follow the same question-answer format as existing datasets
3. Ensure high-quality, accurate, and comprehensive answers
4. Include diverse question formulations for the same topics

### Data Quality Guidelines

- **Accuracy**: All information must be factually correct and up-to-date
- **Completeness**: Answers should be comprehensive and cover the topic thoroughly
- **Clarity**: Use clear and concise language
- **Structure**: Format complex answers with numbered points or bullet lists where appropriate
- **Consistency**: Maintain a consistent tone and level of detail across answers

## Data Processing Pipeline

After creating new datasets:

1. Run the data processing script to validate and prepare the data:

   ```bash
   python src/process_data.py
   ```

2. This script will:
   - Validate the JSON format
   - Check for duplicate questions
   - Perform basic quality checks
   - Prepare the data for training

## Advanced: Data Augmentation

To improve model performance, you can augment the training data:

1. Create variations of existing questions using different phrasing
2. Include common misspellings or informal versions of technical terms
3. Add context-specific variations for the same information

## Dataset Versioning

When making significant changes to datasets:

1. Document the changes in the CHANGELOG.md file
2. Include a version identifier in dataset filenames for major revisions (e.g., `cybersecurity_v2.json`)
3. Consider archiving old versions in an `archive/` subdirectory

## Best Practices for Dataset Management

- Regularly review and update information to ensure it remains accurate
- Balance the dataset to cover all relevant topics equally
- Test new datasets with the model before deploying to production
- Maintain separate development and production datasets when making large changes
