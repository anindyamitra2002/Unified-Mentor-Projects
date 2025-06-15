# Animal Image Classification with PyTorch

## Objective
Build a system that can identify the animal in a given image using deep learning. The project explores neural networks and transfer learning for improved performance.

## Dataset
- **Source:** 15 folders, each representing a class (animal type)
- **Image Size:** 224 x 224 x 3
- **Classes:**
  - Bear
  - Bird
  - Cat
  - Cow
  - Deer
  - Dog
  - Dolphin
  - Elephant
  - Giraffe
  - Horse
  - Kangaroo
  - Lion
  - Panda
  - Tiger
  - Zebra

## Approach
- **Model:** ResNet50 (Transfer Learning)
- **Training:**
  - Initial training with frozen backbone
  - Fine-tuning last layers
- **Data Split:** 70% train, 15% validation, 15% test
- **Transforms:** Data augmentation for training, normalization for all

## Training & Validation Logs
| Epoch | Phase      | Loss   | Accuracy |
|-------|------------|--------|----------|
| 1     | Train      | 1.3757 | 0.7140   |
| 1     | Validation | 0.5237 | 0.9210   |
| 2     | Train      | 0.3980 | 0.9463   |
| 2     | Validation | 0.2762 | 0.9519   |
| 3     | Train      | 0.2564 | 0.9581   |
| 3     | Validation | 0.2091 | 0.9553   |
| 4     | Train      | 0.1877 | 0.9750   |
| 4     | Validation | 0.2096 | 0.9553   |
| 5     | Train      | 0.1405 | 0.9816   |
| 5     | Validation | 0.1882 | 0.9519   |
| 6     | Train      | 0.1198 | 0.9794   |
| 6     | Validation | 0.1673 | 0.9622   |
| 7     | Train      | 0.1053 | 0.9838   |
| 7     | Validation | 0.1326 | 0.9656   |
| 8     | Train      | 0.0828 | 0.9912   |
| 8     | Validation | 0.1335 | 0.9588   |
| 9     | Train      | 0.0845 | 0.9904   |
| 9     | Validation | 0.1399 | 0.9691   |
| 10    | Train      | 0.0770 | 0.9868   |
| 10    | Validation | 0.1380 | 0.9588   |
| 11    | Fine-tune  | 0.0572 | 0.9912   |
| 11    | Validation | 0.1113 | 0.9622   |
| 12    | Fine-tune  | 0.0270 | 0.9993   |
| 12    | Validation | 0.0974 | 0.9691   |
| 13    | Fine-tune  | 0.0199 | 0.9978   |
| 13    | Validation | 0.0906 | 0.9725   |
| 14    | Fine-tune  | 0.0129 | 1.0000   |
| 14    | Validation | 0.1023 | 0.9656   |
| 15    | Fine-tune  | 0.0077 | 1.0000   |
| 15    | Validation | 0.0933 | 0.9691   |

## Test Results
| Metric      | Value   |
|-------------|---------|
| Test Loss   | 0.0878  |
| Test Acc.   | 0.9761  |

## Conclusion
- The model achieved high accuracy on both validation and test sets, demonstrating effective transfer learning.
- Further improvements could include experimenting with different architectures, hyperparameters, or data augmentation techniques. 