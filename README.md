

ResNet18-Cifar

Best test accuracy on normal trained model: 0.9479

Test accuracy aftere embedding uchidas watermark and bit accuracy :
    test_acc: 0.9258  (baseline was 0.9479 -- accuracy_retention so far = 0.9767)
    watermark_bit_acc: 1.000

Attack: Quanitzation
    watermark_retention:      1.000  (1.0 = fully intact, ~0.5 = destroyed)
    task_accuracy_retention:  0.977  (attacked_acc=0.9260 / baseline_acc=0.9479)

Attack: Weight Noise
    watermark_retention:      0.000  (1.0 = fully intact, ~0.5 = destroyed)
    task_accuracy_retention:  0.977  (attacked_acc=0.9258 / baseline_acc=0.9479)

Attack: Deepeclipse Base
  watermark_retention:      0.000  (1.0 = fully intact, ~0.5 = destroyed)
  task_accuracy_retention:  0.977  (attacked_acc=0.9258 / baseline_acc=0.9479)

Attack: Deepeclipse Advanced
 watermark_retention:      0.500  (1.0 = fully intact, ~0.5 = destroyed)
  task_accuracy_retention:  0.968  (attacked_acc=0.9172 / baseline_acc=0.9479)

Attack: Pruning
    sparaity : 50
        watermark_retention:      1.000  (1.0 = fully intact, ~0.5 = destroyed)
        task_accuracy_retention:  0.282  (attacked_acc=0.2669 / baseline_acc=0.9479)

    sparaity : 99.5
        watermark_retention:      0.938  (1.0 = fully intact, ~0.5 = destroyed)
        task_accuracy_retention:  0.105  (attacked_acc=0.1000 / baseline_acc=0.9479)

Attack: Fine Tune
    FTAL:
        watermark_retention:      1.000  (1.0 = fully intact, ~0.5 = destroyed)
        task_accuracy_retention:  0.996  (attacked_acc=0.9443 / baseline_acc=0.9479)
    RTAL:
        watermark_retention:      1.000  (1.0 = fully intact, ~0.5 = destroyed)
        task_accuracy_retention:  0.981  (attacked_acc=0.9297 / baseline_acc=0.9479)

LeNet5 MNIST

Attack: quantization
  watermark_retention:      1.000  (1.0 = fully intact, ~0.5 = destroyed)
  task_accuracy_retention:  0.998  (attacked_acc=0.9896 / baseline_acc=0.9915)

Attack: weight_noise
  watermark_retention:      1.000  (1.0 = fully intact, ~0.5 = destroyed)
  task_accuracy_retention:  0.998  (attacked_acc=0.9894 / baseline_acc=0.9915)

Attack: deepeclipse
  watermark_retention:      0.500  (1.0 = fully intact, ~0.5 = destroyed)
  task_accuracy_retention:  0.998  (attacked_acc=0.9893 / baseline_acc=0.9915)

Attack: fine_tune
  watermark_retention:      1.000  (1.0 = fully intact, ~0.5 = destroyed)
  task_accuracy_retention:  0.999  (attacked_acc=0.9907 / baseline_acc=0.9915)
  
Attack: pruning (0.9)
  watermark_retention:      0.531  (1.0 = fully intact, ~0.5 = destroyed)
  task_accuracy_retention:  0.998  (attacked_acc=0.9894 / baseline_acc=0.9915)
