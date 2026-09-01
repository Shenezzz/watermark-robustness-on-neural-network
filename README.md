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


Attack: Aeepeclipse Advanced
  watermark_retention:      0.000  (1.0 = fully intact, ~0.5 = destroyed)
  task_accuracy_retention:  0.105  (attacked_acc=0.1000 / baseline_acc=0.9479)