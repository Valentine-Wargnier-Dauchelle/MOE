# Explainable Monotonic Networks and Constrained Learning for Interpretable Classification and Weakly Supervised Anomaly Detection

*Please cite : Wargnier-Dauchelle, Valentine, et al. "Explainable Monotonic Networks and Constrained Learning for Interpretable Classification and Weakly Supervised Anomaly Detection." Submitted to Pattern Recognition.*
 
**Abstract**: Deep networks interpretability is fundamental in critical domains like medicine: using easily explainable networks with decisions based on radiological signs and not on spurious confounders would reassure the clinicians. Confidence is reinforced by the integration of intrinsic properties and characteristics of monotonic networks could be used to design such intrinsically explainable networks. As they are considered as too constrained and difficult to train, they are often very shallow and rarely used for image applications. In this work, we propose a procedure to transform any architecture into a trainable monotonic network, identifying the critical importance of weights initialization, and highlight the interest of such networks for explicability and interpretability. By constraining the features and the gradients of a healthy vs pathological images classifier, we show, using counterfactual examples, that the network decision is more based on the radiological signs of the pathology and outperforms state-of-the-art methods for weakly supervised anomaly detection.

# User guidelines 

**Create discriminator parser**
```
import argparse
import architecture

parser = argparse.ArgumentParser(description='Parser')
architecture.createDiscParser(parser)
```

**Create MoE parser**
```
import moe

moe.MoE.add_argument_to(parser)
```

**Create model**
```
args = parser.parse_args()
model = architecture.createDisc(args)
```

**Weights init**
```
import init

init.initByWeightRescaling(model, img_size, verbose=verbose) #img_size=size of random noise used for init, verbose=print mean, var, corr
```

**Counterfactual example**
```
from counterfactual import activation_opti

alpha = activation_opti(model, features, mono=True) #model=trained model, features=initial interpretable features for which we search the counterfactual difference, mono=impose non-negative alpha
```

**Examples of args**

--moe=1/CN/8/1/c8_7/c8_3_1_inorm : monotonic networks with encoder Conv kernel=7, filters=8, instance norm +  Conv kernel=3, filters=8, instance norm

--moe=none/CN/8/1/c8_7/c8_3_1_inorm : same but non-monotonic

--moe=0/CN/8/1/c8_7/c8_3_1_inorm : non-monotonic + no encoder

--moeKLfeat=50/1/KL : KL loss with bins=50 and loss weight=1

