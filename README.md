# Viscosity Analysis using composite model

There seem to be two regimes for the data. Trying to see if a linear model can be used in the physics dominated domain, and then use a neural network (MLP) for the non-linear/complex regime. Several iterations have led to a soft-gating mechanism described next.

#### Soft-Gated Mixture Model

Instead of a hard binary switch (Linear vs. Complex), we can train a model to predict the **expected error (residual)** of the physics model for any given experimental condition. We then use this predicted error as a "Confidence Score" to smoothly blend the two models.

This approach works as follows:

1. **Physics Model:** Calculate $y_{phy} = K_w \cdot x$ (using RANSAC to find $K_w$).
2. **Error Model:** Train a Random Forest to predict the magnitude of the residual i.e. $\epsilon = |y_{meas} - y_{phy}|$ based on Temperature, RPM, etc. This creates a map of "Where does the physics model fail?", hopefully.
3. **Complex Model:** Train an MLP on the full dataset to capture the non-linear behavior.
4. **Soft Blending:** The final prediction is a weighted average:

$y_{pred} = w\cdot y_{phy} + (1-w)\cdot y_{mlp}$


Where $w$ is high (near 1) when the predicted error is low, and  drops to 0 when the predicted error is high.

**Weights of the soft-gating**: These weights are obtained using a Gaussian radial basis function, that acts as a confidence score for the linear physics model. The specific formulation is:

$w = \exp\left(-\frac{\epsilon}^2{2\cdot (3\sigma)^2}\right)$

Where, $\sigma$ is the standard deviation of the residuals for the _inliers_ identified by RANSAC.
