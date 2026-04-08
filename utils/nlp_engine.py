import random

def generate_perceptron_insight(epochs, acc, loss, converged):
    if converged:
        templates = [
            f"The neural unit reached stable convergence at epoch {epochs}. Feature space is now linearly separable with an accuracy ceiling of {acc:.2f}.",
            f"Training halted. The decision boundary perfectly dissects the coordinate plane (Loss: {loss:.4f}). Optimization complete.",
            f"Convergence verified. The perceptron achieved {acc*100:.0f}% accuracy in {epochs} iterations, successfully learning the logic gate."
        ]
    else:
        templates = [
            f"Training incomplete at epoch {epochs}. The network is struggling to map the inputs smoothly, plateauing at {acc:.2f} accuracy.",
            f"The logical threshold is resisting convergence. High chaotic variance detected (Loss: {loss:.4f}).",
            f"Input space remains entangled. The single-layer architecture cannot resolve this non-linear dependency (Acc: {acc:.2f})."
        ]
    return random.choice(templates)

def generate_fwd_insight(activation, loss_type, final_loss):
    templates = [
        f"Forward pass complete. Signals routed through '{activation}' gates yielded a continuous spatial gradient with {loss_type} error of {final_loss:.4f}.",
        f"Matrix multiplication resolved. The {activation} function successfully introduced non-linearity, compressing error variance to {final_loss:.4f}.",
        f"Signal propagation stabilized. The feed-forward network mapped the input tensors into a solvable latent space ({loss_type}: {final_loss:.4f})."
    ]
    return random.choice(templates)

def generate_bwd_insight(epoch, total_epochs, loss, mean_grad):
    templates = [
        f"Backpropagation reached epoch {epoch} of {total_epochs}. The loss is now {loss:.4f}, while the average gradient magnitude is {mean_grad:.4f}.",
        f"The chain rule is actively sending learning signals backward through the network. Residual loss sits at {loss:.4f}, with gradients averaging {mean_grad:.4f}.",
        f"Gradient descent is still shaping the weight landscape. By epoch {epoch}, the model has reduced the error to {loss:.4f} with an average update strength of {mean_grad:.4f}."
    ]
    return random.choice(templates)

def generate_cv_insight(module, mode=""):
    if module == "attendance":
        return "Optical sweep protocol armed. System is actively searching for known biometric facial signatures to log into spatial memory."
    elif module == "vehicle":
        return "Traffic density grid active. Continuous temporal monitoring of physical vehicle entities within the road trajectory sector."
    elif module == "face":
        if mode == "emotion":
            return "Subject engagement active. Capturing high-resolution coordinate micro-expressions to classify underlying human sentiment."
        return "Facial landmark mesh initialized. Extracting precise geometric orientation across 468 focal points in real-time."
    elif module == "sign":
        return "Color-shape isolation matrix active. Neural filters are parsing environmental contours for road signage patterns."
    elif module == "palm":
        return "Kinematic skeletal tracking active. Analyzing hand pivot joints to determine gesture classifications and mechanical positioning."
    return "Optical sensor processing at optimal latency. Spatial parsing active."
