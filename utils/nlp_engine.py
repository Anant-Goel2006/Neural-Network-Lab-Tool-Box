def generate_insight(module_name, metrics):
    """
    NLP Insight Generator
    Takes module metrics and generates a human-readable heuristic insight.
    Returns: (text, color, icon)
    """
    if module_name == "perceptron":
        conv = metrics.get("converged", -1)
        loss = metrics.get("loss", 0)
        acc = metrics.get("accuracy", 0)
        
        if conv > 0:
            return (f"The model successfully recognized the linear pattern and converged at epoch {conv}. "
                    f"A clear decision boundary perfectly separates the data.", 
                    "#10B981", "✨")
        elif acc >= 75:
            return (f"The model struggled to fully converge, finishing with a loss of {loss:.2f}. "
                    f"It reached {acc:.1f}% accuracy, signifying that the dataset might not be completely linearly separable.", 
                    "#F59E0B", "⚖️")
        else:
            return (f"The perceptron failed to find a linear boundary, ending with an accuracy of just {acc:.1f}%. "
                    f"Consider using a Multi-Layer Perceptron (MLP) for this non-linear dataset.", 
                    "#EF4444", "🛑")

    elif module_name == "forward":
        loss = metrics.get("loss", 0.0)
        pred = metrics.get("pred", 0.0)
        target = metrics.get("target", 0.0)
        act = metrics.get("activation", "ReLU")
        diff = abs(pred - target)
        
        if diff < 0.1:
            return (f"Excellent forward pass. The signal propagated through the {act} activations "
                    f"and output a prediction ({pred:.3f}) very close to your target ({target:.3f}).", 
                    "#10B981", "🎯")
        elif diff < 0.5:
            return (f"The network generated an output of {pred:.3f}, giving a moderate loss of {loss:.3f}. "
                    f"Tweaking the weights or changing the {act} activation function may improve accuracy.", 
                    "#06B6D4", "🔍")
        else:
            return (f"High loss detected ({loss:.3f}). The signal transformation via {act} "
                    f"is resulting in a prediction far from the target. Consider running backpropagation to adjust the weights.", 
                    "#F59E0B", "⚠️")

    elif module_name == "backward":
        grad_max = metrics.get("grad_max", 0.0)
        grad_min = metrics.get("grad_min", 1.0)
        layers = metrics.get("layers", 1)
        
        if grad_max > 2.0:
            return (f"Exploding gradients detected! A gradient of {grad_max:.2f} is heavily altering the weights. "
                    f"Consider lowering the learning rate or switching to a stabilizing activation like ReLU or Tanh.", 
                    "#EF4444", "💥")
        elif grad_min < 0.01 and layers > 2:
            return (f"Vanishing gradients. Deep layers are learning too slowly with a minimum gradient of {grad_min:.4f}. "
                    f"The chain rule is shrinking the values; avoid Sigmoid in deep networks.", 
                    "#F59E0B", "📉")
        else:
            return (f"Healthy gradient flow! The chain rule successfully distributed the error backwards "
                    f"without exploding or vanishing, ensuring stable weight updates.", 
                    "#10B981", "🌊")
    
    return ("Analyzing deep learning metrics...", "#C4B5FD", "🧠")
