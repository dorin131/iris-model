from sklearn.linear_model import LogisticRegression
from safetensors import safe_open
from pathlib import Path

model = None


def get_model():
    # Create model
    model = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')

    # Load weights from file
    weights_path = Path(__file__).with_name('model.safetensors').absolute()
    with safe_open(weights_path, framework="numpy", device="cpu") as f:
        model.coef_ = f.get_tensor('coef')
        model.intercept_ = f.get_tensor('intercept')
        model.classes_ = f.get_tensor('classes')

    return model


def handle(payload: dict) -> dict:
    global model
    if not model:
        model = get_model()

    candidates = payload["candidates"]
    predictions = model.predict(candidates)

    result = {
        "score": predictions.tolist(),
    }

    return result
