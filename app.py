import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="LSTM Network IDS",
    page_icon="🛡️",
    layout="wide"
)

st.title("Sequence-Based Deep Learning Model for Network Intrusion Detection and Threat Classification Using LSTM and GRUs")

st.markdown(
    """
    Predict network traffic status (**BENIGN vs ATTACK**) using a
    trained Robust Bidirectional LSTM model exported to ONNX.
    """
)

# =========================================================
# APPLICATION DIRECTORY
# =========================================================
# Always use the directory where THIS app.py is located.
APP_DIR = Path(__file__).resolve().parent

MODEL_FILE = "robust_bidirectional_lstm_model.onnx"
SCALER_FILE = "robust_scaler.joblib"
FEATURE_FILE = "feature_columns.json"

MODEL_PATH = APP_DIR / MODEL_FILE
SCALER_PATH = APP_DIR / SCALER_FILE
FEATURE_PATH = APP_DIR / FEATURE_FILE


# =========================================================
# DIAGNOSTICS
# =========================================================
with st.expander("🔧 Application & File Diagnostics"):

    st.write("**Running app:**")
    st.code(str(Path(__file__).resolve()))

    st.write("**Application directory:**")
    st.code(str(APP_DIR))

    st.write("**Expected files:**")

    files_to_check = {
        "ONNX Model": MODEL_PATH,
        "Scaler": SCALER_PATH,
        "Feature JSON": FEATURE_PATH
    }

    for name, path in files_to_check.items():

        if path.is_file():
            st.success(f"✅ {name} FOUND")
        else:
            st.error(f"❌ {name} MISSING")

        st.code(str(path))

    st.write("**Files currently in application directory:**")

    try:
        directory_files = sorted(
            [file.name for file in APP_DIR.iterdir()]
        )

        for filename in directory_files:
            st.write(f"- `{filename}`")

    except Exception as error:
        st.error(f"Could not list directory: {error}")


# =========================================================
# ONNX RUNTIME
# =========================================================
try:

    import onnxruntime as ort

except ModuleNotFoundError:

    st.error("❌ ONNX Runtime is not installed.")

    st.markdown(
        """
        Activate your virtual environment and run:

        ```powershell
        python -m pip install --no-cache-dir onnxruntime
        ```
        """
    )

    st.stop()


# =========================================================
# CHECK REQUIRED FILES
# =========================================================
missing_files = []

if not MODEL_PATH.is_file():
    missing_files.append(MODEL_FILE)

if not SCALER_PATH.is_file():
    missing_files.append(SCALER_FILE)

if not FEATURE_PATH.is_file():
    missing_files.append(FEATURE_FILE)


if missing_files:

    st.error("❌ Required model files could not be found.")

    st.write("The application is looking in:")

    st.code(str(APP_DIR))

    st.write("Missing files:")

    for filename in missing_files:
        st.write(f"- `{filename}`")

    st.warning(
        "Make sure the filenames match exactly and the files "
        "are in the same folder as app.py."
    )

    st.stop()


# =========================================================
# LOAD MODEL PIPELINE
# =========================================================
@st.cache_resource
def load_pipeline():

    # Load ONNX model
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Load feature names
    with open(
        FEATURE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        features = json.load(file)

    return session, scaler, features


try:

    session, scaler, features = load_pipeline()

except Exception as error:

    st.error("❌ Failed to load model pipeline.")

    st.exception(error)

    st.stop()


# =========================================================
# MODEL INFORMATION
# =========================================================
st.success("✅ Model pipeline loaded successfully.")

try:

    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model Input",
            model_input.name
        )

    with col2:
        st.metric(
            "Features",
            len(features)
        )

    with col3:
        st.metric(
            "Sequence Length",
            10
        )

    st.caption(
        f"ONNX Input Shape: {model_input.shape} | "
        f"Output: {model_output.name}"
    )

except Exception as error:

    st.warning(
        f"Could not read model metadata: {error}"
    )


# =========================================================
# INPUT CONFIGURATION
# =========================================================
SEQUENCE_LENGTH = 10
NUMBER_OF_FEATURES = len(features)

st.markdown("---")

st.subheader("📊 Network Packet Sequence")

st.write(
    f"""
    Enter a sequence of **{SEQUENCE_LENGTH} packets**.

    Each packet contains **{NUMBER_OF_FEATURES} features**.
    The same RobustScaler used during training will be applied
    before inference.
    """
)


# =========================================================
# DEFAULT DATA
# =========================================================
default_values = np.zeros(
    (
        SEQUENCE_LENGTH,
        NUMBER_OF_FEATURES
    ),
    dtype=np.float32
)

default_dataframe = pd.DataFrame(
    default_values,
    columns=features
)


# =========================================================
# DATA EDITOR
# =========================================================
edited_dataframe = st.data_editor(
    default_dataframe,
    use_container_width=True,
    num_rows="fixed",
    key="network_packet_editor"
)


# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_sequence(dataframe):

    # Convert dataframe to NumPy
    raw_values = dataframe.to_numpy(
        dtype=np.float32
    )

    # Expected shape
    expected_shape = (
        SEQUENCE_LENGTH,
        NUMBER_OF_FEATURES
    )

    if raw_values.shape != expected_shape:

        raise ValueError(
            f"Expected input shape {expected_shape}, "
            f"but received {raw_values.shape}."
        )

    # Check invalid values
    if not np.isfinite(raw_values).all():

        raise ValueError(
            "Input contains NaN or infinite values."
        )

    # Apply RobustScaler
    scaled_values = scaler.transform(
        raw_values
    ).astype(np.float32)

    # Convert:
    #
    # (10, 68)
    #
    # to:
    #
    # (1, 10, 68)
    #
    input_tensor = np.expand_dims(
        scaled_values,
        axis=0
    ).astype(np.float32)

    # Get ONNX input name
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(
        None,
        {
            input_name: input_tensor
        }
    )

    if len(outputs) == 0:

        raise RuntimeError(
            "The ONNX model returned no outputs."
        )

    # Convert output to NumPy
    output_array = np.asarray(
        outputs[0]
    )

    # Extract first prediction
    prediction_probability = float(
        output_array.reshape(-1)[0]
    )

    # Keep probability between 0 and 1
    prediction_probability = float(
        np.clip(
            prediction_probability,
            0.0,
            1.0
        )
    )

    # Classification
    if prediction_probability >= 0.5:

        prediction_label = "ATTACK"

    else:

        prediction_label = "BENIGN"

    return (
        prediction_probability,
        prediction_label
    )


# =========================================================
# RUN INFERENCE
# =========================================================
st.markdown("---")

run_inference = st.button(
    "🔍 Run Threat Assessment",
    type="primary",
    use_container_width=True
)


if run_inference:

    try:

        probability, label = predict_sequence(
            edited_dataframe
        )

        # =================================================
        # RESULTS
        # =================================================
        st.markdown("---")

        st.subheader("🚨 Threat Assessment Results")

        result_col1, result_col2 = st.columns(2)

        # Classification
        with result_col1:

            if label == "ATTACK":

                st.error(
                    f"### 🚨 {label}"
                )

            else:

                st.success(
                    f"### 🟢 {label}"
                )

        # Probability
        with result_col2:

            st.metric(
                "Attack Probability",
                f"{probability * 100:.2f}%"
            )

        # Probability bar
        st.progress(
            probability,
            text=(
                f"Attack Probability: "
                f"{probability * 100:.2f}%"
            )
        )

        # =================================================
        # INTERPRETATION
        # =================================================
        if label == "ATTACK":

            st.warning(
                """
                ⚠️ The trained LSTM model classified this
                packet sequence as **ATTACK**.

                Investigate the corresponding network traffic,
                source/destination information, ports, and
                traffic characteristics.
                """
            )

        else:

            st.info(
                """
                🟢 The trained LSTM model classified this
                packet sequence as **BENIGN**.
                """
            )

        # =================================================
        # TECHNICAL DETAILS
        # =================================================
        with st.expander("🔬 Technical Inference Details"):

            st.write(
                "**Raw Input Shape:**",
                edited_dataframe.shape
            )

            st.write(
                "**Scaled Input Shape:**",
                (
                    SEQUENCE_LENGTH,
                    NUMBER_OF_FEATURES
                )
            )

            st.write(
                "**ONNX Tensor Shape:**",
                (
                    1,
                    SEQUENCE_LENGTH,
                    NUMBER_OF_FEATURES
                )
            )

            st.write(
                "**Classification Threshold:**",
                "0.50"
            )

            st.write(
                "**ONNX Input:**",
                session.get_inputs()[0].name
            )

            st.write(
                "**ONNX Output:**",
                session.get_outputs()[0].name
            )

    except Exception as error:

        st.error(
            "❌ Inference failed."
        )

        st.exception(error)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "Robust Bidirectional LSTM Network IDS • "
    "ONNX Runtime • Streamlit"
)