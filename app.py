import numpy as np
import tensorflow as tf
import gradio as gr
import cv2
from PIL import Image
import traceback

IMG_SIZE = 260
THRESHOLD = 0.5
WEIGHTS_PATH = "model/epoch_10.weights.h5"

#  BUILD MODEL ARCHITECTURE 
def build_model():
    base = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    x_in = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(x_in, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    x_out = tf.keras.layers.Dense(2, activation="sigmoid", name="dense_1")(x)
    return tf.keras.Model(x_in, x_out, name="CheXchoNet")

model = build_model()
_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32), training=False)
model.load_weights(WEIGHTS_PATH)
print("Weights loaded:", WEIGHTS_PATH)

# Layers for Grad-CAM manual forward
base = model.get_layer("efficientnetb2")
gap = model.get_layer("global_average_pooling2d")
fc1 = model.get_layer("dense")
drop = model.get_layer("dropout")
out = model.get_layer("dense_1")

# PREPROCESS
def preprocess_image_tf(pil_img: Image.Image) -> tf.Tensor:
    pil_img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(pil_img).astype(np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    x = tf.convert_to_tensor(arr)
    return tf.expand_dims(x, axis=0)

# GRAD-CAM 
def generate_gradcam(pil_img: Image.Image, output_index: int):
    x = preprocess_image_tf(pil_img)

    with tf.GradientTape() as tape:
        conv_out = base(x, training=False)
        tape.watch(conv_out)

        y = gap(conv_out)
        y = fc1(y)
        y = drop(y, training=False)
        preds = out(y)

        loss = preds[:, output_index]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

    heatmap = heatmap.numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

    img_rgb = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(img_rgb, 0.6, heatmap_rgb, 0.4, 0)

# PREDICT 
def predict(image: Image.Image, cam_target: str):
    try:
        if image is None:
            return (
                {
                    "SLVH Score": 0.0,
                    "DLV Score": 0.0,
                    "SLVH Status": "No image",
                    "DLV Status": "No image",
                    "Overall": "No image",
                    "Threshold": THRESHOLD
                },
                None,
                "No image uploaded"
            )

        x = preprocess_image_tf(image)
        preds = model(x, training=False).numpy()[0]

        slvh = float(round(float(preds[0]), 4))
        dlv  = float(round(float(preds[1]), 4))

        slvh_status = "SLVH Abnormal" if slvh >= THRESHOLD else "SLVH Normal"
        dlv_status  = "DLV Abnormal"  if dlv  >= THRESHOLD else "DLV Normal"
        overall = "Abnormal" if (slvh >= THRESHOLD or dlv >= THRESHOLD) else "Normal"

        result = {
            "SLVH Score": slvh,
            "DLV Score": dlv,
            "SLVH Status": slvh_status,
            "DLV Status": dlv_status,
            "Overall": overall,
            "Threshold": THRESHOLD
        }

        idx = 0 if cam_target == "SLVH" else 1
        cam_img = generate_gradcam(image, idx)

        return result, cam_img, "Success"

    except Exception:
        return (
            {
                "SLVH Score": 0.0,
                "DLV Score": 0.0,
                "SLVH Status": "Error",
                "DLV Status": "Error",
                "Overall": "Error",
                "Threshold": THRESHOLD
            },
            None,
            traceback.format_exc()
        )

# UI
custom_css = """
:root { --radius: 18px; }

.gradio-container {
  max-width: 1100px !important;
  margin: 0 auto !important;
}

.hero {
  padding: 18px 8px 6px 8px;
}

.hero h1 {
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 6px;
}

.subtle {
  opacity: 0.75;
  font-size: 14px;
}

.card {
  border-radius: var(--radius) !important;
  overflow: hidden;
}

.pill {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  opacity: 0.9;
  border: 1px solid rgba(127,127,127,0.25);
}

.footer-note {
  opacity: 0.7;
  font-size: 12px;
  line-height: 1.35;
}

/* Make images look premium */
img {
  border-radius: 14px !important;
}

/* Better mobile spacing */
@media (max-width: 840px) {
  .gradio-container { padding: 8px !important; }
}
"""

with gr.Blocks(title="CheXchoNet – Cardiac Diagnosis", css=custom_css) as demo:
    gr.HTML(f"""
    <div class="hero">
      <h1>CheXchoNet</h1>
      <div class="subtle">Cardiac screening with Grad-CAM • EfficientNetB2 • Threshold <span class="pill">{THRESHOLD}</span></div>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=340):
            with gr.Group(elem_classes=["card"]):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=360
                )

                cam_target = gr.Radio(
                    choices=["SLVH", "DLV"],
                    value="SLVH",
                    label="Grad-CAM Target",
                    interactive=True
                )

                btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=1, min_width=340):
            with gr.Group(elem_classes=["card"]):
                gradcam_output = gr.Image(
                    label="Grad-CAM Visualization",
                    height=360
                )

    with gr.Row():
        with gr.Column(scale=1, min_width=340):
            prediction_output = gr.JSON(label="Prediction + Status")

    gr.HTML("""
    <div class="footer-note">
      Note: This tool is for educational/research use and not a clinical diagnosis. If you need medical advice, consult a qualified professional.
    </div>
    """)

    btn.click(
        fn=predict,
        inputs=[image_input, cam_target],
        outputs=[prediction_output, gradcam_output]
    )


demo.launch()
