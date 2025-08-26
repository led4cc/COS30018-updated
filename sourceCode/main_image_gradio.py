import gradio as gr
import webbrowser
import cv2
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
from utils_LP import crop_n_rotate_LP

ocr = PaddleOCR(lang='en')
model_LP = YOLO('license_plate_detector.pt')

def process_image(img_pil):
    source_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    results = model_LP.predict(source_img, conf=0.25)

    LP_detected_img = source_img.copy()
    texts, crops = [], []

    for result in results:
        if result.boxes is None or result.boxes.xyxy is None:
            continue
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist()[:4])
            angle, _, LP_rotated = crop_n_rotate_LP(source_img, x1, y1, x2, y2)
            if LP_rotated is None:
                continue

            LP_rotated_rgb = cv2.cvtColor(LP_rotated, cv2.COLOR_BGR2RGB)
            result_ocr = ocr.ocr(LP_rotated_rgb, cls=True)
            text = ' '.join([line[1][0] for line in result_ocr[0]]) if result_ocr and result_ocr[0] else "Không nhận dạng được"
            texts.append(text)
            crops.append(Image.fromarray(LP_rotated_rgb))

            cv2.putText(LP_detected_img, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    output_img = Image.fromarray(cv2.cvtColor(LP_detected_img, cv2.COLOR_BGR2RGB))
    one_crop = crops[0] if crops else img_pil
    final_text = texts[0] if texts else "Không nhận dạng được"
    return output_img, one_crop, final_text

# CSS dùng ID chứ không dùng class
css = """
#result_image, #rotated_image {
    padding-top: 32px;
    padding-bottom: 16px;
    padding-left: 16px;
    padding-right: 16px;
    margin: 0 auto;
    align-items: center;
    display: flex;
    justify-content: center;
    border-radius: 6px;
    border: 1px solid #ddd;
    background-color: #fafafa;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## Nhận diện biển số xe")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Tải ảnh biển số lên")
            submit_btn = gr.Button("Xử lý")
            text_output = gr.Textbox(label="Biển số nhận dạng")
        with gr.Column():
            rotated_img = gr.Image(label="Ảnh biển số đã xoay", elem_id="rotated_image")
            result_img = gr.Image(label="Ảnh kèm kết quả", elem_id="result_image")

    submit_btn.click(fn=process_image, inputs=input_img,
                     outputs=[result_img, rotated_img, text_output])

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:7860")
    demo.launch()
