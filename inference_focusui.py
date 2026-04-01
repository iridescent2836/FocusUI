from PIL import Image
import torch
from transformers import AutoProcessor

from focusui.modeling_focusui_qwen25vl import FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer
# from focusui.modeling_focusui_qwen3vl import FocusUI_Qwen3VLForConditionalGenerationWithPointer
from focusui.inference import inference_focusui_token_select
from focusui.constants import grounding_system_message_guiactor_qwen25vl

from evaluation.shared_grounding_eval import save_patch_saliency_heatmap_overlay, draw_point


# Load model and processor
model_path = "./checkpoints/FocusUI-3B"   # if not downloaded, use url: "yyyang/FocusUI-Qwen3VL-3B"
model = FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",  # "flash_attention_2" if available
).eval()

# model_path = "./checkpoints/FocusUI-Qwen-3VL-2B"  # if not downloaded, use url: "yyyang/FocusUI-Qwen-3VL-2B"
# model = FocusUI_Qwen3VLForConditionalGenerationWithPointer.from_pretrained(
#     model_path,
#     dtype=torch.bfloat16,
#     device_map="cuda",
#     attn_implementation="sdpa",  # "flash_attention_2" if available
# ).eval()
processor = AutoProcessor.from_pretrained(model_path)

# Prepare conversation
image_path = "assets/example_screenshot.png"
image_path = "./datasets/Example-Data/images/1c6422e3-8eea-44db-9d70-67e74920ae02.png"

conversation = [
    {
        "role": "system",
        "content": [{"type": "text", "text": grounding_system_message_guiactor_qwen25vl}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            # {"type": "text", "text": "Go to 'Watch Live'."}
            {"type": "text", "text": "Submit an application to develop and list an app."}

        ]
    }
]

# Configure visual token selection


# Run inference
result = inference_focusui_token_select(
    conversation=conversation,
    model=model,
    tokenizer=processor.tokenizer,
    data_processor=processor,
    topk=3,
)

# Get predicted coordinates
topk_points = result['topk_points']
top1_point = topk_points[0]
print(f"Top-1 point: {top1_point}")

# Save patch score prediction overlay
heatmap_saved = save_patch_saliency_heatmap_overlay(
    screenshot=image_path,
    pred=result,
    save_name=f"patch_saliency_score_prediction.png",
    image_patch_size=16,
)
print(f"Heatmap saved to: {heatmap_saved}")

# Draw point on the image
image = draw_point(Image.open(image_path), top1_point)
image.save(f"./point_on_image.png")
print(f"Grounding result saved to: ./point_on_image.png")

patch_score_pred = result.get("patch_score_pred", None)

if patch_score_pred is not None:
    print(f"Patch score prediction shape: {patch_score_pred.shape}")
    torch.save(patch_score_pred, "patch_score_pred.pt")
else:
    print("No patch score prediction found in the result.")


image_grid_thw = result.get("image_grid_thw", None)
if image_grid_thw is not None:
    print(f"Image grid dimensions (thw): {image_grid_thw}")
else:
    print("No image grid dimensions found in the result.")