# ComfyUI_ZIM
The unofficial implementation of [ZIM](https://github.com/naver-ai/ZIM) in ComfyUI.

[ZIM](https://github.com/naver-ai/ZIM): Zero-Shot Image Matting for Anything
![image](https://github.com/user-attachments/assets/dd93c774-eb06-4587-9708-a1b2cd2d7718)
workflow:

![image](https://github.com/user-attachments/assets/9388d459-6144-43ef-8e06-a7afacd8649a)

# How To Use
Use the [ComfyUI-KJnodes](https://github.com/kijai/ComfyUI-KJNodes) to obtain positive points input, or you can also use a node related to bounding boxes (bbox) to input bbox (similar to SAM2), but the effect is not very good.

Download the model from [here](https://huggingface.co/naver-iv/zim-anything-vitb/tree/main/zim_vit_b_2043) or [here](https://huggingface.co/naver-iv/zim-anything-vitl/tree/main/zim_vit_l_2092)  and place it in the **models\ZIM** directory, such as **models\zim\zim_vit_l_2092** or **models\zim\zim_vit_b_2043**, where the **zim_vit_l_2092** folder includes **encoder.onnx** and **decoder.onnx**.

# Note
I used Gemini to write this node, and I'm not very proficient in Python. Any suggestions for code optimization are welcome!





