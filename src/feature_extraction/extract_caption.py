import torch
from PIL import Image

from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer



class CaptionExtractor():
    def __init__(self) -> None:
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        max_length = 16
        num_beams = 4
        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
       
    def predict_batch(self, image_paths):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

def main():
    caption_extractor = CaptionExtractor()
    preds = caption_extractor.predict_batch(['data/MAMI_processed/images/train/1.jpg']) # ['a woman in a hospital bed with a woman in a hospital bed']
    
    for pred in preds:
        print(pred)
    
if __name__ == '__main__':
    main()
    
