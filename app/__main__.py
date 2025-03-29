import replicate
import requests
import os
import argparse
from dotenv import load_dotenv
from datetime import datetime


class ImageGenerator:
    def __init__(self, prompt, model):
        # Load environment variables from .env file
        load_dotenv()

        self.prompt = prompt
        self.model = model
        self.current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.output_dir = f"outputs/{self.current_datetime}"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_images(self, aspect_ratio="1:1", num_images=10):
        for seed in range(num_images):
            if self.model == "pro":
                input_dict = {
                    "prompt": self.prompt,
                    "aspect_ratio": aspect_ratio,
                    "safety_tolerance": 5,
                    "seed": seed,
                    "output_format": "jpg",
                }
                output = replicate.run(
                    "black-forest-labs/flux-1.1-pro-ultra", input=input_dict
                )
                outputs = [output]
            elif self.model == "schnell":
                input_dict = {
                    "prompt": self.prompt,
                    "num_inference_steps": 4,
                    "seed": seed,
                    "output_format": "jpg",
                }
                outputs = replicate.run(
                    "black-forest-labs/flux-schnell", input=input_dict
                )
            else:
                raise ValueError("Invalid model. Choose 'pro' or 'schnell'.")

            # Download the images from the URLs
            for i, output_url in enumerate(outputs):
                response = requests.get(output_url)
                if response.status_code == 200:
                    with open(f"{self.output_dir}/output_{seed}_{i}.jpg", "wb") as f:
                        f.write(response.content)
                else:
                    print(f"Failed to download image for seed {seed}, output {i}")


def main():
    parser = argparse.ArgumentParser(description="Generate images using Flux models.")
    parser.add_argument("prompt", type=str, help="The prompt for image generation")
    parser.add_argument(
        "--model",
        type=str,
        choices=["pro", "schnell"],
        default="pro",
        help="The model to use (pro or schnell)",
    )
    parser.add_argument(
        "--aspect_ratio", type=str, default="1:1", help="Aspect ratio of image"
    )
    parser.add_argument(
        "--num_images", type=int, default=1, help="Number of images to generate"
    )

    args = parser.parse_args()

    generator = ImageGenerator(args.prompt, args.model)
    generator.generate_images(args.aspect_ratio, args.num_images)


if __name__ == "__main__":
    main()
