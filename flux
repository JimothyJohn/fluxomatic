#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "replicate",
#     "requests",
#     "python-dotenv",
# ]
# ///

import replicate
import requests
import os
import re
import argparse
from dotenv import load_dotenv
from datetime import datetime

class ImageGenerator:
    def __init__(self, prompt, model):
        # Load environment variables from .env file
        load_dotenv()

        self.prompt = prompt
        self.model = model
        
        # Create folder name from the first 4 words of the prompt
        clean_prompt = re.sub(r"[^\w\s]", "", self.prompt)
        words = clean_prompt.split()
        folder_name = "_".join(words[:4])
        
        self.output_dir = f"{os.getenv('HOME')}/Pictures/flux/{folder_name}"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_images(self, aspect_ratio="1:1", num_images=1, raw=False):
        for seed in range(num_images):
            # seed += 50
            if self.model == "pro":
                input_dict = {
                    "prompt": self.prompt,
                    "aspect_ratio": aspect_ratio,
                    "safety_tolerance": 5,
                    "seed": seed,
                    "output_format": "jpg",
                    "image_prompt_strength": 0,
                    "raw": raw,
                }
                output = replicate.run(
                    "black-forest-labs/flux-2-pro", input=input_dict
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
                    # User requested simple filenames: 0.jpg, 1.jpg, etc.
                    # We need a unique index across all seeds if num_images > 1?
                    # The loop is `for seed in range(num_images)`. 
                    # If each seed produces 1 image (outputs is list of 1 usually), then seed is the index.
                    # If multiple images per seed, we need a global counter.
                    # `num_images` drives the seed loop. `outputs` loop is usually 1.
                    # Let's use a global index based on (seed * len(outputs) + i) or just simplest assumption.
                    # Actually, since `seed` goes from 0 to num_images-1, and likely 1 output per seed:
                    filename = f"{seed}.jpg" 
                    if len(outputs) > 1:
                         # Fallback if multiple outputs per seed
                         filename = f"{seed}_{i}.jpg"
                    
                    with open(f"{self.output_dir}/{filename}", "wb") as f:
                        f.write(response.content)
                else:
                    print(f"Failed to download image for seed {seed}, output {i}")


def main():
    parser = argparse.ArgumentParser(description="Generate images using Flux models.")
    parser.add_argument("prompt", type=str, help="The prompt for image generation")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["pro", "schnell"],
        default="pro",
        help="The model to use (pro or schnell)",
    )
    parser.add_argument(
        "-ar", "--aspect_ratio", type=str, default="1:1", help="Aspect ratio of image"
    )
    parser.add_argument(
        "-n", "--num_images", type=int, default=1, help="Number of images to generate"
    )
    parser.add_argument(
        "-r", "--raw", action="store_true", help="Whether to use raw output"
    )
    args = parser.parse_args()

    generator = ImageGenerator(args.prompt, args.model)
    generator.generate_images(args.aspect_ratio, args.num_images, args.raw)


if __name__ == "__main__":
    main()
