import os
import sys
import argparse
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel



# Set up defaults and get API key from environment variable
defaults = {
   
    "size": "1024x1024",
    "quality": "standard",
    "number": "1",
}

# create VertexAI client

# TODO(developer): Update and un-comment below lines
# create VertexAI client
project_id = os.getenv("GCP_PROJECT_ID")
location_id=os.getenv("GCP_LOCATION")
img_model_id=os.getenv("GCP_IMG_GEN_MODEL")
output_file = "page1.png"
#prompt = "Generate a cover book for story Binky-The-Adventurous-Bunny" # The text prompt describing what you want to see.

# Initialize VertexAI client
vertexai.init(project=project_id, location=location_id)
# Initialize Image generaton Model 
model = ImageGenerationModel.from_pretrained(img_model_id)


# Function to validate and parse arguments
def validate_and_parse_args(parser):
    args = parser.parse_args()

    for key, value in vars(args).items():
        if not value:
            args.__dict__[key] = parser.get_default(key)

    
    if not args.prompt:
        parser.error('The --prompt argument is required.')
    if not args.number.isdigit():
        parser.error('The --number argument must be a number.')
    args.number = int(args.number)

    return args

def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="CLI for image generation prompt using Vertex AI ImageGen model.")
   
    parser.add_argument('-p', '--prompt', type=str, required=True, help='Prompt for image generation.')
   
    parser.add_argument('-s', '--size', type=str, default=defaults["size"],
                        help=f'Size of the image to generate, format WxH (e.g. {defaults["size"]}). Default is {defaults["size"]}.')
    parser.add_argument('-q', '--quality', type=str, default=defaults["quality"],
                        help=f'Quality of the generated image. Allowed values are "standard" or "hd". Default is "{defaults["quality"]}"')
    parser.add_argument('-n', '--number', type=str, default=defaults["number"],
                        help='Number of images to generate. Default is 1.')
    args = validate_and_parse_args(parser)

    # Initialize Vertex AI client
   
   
    # Make request to the Vertex AI API
    try:
        images = model.generate_images(
        prompt=args.prompt,
        # Optional parameters
        number_of_images=1,
        language="en",
        # You can't use a seed value and watermark at the same time.
        # add_watermark=False,
        # seed=100,
        aspect_ratio="1:1",
        safety_filter_level="block_some",
        person_generation="allow_adult",
        )

        images[0].save(location=output_file, include_generation_parameters=False)

        # Optional. View the generated image in a notebook.
        # images[0].show()

        print(f"Created output image using {len(images[0]._image_bytes)} bytes")
    except Exception as e:
        print(f"Received an error code while generating images: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
