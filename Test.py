import os
from handwriting_generator import HandwritingGenerator  # Replace with your script's filename
import matplotlib.pyplot as plt

def test_handwriting():
    # Configure paths
    model_path = "models/my_handwriting_model.h5"  # Update if needed
    font_path = "fonts/your_font.ttf"              # Optional: replace with your .ttf font
    output_path = "outputs/test_output.png"

    # Input text
    text = "Hello, this is a handwriting test!"

    # Initialize generator
    generator = HandwritingGenerator(model_path=model_path, font_path=font_path)

    # Generate handwriting image
    img = generator.generate_handwriting_sample(text, output_path=output_path, style_variation=0.6)

    # Display the result
    plt.imshow(img)
    plt.title("Generated Handwriting Sample")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_handwriting()
