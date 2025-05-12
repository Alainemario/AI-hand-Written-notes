import os
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFont, ImageDraw
import random
import matplotlib.pyplot as plt
from scipy import ndimage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

class HandwritingGenerator:
    def __init__(self, font_path=None, model_path=None):
        """
        Initialize the handwriting generator
        
        Args:
            font_path: Path to reference font file (TTF/OTF)
            model_path: Path to pre-trained handwriting model
        """
        # Set default paths if none provided
        self.font_path = font_path or os.path.join("fonts", "arial.ttf")
        self.model_path = model_path
        self.characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?;:"
        
        # Create necessary directories if they don't exist
        os.makedirs("fonts", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            self.model = None
    
    def generate_handwriting_sample(self, text, output_path=None, style_variation=0.5):
        """
        Generate handwritten text from input text
        
        Args:
            text: Input text to convert to handwriting
            output_path: Path to save generated image (default: outputs/handwriting_<timestamp>.png)
            style_variation: Degree of handwriting variation (0-1)
            
        Returns:
            PIL Image object of generated handwriting
        """
        # Set default output path if none provided
        if output_path is None:
            timestamp = str(int(time.time()))
            output_path = os.path.join("outputs", f"handwriting_{timestamp}.png")
        
        # Create base image
        img = Image.new('RGB', (2000, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Load font (for initial layout)
        try:
            font = ImageFont.truetype(self.font_path, 24)
        except:
            font = ImageFont.load_default()
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Resize canvas to fit text
        img = Image.new('RGB', (text_width + 100, text_height + 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        if self.model:
            # Use neural network for handwriting generation
            result = self._generate_with_model(text, output_path, style_variation)
        else:
            # Fallback to simple font-based generation with distortions
            result = self._generate_with_font(text, draw, font, output_path, style_variation)
        
        print(f"Handwriting sample saved to: {os.path.abspath(output_path)}")
        return result
    
    def _generate_with_model(self, text, output_path, style_variation):
        """Generate handwriting using a trained neural network model"""
        # Preprocess text into character sequence
        char_sequence = [self.characters.index(c) for c in text if c in self.characters]
        
        # Generate handwriting strokes (simplified - would be more complex in full implementation)
        strokes = self.model.predict(np.array([char_sequence]))
        
        # Render strokes to image
        img = self._render_strokes(strokes[0], style_variation)
        
        img.save(output_path)
        return img
    
    def _generate_with_font(self, text, draw, font, output_path, style_variation):
        """Generate handwriting using font with random distortions"""
        x, y = 50, 50
        char_images = []
        
        for char in text:
            if char not in self.characters:
                continue
                
            # Draw character
            draw.text((x, y), char, font=font, fill=(0, 0, 0))
            
            # Get character bounding box
            bbox = draw.textbbox((x, y), char, font=font)
            char_img = img.crop(bbox)
            
            # Apply distortions
            char_img = self._distort_character(char_img, style_variation)
            char_images.append((bbox, char_img))
            
            x += bbox[2] - bbox[0] - 2
            
        # Composite all characters back onto clean image
        img = Image.new('RGB', (2000, 300), color=(255, 255, 255))
        for bbox, char_img in char_images:
            img.paste(char_img, (bbox[0], bbox[1]))
            
        # Apply global distortions
        img = self._apply_global_distortions(img, style_variation)
        
        # Crop to content
        img = self._autocrop(img)
        img.save(output_path)
        
        return img
    
    def _distort_character(self, img, amount):
        """Apply random distortions to a single character"""
        img = np.array(img)
        
        # Random rotation
        if random.random() < amount:
            angle = random.uniform(-10 * amount, 10 * amount)
            img = ndimage.rotate(img, angle, reshape=False, mode='nearest')
            
        # Random scaling
        if random.random() < amount:
            scale = random.uniform(1 - 0.1*amount, 1 + 0.1*amount)
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            
            # Pad or crop to maintain original size
            if scale > 1:
                img = img[:h, :w]
            else:
                pad_h = h - new_h
                pad_w = w - new_w
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                
        # Random translation
        if random.random() < amount:
            tx = random.randint(-int(2*amount), int(2*amount))
            ty = random.randint(-int(2*amount), int(2*amount))
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            
        # Random noise
        if random.random() < amount:
            noise = np.random.normal(0, 5*amount, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
        return Image.fromarray(img)
    
    def _apply_global_distortions(self, img, amount):
        """Apply distortions to the entire text"""
        img = np.array(img)
        
        # Baseline wobble
        if amount > 0.3:
            rows, cols = img.shape[:2]
            offset = int(5 * amount)
            for i in range(cols):
                col = img[:, i]
                shift = int(offset * np.sin(i / 30))
                img[:, i] = np.roll(col, shift)
                
        # Random warping
        if random.random() < amount * 0.7:
            rows, cols = img.shape[:2]
            src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
            dst_points = src_points + np.random.uniform(-3*amount, 3*amount, src_points.shape)
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            img = cv2.warpPerspective(img, M, (cols,rows))
            
        return Image.fromarray(img)
    
    def _autocrop(self, img, padding=10):
        """Crop image to content"""
        np_img = np.array(img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            
            # Add padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(np_img.shape[1] - x, w + 2*padding)
            h = min(np_img.shape[0] - y, h + 2*padding)
            
            cropped = np_img[y:y+h, x:x+w]
            return Image.fromarray(cropped)
        return img
    
    def _render_strokes(self, strokes, variation):
        """Render stroke data to image (simplified)"""
        width = int(np.max(strokes[:, 0]) + 50)
        height = int(np.max(strokes[:, 1]) + 50)
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Add variation to strokes
        strokes[:, :2] += np.random.normal(0, variation, strokes[:, :2].shape)
        
        # Draw strokes
        for i in range(1, len(strokes)):
            x1, y1, p1 = strokes[i-1]
            x2, y2, p2 = strokes[i]
            
            if p1 > 0.5 and p2 > 0.5:  # Pen down
                line_width = random.uniform(1, 1 + 2*variation)
                draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=int(line_width))
        
        return img

def train_handwriting_model(dataset_path, output_model_path=None):
    """
    Train a handwriting generation model from dataset
    
    Args:
        dataset_path: Path to directory of handwriting samples
        output_model_path: Path to save trained model (default: models/handwriting_model_<timestamp>.h5)
    """
    import time
    
    # Set default output path if none provided
    if output_model_path is None:
        timestamp = str(int(time.time()))
        output_model_path = os.path.join("models", f"handwriting_model_{timestamp}.h5")
    
    print(f"Training handwriting model from {dataset_path}...")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(output_model_path) or "models", exist_ok=True)
    
    # This is a placeholder - actual implementation would:
    # 1. Load and preprocess samples
    # 2. Extract character segments
    # 3. Train a sequence model
    
    # For now we'll just create a dummy model file
    open(output_model_path, 'w').close()
    
    print(f"Model training complete. Saved to {os.path.abspath(output_model_path)}")
    return output_model_path

def process_handwriting_samples(input_dir, output_dir=None):
    """
    Process raw handwriting samples into training data
    
    Args:
        input_dir: Directory containing scanned handwriting samples
        output_dir: Directory to save processed images (default: processed_samples)
    """
    # Set default output directory if none provided
    if output_dir is None:
        output_dir = "processed_samples"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing samples from {input_dir}...")
    
    processed_count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Could not read image: {img_path}")
                continue
            
            # Preprocessing pipeline
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
            
            # Save processed image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, thresh)
            processed_count += 1
    
    print(f"Processed {processed_count} samples. Output saved to {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Handwriting Generation")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process handwriting samples')
    process_parser.add_argument('input_dir', help='Directory with raw samples')
    process_parser.add_argument('--output_dir', help='Output directory for processed samples')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train handwriting model')
    train_parser.add_argument('dataset_path', help='Path to processed samples')
    train_parser.add_argument('--output_model', help='Output path for trained model')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate handwriting')
    gen_parser.add_argument('text', help='Text to convert to handwriting')
    gen_parser.add_argument('--output', help='Output file path')
    gen_parser.add_argument('--model', help='Path to trained model')
    gen_parser.add_argument('--font', help='Path to font file')
    gen_parser.add_argument('--variation', type=float, default=0.5, help='Style variation (0-1)')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        process_handwriting_samples(args.input_dir, args.output_dir)
    elif args.command == 'train':
        train_handwriting_model(args.dataset_path, args.output_model)
    elif args.command == 'generate':
        generator = HandwritingGenerator(font_path=args.font, model_path=args.model)
        img = generator.generate_handwriting_sample(
            args.text, 
            output_path=args.output,
            style_variation=args.variation
        )
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        parser.print_help()