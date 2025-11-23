"""Vision processing for image and video analysis."""

import base64
from typing import Dict, Any, List, Optional
from pathlib import Path
import io
from PIL import Image
import google.generativeai as genai

from src.core.config import settings
from src.core.logger import setup_logger

logger = setup_logger(__name__)


class VisionProcessor:
    """Process images and videos using Gemini Vision."""

    def __init__(self):
        """Initialize vision processor."""
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel("gemini-pro-vision")
        self.logger = setup_logger(__name__)

    async def analyze_image(
        self,
        image_path: str = None,
        image_bytes: bytes = None,
        prompt: str = "Describe this image in detail.",
    ) -> Dict[str, Any]:
        """
        Analyze an image.

        Args:
            image_path: Path to image file
            image_bytes: Image as bytes
            prompt: Analysis prompt

        Returns:
            Analysis results
        """
        try:
            # Load image
            if image_path:
                image = Image.open(image_path)
            elif image_bytes:
                image = Image.open(io.BytesIO(image_bytes))
            else:
                raise ValueError("Must provide either image_path or image_bytes")

            self.logger.info(f"Analyzing image with prompt: {prompt[:50]}...")

            # Analyze with Gemini Vision
            response = await self.model.generate_content_async([prompt, image])

            return {
                "success": True,
                "prompt": prompt,
                "analysis": response.text,
                "image_size": image.size,
                "image_mode": image.mode,
            }

        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def detect_objects(
        self, image_path: str = None, image_bytes: bytes = None
    ) -> Dict[str, Any]:
        """
        Detect objects in image.

        Args:
            image_path: Path to image
            image_bytes: Image as bytes

        Returns:
            Detected objects
        """
        prompt = "List all objects you can see in this image. For each object, provide its name and location description."
        return await self.analyze_image(image_path, image_bytes, prompt)

    async def read_text(
        self, image_path: str = None, image_bytes: bytes = None
    ) -> Dict[str, Any]:
        """
        Extract text from image (OCR).

        Args:
            image_path: Path to image
            image_bytes: Image as bytes

        Returns:
            Extracted text
        """
        prompt = "Extract and return all text visible in this image. Maintain the original formatting and structure."
        return await self.analyze_image(image_path, image_bytes, prompt)

    async def describe_scene(
        self, image_path: str = None, image_bytes: bytes = None
    ) -> Dict[str, Any]:
        """
        Describe the scene in detail.

        Args:
            image_path: Path to image
            image_bytes: Image as bytes

        Returns:
            Scene description
        """
        prompt = "Provide a detailed description of this scene, including: setting, people, objects, activities, mood, and any notable details."
        return await self.analyze_image(image_path, image_bytes, prompt)

    async def answer_about_image(
        self, question: str, image_path: str = None, image_bytes: bytes = None
    ) -> Dict[str, Any]:
        """
        Answer a question about an image.

        Args:
            question: Question to answer
            image_path: Path to image
            image_bytes: Image as bytes

        Returns:
            Answer
        """
        return await self.analyze_image(image_path, image_bytes, question)

    async def compare_images(
        self, image1_path: str, image2_path: str, aspect: str = "differences"
    ) -> Dict[str, Any]:
        """
        Compare two images.

        Args:
            image1_path: First image path
            image2_path: Second image path
            aspect: What to compare (differences, similarities, etc.)

        Returns:
            Comparison results
        """
        try:
            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)

            prompt = f"Compare these two images and describe their {aspect} in detail."

            response = await self.model.generate_content_async(
                [prompt, image1, image2]
            )

            return {
                "success": True,
                "comparison": response.text,
                "aspect": aspect,
            }

        except Exception as e:
            self.logger.error(f"Image comparison failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


class ImageProcessor:
    """Process and manipulate images."""

    def __init__(self):
        """Initialize image processor."""
        self.logger = setup_logger(__name__)

    def resize_image(
        self, image_path: str, max_width: int = 1024, max_height: int = 1024
    ) -> Image.Image:
        """
        Resize image maintaining aspect ratio.

        Args:
            image_path: Path to image
            max_width: Maximum width
            max_height: Maximum height

        Returns:
            Resized image
        """
        image = Image.open(image_path)

        # Calculate new size maintaining aspect ratio
        ratio = min(max_width / image.width, max_height / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))

        resized = image.resize(new_size, Image.LANCZOS)
        self.logger.info(f"Resized image from {image.size} to {resized.size}")

        return resized

    def convert_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """
        Convert PIL Image to bytes.

        Args:
            image: PIL Image
            format: Output format

        Returns:
            Image bytes
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()

    def encode_base64(self, image_path: str) -> str:
        """
        Encode image as base64.

        Args:
            image_path: Path to image

        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        return base64.b64encode(image_bytes).decode("utf-8")

    def crop_image(
        self, image_path: str, x: int, y: int, width: int, height: int
    ) -> Image.Image:
        """
        Crop image to specified region.

        Args:
            image_path: Path to image
            x: X coordinate
            y: Y coordinate
            width: Crop width
            height: Crop height

        Returns:
            Cropped image
        """
        image = Image.open(image_path)
        cropped = image.crop((x, y, x + width, y + height))
        return cropped


class VideoProcessor:
    """Process video files (placeholder for future implementation)."""

    def __init__(self):
        """Initialize video processor."""
        self.logger = setup_logger(__name__)

    async def extract_frames(
        self, video_path: str, num_frames: int = 10
    ) -> List[Image.Image]:
        """
        Extract frames from video.

        Args:
            video_path: Path to video
            num_frames: Number of frames to extract

        Returns:
            List of frames as PIL Images
        """
        # TODO: Implement video frame extraction using OpenCV
        self.logger.warning("Video processing not yet implemented")
        return []

    async def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze video content.

        Args:
            video_path: Path to video

        Returns:
            Analysis results
        """
        # TODO: Implement video analysis
        self.logger.warning("Video analysis not yet implemented")
        return {
            "success": False,
            "error": "Video analysis not implemented",
        }


# Global instances
_vision_processor: Optional[VisionProcessor] = None
_image_processor: Optional[ImageProcessor] = None
_video_processor: Optional[VideoProcessor] = None


def get_vision_processor() -> VisionProcessor:
    """Get global vision processor instance."""
    global _vision_processor
    if _vision_processor is None:
        _vision_processor = VisionProcessor()
    return _vision_processor


def get_image_processor() -> ImageProcessor:
    """Get global image processor instance."""
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor


def get_video_processor() -> VideoProcessor:
    """Get global video processor instance."""
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor
