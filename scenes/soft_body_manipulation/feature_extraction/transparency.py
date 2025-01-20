from PIL import Image
import numpy as np

def adjust_transparency(image, transparency_factor):
    """
    Adjust the transparency of an image. The transparency_factor is a float value where
    1.0 means full opacity and 0.0 means fully transparent.
    """
    # Ensure the image has an alpha channel (RGBA)
    image = image.convert("RGBA")
    
    # Get image data
    data = np.array(image)
    
    # Adjust the alpha channel (index 3) to decrease transparency
    data[..., 3] = (data[..., 3] * transparency_factor).astype(np.uint8)
    
    # Convert back to image
    return Image.fromarray(data)

def overlay_images(images):
    """
    Overlay images on top of each other with increasing transparency for each subsequent image.
    """
    # Start with a blank canvas (first image) in RGBA mode (with alpha channel for transparency)
    base_image = images[0].convert("RGBA")
    
    # Overlay each image on top of the previous one with increased transparency
    for idx, img in enumerate(images, start=1):
        # Set a transparency factor that increases with each layer
        transparency_factor = 0.7-((len(images)-idx) * 0.08)
        # print(f"transparency{idx} : {transparency_factor}")
        # Adjust transparency of the current image
        img_with_adjusted_transparency = adjust_transparency(img, transparency_factor)
        
        # Overlay the adjusted image on the base image (blending)
        base_image = Image.alpha_composite(base_image, img_with_adjusted_transparency)
    
    return base_image

if __name__=="__main__":
    # Example usage
    image_paths = ["/home/sofa/sofa_utils/misc/dataset_sofa/000001.jpg", "/home/sofa/sofa_utils/misc/dataset_sofa/000006.jpg", "/home/sofa/sofa_utils/misc/dataset_sofa/000009.jpg"]  # List your image paths
    images = [Image.open(img_path) for img_path in image_paths]

    # Overlay the images with increasing transparency
    final_image = overlay_images(images)

    # Save the final image
    final_image.save("final_overlay_image.png")
    final_image.show()
