import os
from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path: str, output_folder: str, dpi: int = 150, fmt: str = "jpeg") -> list:
    """
    Converts each page of a PDF into image files (JPEG by default), optimized for OCR.
    
    Args:
        pdf_path: Path to the PDF file.
        output_folder: Directory to store output images.
        dpi: Image resolution (higher = better OCR accuracy, slower processing).
        fmt: Output image format ("jpeg" or "png").

    Returns:
        List of file paths to the converted image pages.
    """
    os.makedirs(output_folder, exist_ok=True)

    print(f"[PDF → Images] Converting {pdf_path} to images at {dpi} DPI...")

    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        fmt=fmt,
        output_folder=output_folder,
        paths_only=True,
        thread_count=8  # Adjust this depending on CPU capabilities
    )

    final_paths = []
    for idx, img_path in enumerate(sorted(images)):
        target_path = os.path.join(output_folder, f"page_{idx + 1}.{fmt}")
        os.rename(img_path, target_path)
        final_paths.append(target_path)

    print(f"[PDF → Images] Done: {len(final_paths)} pages saved in {output_folder}")
    return final_paths
