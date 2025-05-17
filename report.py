import sys
from pathlib import Path
import io # <-- Import io for in-memory streams

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import TableStyle

# Import Pillow (ensure you have it installed: pip install Pillow)
try:
    from PIL import Image as PILImage
    # Newer Pillow versions use Resampling enum
    try:
        RESAMPLE_FILTER = PILImage.Resampling.LANCZOS
    except AttributeError:
        # Fallback for older Pillow versions
        RESAMPLE_FILTER = PILImage.LANCZOS
except ImportError:
    print("Error: Pillow library not found. Please install it: pip install Pillow", file=sys.stderr)
    sys.exit(1)

# Assume _name_mapping and file_to_dict are defined as in your original code
_name_mapping = {0: 'Macchia Chiara',
                 1: 'Scrostamento',
                 3: 'Area di Intervento'}

def file_to_dict(file_path: str) -> dict:
    # --- Your file_to_dict function goes here (identical to your original) ---
    data_dict = {}
    line_number = 0
    try:
        with open(file_path, 'r') as f: # 'r' is for read mode
            for line in f:
                line_number += 1
                line = line.strip()
                if not line:
                    continue
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key_str, value_str = parts
                    try:
                        key = int(key_str.strip())
                        value = int(value_str.strip())
                        data_dict[key] = value
                    except ValueError:
                        print(f"Warning: Could not convert key '{key_str}' or value '{value_str}' to integer on line {line_number}. Storing as strings.", file=sys.stderr)
                        pass # Or handle differently
                else:
                    print(f"Warning: Skipping malformed line {line_number}: '{line}' (Expected format 'key:value')", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'", file=sys.stderr)
        return {} # Return empty on file not found
    except Exception as e:
        print(f"An unexpected error occurred reading {file_path}: {e}", file=sys.stderr)
        return {} # Return empty on other errors
    return data_dict
# --- End of file_to_dict function ---


def create_image_report(image_directory, output_pdf_filename="image_report.pdf", train_file_path="train.txt"):
    """
    Generates a PDF report showing image pairs side-by-side with captions.
    Images are scaled down before embedding to reduce PDF size.

    Args:
        image_directory (str): The path to the directory containing the images.
                               Images should be named like '1_gt.png', '1_pred.png', etc.
        output_pdf_filename (str): The name of the output PDF file.
        train_file_path (str): Path to the 'train.txt' file for category mapping.
    """
    _the_dict = file_to_dict(train_file_path)
    if not _the_dict:
        print(f"Error: Could not read or process '{train_file_path}'. Cannot map categories.")
        # Decide if you want to proceed without categories or exit
        # return # Option: exit if categories are essential

    img_dir = Path(image_directory)
    if not img_dir.is_dir():
        print(f"Error: Directory not found: {image_directory}")
        return

    print(f"Scanning directory: {img_dir}")

    # --- 1. Find Image Pairs ---
    image_pairs = []
    # Use iglob for potentially large directories (iterator)
    # Sort later based on numeric ID
    potential_gt_files = img_dir.glob('*_gt.*')
    found_pairs_count = 0

    # Store temporarily to sort numerically if possible
    temp_pairs = []

    for gt_path in potential_gt_files:
        base_name = gt_path.name.replace(gt_path.suffix, '').replace('_gt', '')
        pred_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            potential_pred_path = img_dir / f"{base_name}_pred{ext}"
            if potential_pred_path.exists():
                pred_path = potential_pred_path
                break

        if pred_path:
            try:
                numeric_part = int(base_name)
            except ValueError:
                # Handle non-numeric base names if necessary, using the string itself
                # For sorting, this might interleave '1', '10', '2' unless handled carefully later
                numeric_part = base_name
                print(f"  Warning: Base name '{base_name}' is not purely numeric. Sorting may be alphabetical.")

            # Get category, handling potential KeyError
            category_id = _the_dict.get(numeric_part) # Use .get for safer access
            category_name = "Unknown Category" # Default
            if category_id is not None:
                category_name = _name_mapping.get(category_id, f"Unknown Category ID: {category_id}")
            elif numeric_part not in _the_dict:
                 print(f"  Warning: ID '{numeric_part}' (from {base_name}) not found in '{train_file_path}'. Setting category to Unknown.")


            temp_pairs.append({
                'id': numeric_part,
                'cat': category_id, # Can be None
                'cat_name': category_name,
                'base_name': base_name,
                'gt': gt_path,
                'pred': pred_path
            })
            found_pairs_count += 1
            # Defer printing until after sorting if needed, or keep for progress
            # print(f"  Found pair: {gt_path.name} and {pred_path.name}")
        else:
             print(f"  Warning: Found {gt_path.name} but no corresponding prediction file ({base_name}_pred.*).")


    if not temp_pairs:
        print("No complete image pairs (_gt and _pred) found.")
        return

    print(f"Found {found_pairs_count} image pairs.")

    # Sort pairs: Try numeric sort first, fallback to string sort if IDs are mixed/non-numeric
    try:
        # Attempt numeric sort on 'id'
        image_pairs = sorted(temp_pairs, key=lambda x: int(x['id']))
        print("Sorted pairs numerically by ID.")
    except (ValueError, TypeError):
        # If 'id' contains non-integers, sort by 'id' as string
        print("Warning: Could not sort all pairs numerically. Sorting alphabetically by base name.")
        image_pairs = sorted(temp_pairs, key=lambda x: str(x['id']))


    # --- 2. Prepare PDF Document ---
    doc = SimpleDocTemplate(output_pdf_filename)
    styles = getSampleStyleSheet()
    story = []

    title_style = styles['h1']
    title_style.alignment = TA_CENTER
    story.append(Paragraph("Image Comparison Report", title_style))
    story.append(Spacer(1, 0.5*inch))

    # --- 3. Process Each Pair and Add to Story ---
    # Define desired MAX width for each image in the PDF (in inches)
    max_img_width_inches = 2.8
    max_img_width_points = max_img_width_inches * inch # Convert to points for ReportLab

    # Define the target resolution (dots per inch) for the scaled images.
    # Higher DPI means better quality but larger file size. 96 is standard screen DPI.
    # 150 or 200 might be a good compromise for print/zoom quality vs size.
    TARGET_DPI = 150 # Adjust as needed

    for i, pair in enumerate(image_pairs):
        print(f"Processing pair {i+1}/{len(image_pairs)}: {pair['base_name']}")
        gt_path_str = str(pair['gt'])
        pred_path_str = str(pair['pred'])

        try:
            # --- Process Ground Truth Image ---
            with PILImage.open(gt_path_str) as pil_img_gt:
                gt_w, gt_h = pil_img_gt.size
                gt_aspect = gt_h / float(gt_w)

                # Calculate display size in points (limited by max_img_width_points)
                gt_display_width = min(max_img_width_points, gt_w) # Avoid upscaling display beyond original width in points
                gt_display_height = gt_display_width * gt_aspect

                # Calculate the pixel dimensions for resizing based on target DPI
                # Target width in pixels = display width in inches * target DPI
                target_pixel_w = int( (gt_display_width / inch) * TARGET_DPI )
                target_pixel_h = int(target_pixel_w * gt_aspect)

                # Resize using Pillow for high-quality downscaling
                print(f"  Resizing GT '{pair['gt'].name}' from {gt_w}x{gt_h} to {target_pixel_w}x{target_pixel_h} pixels...")
                pil_img_gt_resized = pil_img_gt.resize((target_pixel_w, target_pixel_h), resample=RESAMPLE_FILTER)

                # Save resized image to an in-memory buffer
                gt_buffer = io.BytesIO()
                # Determine format - use original if known and suitable, else default (e.g., PNG)
                img_format = pil_img_gt.format if pil_img_gt.format in ['JPEG', 'PNG'] else 'PNG'
                pil_img_gt_resized.save(gt_buffer, format=img_format, dpi=(TARGET_DPI, TARGET_DPI)) # Include DPI info
                gt_buffer.seek(0) # Rewind buffer

                # Create ReportLab Image from the buffer, using the calculated display size
                img_gt = Image(gt_buffer, width=gt_display_width, height=gt_display_height)
                # Clean up buffer explicitly? Not strictly necessary with 'with' for BytesIO, but good practice if used outside.
                # gt_buffer.close() # Usually handled by garbage collection

            # --- Process Prediction Image (similar logic) ---
            with PILImage.open(pred_path_str) as pil_img_pred:
                pred_w, pred_h = pil_img_pred.size
                pred_aspect = pred_h / float(pred_w)

                pred_display_width = min(max_img_width_points, pred_w)
                pred_display_height = pred_display_width * pred_aspect

                target_pixel_w = int( (pred_display_width / inch) * TARGET_DPI )
                target_pixel_h = int(target_pixel_w * pred_aspect)

                print(f"  Resizing Pred '{pair['pred'].name}' from {pred_w}x{pred_h} to {target_pixel_w}x{target_pixel_h} pixels...")
                pil_img_pred_resized = pil_img_pred.resize((target_pixel_w, target_pixel_h), resample=RESAMPLE_FILTER)

                pred_buffer = io.BytesIO()
                img_format = pil_img_pred.format if pil_img_pred.format in ['JPEG', 'PNG'] else 'PNG'
                pil_img_pred_resized.save(pred_buffer, format=img_format, dpi=(TARGET_DPI, TARGET_DPI))
                pred_buffer.seek(0)

                img_pred = Image(pred_buffer, width=pred_display_width, height=pred_display_height)
                # pred_buffer.close()


            # --- Create Table and Add to Story ---
            image_table_data = [[img_gt, img_pred]]
            # Use the display width for column widths, add padding between cols
            col_padding = 0.1 * inch
            image_table = Table(image_table_data, colWidths=[gt_display_width + col_padding / 2, pred_display_width + col_padding / 2])

            image_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (0, 0), 0),          # Left padding for first image
                ('RIGHTPADDING', (0, 0), (0, 0), col_padding),# Add space after first image
                ('LEFTPADDING', (1, 0), (1, 0), 0),          # Left padding for second image
                ('RIGHTPADDING', (1, 0), (1, 0), 0),         # Right padding for second image
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),      # Space below images before caption
                ('TOPPADDING', (0, 0), (-1, -1), 0),
               # ('GRID', (0,0), (-1,-1), 1, colors.red) # Debugging grid
            ]))

            story.append(image_table)

            # Caption
            caption_text = f"Ground Truth ({pair['cat_name']}) vs. Prediction (right)."
            caption_style = styles['BodyText']
            caption_style.alignment = TA_LEFT
            story.append(Paragraph(caption_text, caption_style))
            story.append(Spacer(1, 0.4*inch)) # Space before next pair

        except FileNotFoundError as fnf_err:
             print(f"  Error: Image file not found during processing: {fnf_err}", file=sys.stderr)
             error_text = f"Error: Could not find image file for item '{pair['base_name']}'. File: {fnf_err.filename}"
             story.append(Paragraph(error_text, styles['Italic']))
             story.append(Spacer(1, 0.4*inch))
        except Exception as e:
            print(f"  Error processing image pair {pair['base_name']}: {e}", file=sys.stderr)
            # Optionally add detailed error to PDF for debugging
            error_text = f"Error processing item '{pair['base_name']}'. Check files: {pair['gt'].name}, {pair['pred'].name}. Error type: {type(e).__name__}"
            story.append(Paragraph(error_text, styles['Italic']))
            story.append(Spacer(1, 0.4*inch))

    # --- 4. Build the PDF ---
    try:
        print(f"\nGenerating PDF: {output_pdf_filename}...")
        doc.build(story)
        print("PDF generation successful!")
    except Exception as e:
        print(f"Error building PDF: {e}", file=sys.stderr)

# --- How to use it ---
if __name__ == "__main__":
    # IMPORTANT: Use raw string (r"...") or forward slashes for Windows paths
    image_folder = r"C:\Users\massoud\PycharmProjects\copied_focal\wsss_resnet\res2" # Or "/path/to/your/images" on Linux/macOS
    train_labels_file = r"C:\Users\massoud\PycharmProjects\copied_focal\wsss_resnet\train.txt" # Make sure this path is correct

    # IMPORTANT: Replace with your desired output PDF file name
    pdf_output_file = "report_4.pdf" # Changed name slightly

    # Check if train file exists before starting
    if not Path(train_labels_file).is_file():
        print(f"Error: Train labels file not found at '{train_labels_file}'", file=sys.stderr)
    else:
        create_image_report(image_folder, pdf_output_file, train_file_path=train_labels_file)
