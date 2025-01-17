# mypy: ignore-errors
from PIL import Image, ImageDraw, ImageFont


def create_sieve_pattern(width, height, dot_radius=5, spacing=20):
    """
    Create an RGBA image (black background, alpha=255)
    with transparent polka-dot holes (alpha=0).
    """
    # Start with a fully opaque black image
    pattern = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(pattern)

    # "Punch out" holes by drawing circles with (0,0,0,0) = transparent
    for y in range(0, height, spacing):
        for x in range(0, width, spacing):
            left = x - dot_radius
            top = y - dot_radius
            right = x + dot_radius
            bottom = y + dot_radius
            draw.ellipse([left, top, right, bottom], fill=(0, 0, 0, 0))

    return pattern


def create_text_mask(text, font_path, font_size, image_size):
    """
    Create a grayscale (L-mode) mask with white text on black background.
    White = 255 => opaque region, black = 0 => transparent region.
    """
    mask_img = Image.new("L", image_size, color=0)  # black by default
    draw = ImageDraw.Draw(mask_img)

    font = ImageFont.truetype(font_path, font_size)

    # Use textbbox in newer Pillow (10.0+), since textsize is deprecated
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x_pos = (image_size[0] - text_width) // 2
    y_pos = (image_size[1] - text_height) // 2

    # White text on black background
    draw.text((x_pos, y_pos), text, fill=255, font=font)

    return mask_img


def create_sieve_text_image(text, font_path, output_path="sieve_text.png"):
    width, height = 800, 400

    # 1) Create the “sieve” pattern (black with transparent holes)
    pattern_img = create_sieve_pattern(width, height, dot_radius=3, spacing=18)

    # 2) Create a text mask (white text on black background, "L" mode)
    text_mask = create_text_mask(text=text, font_path=font_path, font_size=100, image_size=(width, height))

    # 3) Create a transparent canvas
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # 4) Paste the pattern onto the canvas wherever text_mask is non-zero
    # (i.e., where the text is white)
    canvas.paste(pattern_img, (0, 0), text_mask)

    # 5) Save
    canvas.save(output_path, "PNG")
    print(f"Saved sieve-style text with transparent holes to: {output_path}")


# ---------------------------------
# Example usage:
# ---------------------------------
if __name__ == "__main__":
    create_sieve_text_image(
        text="sieves", font_path="/home/raphael/.local/share/fonts/Hack-Bold.ttf", output_path="sieves_sieve_style.png"
    )
